#!/bin/bash
# runone.sh — run a single config's pressure benchmark and append the result to results.tsv.
# Designed for LLM-in-the-loop sweeps: the model decides which config to run next; this
# script only executes one.
#
# env (overridable): IMG MODEL WORK_DIR TTFT_SLO TPOT_SLO PORT CTX_LEN CTR FAKE_KV BENCH_EXTRA
#   WORK_DIR: host dir mounted into the container at /sweep (results land in $WORK_DIR/auto)
#   MODELS_DIR: host dir mounted at /models
# usage: runone.sh <phase P|D> <label> <tp> <pp> <dpa> <ep> <chunk> <mtp> <isl> <osl> <conc_csv> -- <sglang extra args...>
#   Prefill pressure:            isl=ISL osl=1   (radix cache disabled automatically)
#   Decode low-fidelity ranking: isl=128 osl=OSL
#   Decode high-fidelity (official pure-D): FAKE_KV=1 with isl=ISL osl=OSL — the script
#     adds the fake transfer backend server-side and injects the fake bootstrap fields
#     into each request. KV is allocated per request at full ISL (garbage contents,
#     correct performance semantics); prefill compute is skipped. Judge TPOT only.
set -u
IMG=${IMG:-lmsysorg/sglang:latest}
CTR=${CTR:-sgl_sweep_auto}
MODEL=${MODEL:?set MODEL to the model path inside the container}
WORK_DIR=${WORK_DIR:?set WORK_DIR to the host sweep dir (mounted at /sweep)}
MODELS_DIR=${MODELS_DIR:-$(dirname "$WORK_DIR")/models}
PORT=${PORT:-30000}
CTX_LEN=${CTX_LEN:-16384}
TTFT_SLO=${TTFT_SLO:-500}; TPOT_SLO=${TPOT_SLO:-15}
BENCH_EXTRA=${BENCH_EXTRA:-}
A=$WORK_DIR/auto      # host-side result paths (this script runs on the host)
CA=/sweep/auto        # container-side paths (for logs written via docker exec)
mkdir -p $A/raw

phase=$1; label=$2; tp=$3; pp=$4; dpa=$5; ep=$6; chunk=$7; mtp=$8; isl=$9; osl=${10}; concs=${11}
shift 11; [ "${1:-}" = "--" ] && shift; extra="$*"
# Prefill pressure must disable the radix cache: concurrent equal-length prompts hit the
# prefix cache and produce fantasy-low TTFT.
[ "$phase" = "P" ] && extra="$extra --disable-radix-cache"
# FAKE_KV=1: official pure-decode mode. "2.2.2.2" is FAKE_BOOTSTRAP_HOST in the source;
# FakeKVReceiver.poll() returns Success immediately -> no prefill, full-ISL KV allocation.
if [ "${FAKE_KV:-0}" = "1" ]; then
  extra="$extra --disaggregation-mode decode --disaggregation-transfer-backend fake"
  # single-quote the JSON: prevents the inner bash from brace-expanding {a,b}
  BENCH_EXTRA="--extra-request-body '{\"bootstrap_host\":\"2.2.2.2\",\"bootstrap_room\":0}'"
fi

log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $A/driver.log; }

[ -f $A/results.tsv ] || echo -e "phase\tlabel\ttp\tpp\tdpa\tep\tchunk\tmtp\tconc\tttft_p50\ttpot_mean\treq_tps\tout_tps\tinp_tps\tstatus\tverdict" > $A/results.tsv

# Container (created on first use; image digest recorded)
if ! docker ps --filter name=$CTR --format '{{.Names}}' | grep -q $CTR; then
  docker run -d --name $CTR --gpus all --network host --ipc host --shm-size 32g \
    -v $MODELS_DIR:/models -v $WORK_DIR:/sweep $IMG sleep infinity >/dev/null 2>&1
  docker images --digests --format '{{.Repository}}:{{.Tag}} {{.Digest}}' | grep -m1 "${IMG%%:*}" >> $A/image_used.txt
fi

# Kill ALL sglang processes (schedulers included) and wait for VRAM to drain before the
# next server, or leftovers cause phantom OOM / instant exits.
docker exec $CTR pkill -9 -f sglang 2>/dev/null; sleep 8
for i in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
  [ "${used:-99999}" -lt 2000 ] && break || sleep 2
done
# NOTE: the server log must use the container-side path $CA, not the host path $A.
docker exec $CTR bash -c "cd / && nohup python3 -m sglang.launch_server \
  --model-path $MODEL --tokenizer-path $MODEL \
  --trust-remote-code --context-length $CTX_LEN --host 127.0.0.1 --port $PORT $extra > $CA/server_${label}.log 2>&1 &"
log "starting server: $label ($extra)"
# Liveness: poll /health while scanning the log for fatal errors (fail fast). PP cold
# start is slow — allow up to 600s.
ok=0
for i in $(seq 1 300); do
  docker exec $CTR curl -s -m 3 127.0.0.1:$PORT/health >/dev/null 2>&1 && { ok=1; break; }
  if docker exec $CTR bash -c "grep -qiE 'Traceback|CUDA out of memory|RuntimeError|AssertionError|ValueError|address already in use|NCCL error' $CA/server_${label}.log 2>/dev/null"; then
    log "!! $label FATAL in log:"; docker exec $CTR bash -c "grep -iE 'Traceback|out of memory|Error|assert|already in use' $CA/server_${label}.log|tail -8"|sed 's/^/   /'|tee -a $A/driver.log
    echo -e "$phase\t$label\t$tp\t$pp\t$dpa\t$ep\t$chunk\t$mtp\tNA\tNA\tNA\tNA\tNA\tNA\tSRVFAIL\t-" >> $A/results.tsv; exit 2
  fi
  sleep 2
done
[ $ok -eq 1 ] || { log "!! $label TIMEOUT"; echo -e "$phase\t$label\t$tp\t$pp\t$dpa\t$ep\t$chunk\t$mtp\tNA\tNA\tNA\tNA\tNA\tNA\tSRVFAIL\t-" >> $A/results.tsv; exit 2; }
log "UP ($label)"

# Concurrency ladder; stop on FAIL (monotonicity) or throughput plateau.
for c in $(echo $concs | tr ',' ' '); do
  raw=$CA/raw/${label}_c${c}_$(date +%m%d%H%M).txt
  np=$((c*2)); [ "$phase" = "P" ] && np=$((c*3)); [ $np -lt 8 ] && np=8
  docker exec $CTR bash -c "cd / && timeout 900 python3 -m sglang.bench_serving \
    --backend sglang --host 127.0.0.1 --port $PORT --model $MODEL --dataset-name random \
    --random-input-len $isl --random-output-len $osl --random-range-ratio 1 \
    --num-prompts $np --max-concurrency $c --request-rate inf $BENCH_EXTRA > $raw 2>&1"
  ttft=$(docker exec $CTR bash -c "grep -iE 'Median TTFT' $raw|grep -oE '[0-9]+\.[0-9]+'|head -1")
  tpot=$(docker exec $CTR bash -c "grep -iE 'Median ITL' $raw|grep -oE '[0-9]+\.[0-9]+'|head -1")
  rtps=$(docker exec $CTR bash -c "grep -iE 'Request throughput' $raw|grep -oE '[0-9]+\.[0-9]+'|head -1")
  otps=$(docker exec $CTR bash -c "grep -iE 'Output token throughput' $raw|grep -oE '[0-9]+\.[0-9]+'|head -1")
  itps=$(docker exec $CTR bash -c "grep -iE 'Input token throughput' $raw|grep -oE '[0-9]+\.[0-9]+'|head -1")
  ttft=${ttft:-99999}; tpot=${tpot:-99999}; rtps=${rtps:-NA}; otps=${otps:-NA}; itps=${itps:-NA}
  verdict="PASS"
  if [ "$phase" = "P" ]; then awk "BEGIN{exit !($ttft<=$TTFT_SLO)}" || verdict="FAIL"
  else awk "BEGIN{exit !($tpot<=$TPOT_SLO)}" || verdict="FAIL"; fi
  echo -e "$phase\t$label\t$tp\t$pp\t$dpa\t$ep\t$chunk\t$mtp\t$c\t$ttft\t$tpot\t$rtps\t$otps\t$itps\tDONE\t$verdict" >> $A/results.tsv
  log "  $label c=$c -> TTFT=$ttft TPOT=$tpot inpTPS=$itps outTPS=$otps [$verdict]"
  [ "$verdict" = "FAIL" ] && { log "  stop climbing $label at c=$c"; break; }
  # Decode plateau stop: TPOT still passing but outTPS gained <5% means the instance is
  # saturated (extra concurrency becomes queueing) — climbing further is wasted time.
  if [ "$phase" = "D" ] && [ "${prev_otps:-}" != "" ] && [ "$otps" != "NA" ]; then
    awk "BEGIN{exit !($otps < $prev_otps*1.05)}" && { log "  plateau: stop climbing $label at c=$c (outTPS +<5%)"; break; }
  fi
  prev_otps=$otps
done
docker exec $CTR pkill -9 -f sglang 2>/dev/null
log "done $label"
