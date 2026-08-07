#!/bin/bash
# runone.sh — 执行单个 config 的单压 benchmark, 结果追加到 results.tsv
# 由模型(LLM)在环调用: 模型根据已有数据决定跑哪个 config, 本脚本只负责执行.
# 部署位置: $WORK_DIR/runone.sh (容器内挂载为 /sweep/runone.sh); 站点默认值见 references/site.md
#
# env(可覆盖): IMG MODEL WORK_DIR MODELS_DIR TTFT_SLO TPOT_SLO PORT CTX_LEN BENCH_EXTRA QUANT_ARGS
# usage: runone.sh <phase P|D> <label> <tp> <pp> <dpa> <ep> <chunk> <mtp> <isl> <osl> <conc_csv> -- <sglang extra args...>
#   P 单压约定: isl=ISL osl=1  (纯 prefill, 自动加 --disable-radix-cache)
#   D 单压(一律官方 pure-D 高保真): FAKE_KV=1 runone.sh D <label> ... ISL OSL ... ,
#     脚本自动给 server 加 fake 传输后端、给 bench 注入 FAKE_BOOTSTRAP_HOST 请求体,
#     每请求按真实 ISL 独立分配 KV(内容垃圾/性能语义正确), 跳过 prefill 计算. 只看 TPOT.
set -u
IMG=${IMG:-lmsysorg/sglang:latest}
CTR=${CTR:-sgl_sweep_auto}
MODEL=${MODEL:?set MODEL (container path, see site.md)}
WORK_DIR=${WORK_DIR:?set WORK_DIR (host sweep dir, see site.md)}
MODELS_DIR=${MODELS_DIR:?set MODELS_DIR (host models dir, see site.md)}
PORT=${PORT:-30000}
CTX_LEN=${CTX_LEN:-16384}
TTFT_SLO=${TTFT_SLO:-500}; TPOT_SLO=${TPOT_SLO:-15}
BENCH_EXTRA=${BENCH_EXTRA:-}
# 量化按模型传入: nvfp4 checkpoint 用 "--quantization modelopt_fp4 --kv-cache-dtype fp8_e4m3",
# fp8/bf16 checkpoint 通常留空让 sglang 从 config 自检(可选 --kv-cache-dtype fp8_e4m3 省 KV)
QUANT_ARGS=${QUANT_ARGS:-}
A=$WORK_DIR/auto                   # 宿主机路径(本脚本在宿主机跑, 读写结果用这个)
CA=/sweep/auto                     # 容器内路径(docker exec 内部写日志; 挂载 $WORK_DIR -> /sweep)
mkdir -p $A/raw

phase=$1; label=$2; tp=$3; pp=$4; dpa=$5; ep=$6; chunk=$7; mtp=$8; isl=$9; osl=${10}; concs=${11}
shift 11; [ "${1:-}" = "--" ] && shift; extra="$*"
# prefill 单压必须禁 radix cache: 等长 random prompt 并发命中 prefix cache -> 产出假的超低 TTFT
[ "$phase" = "P" ] && extra="$extra --disable-radix-cache"
# FAKE_KV=1: 官方 pure-D 模式. server 走 fake KV 后端; bench 注入 fake bootstrap 字段
# ("2.2.2.2" = 源码 FAKE_BOOTSTRAP_HOST, FakeKVReceiver.poll 直接 Success -> 零 prefill, KV 足额分配)
if [ "${FAKE_KV:-0}" = "1" ]; then
  extra="$extra --disaggregation-mode decode --disaggregation-transfer-backend fake"
  # 单引号包 JSON: 防内层 bash 对 {a,b} 做 brace expansion 劈开参数
  BENCH_EXTRA="--extra-request-body '{\"bootstrap_host\":\"2.2.2.2\",\"bootstrap_room\":0}'"
fi

log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $A/driver.log; }

[ -f $A/results.tsv ] || echo -e "phase\tlabel\ttp\tpp\tdpa\tep\tchunk\tmtp\tconc\tttft_p50\ttpot_mean\treq_tps\tout_tps\tinp_tps\tstatus\tverdict" > $A/results.tsv

# 容器(首次自动创建, 记录镜像digest)
if ! docker ps --filter name=$CTR --format '{{.Names}}' | grep -q $CTR; then
  docker run -d --name $CTR --gpus all --network host --ipc host --shm-size 32g \
    -v $MODELS_DIR:/models -v $WORK_DIR:/sweep $IMG sleep infinity >/dev/null 2>&1
  docker images --digests --format '{{.Repository}}:{{.Tag}} {{.Digest}}' | grep -m1 "${IMG%%:*}" >> $A/image_used.txt
fi

# 起 server 前必须杀全部 sglang(含 scheduler 子进程), 并等显存真正释放, 否则残留致假 OOM/秒退
docker exec $CTR pkill -9 -f sglang 2>/dev/null; sleep 8
for i in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
  [ "${used:-99999}" -lt 2000 ] && break || sleep 2
done
# 注意: server.log 必须用容器内路径 $CA, 用宿主机 $A 重定向失败 server 起不来!
docker exec $CTR bash -c "cd / && nohup python3 -m sglang.launch_server \
  --model-path $MODEL --tokenizer-path $MODEL $QUANT_ARGS \
  --trust-remote-code --context-length $CTX_LEN --host 127.0.0.1 --port $PORT $extra > $CA/server_${label}.log 2>&1 &"
log "starting server: $label ($extra)"
# 判活: health 轮询 + 扫日志抓致命错误快速失败. PP 冷启动 35s+, 给足 600s.
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

# 并发梯队逐档跑, FAIL 即停爬(单调假设)
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
  # D 阶段吞吐平台期即停: TPOT 不 FAIL 但 outTPS 增益 <5% 说明已饱和(纯排队), 再爬无意义
  if [ "$phase" = "D" ] && [ "${prev_otps:-}" != "" ] && [ "$otps" != "NA" ]; then
    awk "BEGIN{exit !($otps < $prev_otps*1.05)}" && { log "  plateau: stop climbing $label at c=$c (outTPS +<5%)"; break; }
  fi
  prev_otps=$otps
done
docker exec $CTR pkill -9 -f sglang 2>/dev/null
log "done $label"
