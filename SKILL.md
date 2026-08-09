---
name: pro5k-sgl-sweep
description: Fully-automatic SGLang inference performance sweep on the pro5k server (8x RTX Pro Blackwell 73GB, PCIe-only, no NVLink). Given ANY SLA (TTFT/TPOT) and workload (ISL/OSL), sweeps prefill/decode parallel configs (TP/PP/DPA/EP/MTP/chunk), picks the best P and D configs, does QPS matching for the PD ratio, and reports max throughput per 1000 GPUs. Use when asked to tune, sweep, or optimize SGLang serving performance on pro5k, or to find the best config under an SLA.
---

# SGLang PD parallel-strategy auto-sweep

Goal: given any SLA (TTFT / TPOT) + workload profile (ISL / OSL) + model, automatically
find the SLA-compliant throughput-optimal Prefill config and Decode config, then do
**QPS matching** to pick the PD ratio and report **QPS per 1000 GPUs**
(= 1000 × system QPS ÷ total GPUs).

## 0. Inputs (confirm all before running)

| Param | Meaning | Default |
|---|---|---|
| `TTFT_SLO` | prefill SLA, P50 TTFT ceiling (ms) | 500 |
| `TPOT_SLO` | decode SLA, mean TPOT/ITL ceiling (ms) | 15 |
| `ISL` / `OSL` | input/output length profile | 10000 / 700 |
| `MODEL` | model path inside the container | see references/site.md |
| `IMG` | SGLang image; **pull latest by default** and record the resolved tag+digest in the results | latest |

If the user only gives an SLA, use defaults and say so explicitly. The execution channel
(SSH/MCP tool), host and container paths, container naming, and model locations are
**all in `references/site.md` (private file, never published with the methodology)**.

## 0.5 Sweep Plan (present before every run; user confirms before execution)

Before any sweep action (including follow-up measurements), produce a **Sweep Plan**:

1. **Environment snapshot**: GPU idleness (`nvidia-smi`), image tag+digest to be used,
   model path, host disk headroom.
2. **Parameter echo**: SLA (TTFT/TPOT), ISL/OSL, criterion (P50 vs P90).
3. **Candidate table**: P-phase and D-phase config lists — label, parallel combo,
   chunk/MTP, concurrency ladder, and **the derivation behind each entry** (§3.5 rules
   or hardware reasoning).
4. **Execution order & decision points**: which configs run unconditionally, which
   depend on prior results; pruning rules written out.
5. **Budget**: per-config time estimate (cold start 1-6 min + 2-10 min per concurrency
   rung), total ceiling; which low-priority configs get cut first when over budget.
6. **Artifacts**: results.tsv path, report filename (timestamped).

Present the plan as a compact table. Execute only after the user confirms (or amends).
Flag any deviation during execution (added/skipped configs) with the reason. If a plan
assumption breaks mid-run (image behavior change, OOM boundary moved), stop and revise
the plan — don't push through silently.

## 1. Hardware priors (prune the search space; see references/hardware.md)

- **PCIe-only box, no NVLink**: GPUs are connected only via PCIe (PXB/NODE within a
  NUMA node, SYS across) + RDMA NICs (GPU count / memory / topology in site.md and
  hardware.md).
- **Large TP is poison**: TP's per-layer all-reduce rides PCIe. Decode is memory-bound
  with per-token all-reduce — the larger the TP, the higher the communication share;
  big TP collapses per-GPU decode throughput catastrophically. Keep TP within one NUMA
  node's GPU set.
- Parallelism that crosses the NUMA boundary must be PP (activations only) or DPA/EP
  (communication amenable to RDMA/A2A).

## 2. Knobs available to P / D (verified against server_args.py)

`tp_size × pp_size = GPUs per unit`. Hard constraints (asserted in source):
- DPA: `--enable-dp-attention --dp-size N`, requires **tp % dp == 0**;
  chunked_prefill_size is divided by dp_size internally.
- EP: `--ep-size N`, requires **ep × moe_dp_size == tp** (moe_dp defaults to 1, so
  usually ep == tp, or configure `--moe-dp-size`).
- FlashInfer A2A: `--moe-a2a-backend flashinfer` is only legal with **dp == tp and DPA
  on**.
- A2A backend choices: `none, deepep, mooncake, nixl, mori, flashinfer, megamoe`.
- Decode-side radix cache (PD mode `--disaggregation-decode-enable-radix-cache`) is
  **incompatible with speculative decoding** — never combine it with MTP.
- PP is incompatible with context parallelism / elastic EP.

| Knob | flag | P | D |
|---|---|---|---|
| TP / PP | `--tp-size N --pp-size M` | ✅ PP is the prefill lever | PP hurts TPOT; rarely for D |
| chunk | `--chunked-prefill-size N` | ✅ core knob | n/a |
| DPA | `--enable-dp-attention --dp-size N` | little gain at long ISL | ✅ key decode-throughput lever |
| EP | `--ep-size N` | usually no prefill gain | ✅ spreads MoE experts |
| MTP | `--speculative-algorithm NEXTN --speculative-num-steps a --speculative-eagle-topk b --speculative-num-draft-tokens c` | n/a | ✅ top lever; depth has a sweet spot |
| A2A | `--moe-a2a-backend flashinfer` | – | compare once when EP>1 |

MTP notation a/b/c = steps/topk/draft-tokens, e.g. 3/1/4. If a new image rejects
`NEXTN` with "Unknown speculative algorithm" (it's plugin-registered), fall back to
`EAGLE`. Notes: on 0.5.16, NEXTN with `--speculative-eagle-topk > 1` crashes at server
start (dimension closed); `--mem-fraction-static` is a formal decode coordinate
(default → 0.95 as one step; a larger KV pool can push the saturation point up; too
high OOMs during load — SRVFAIL marks the boundary).

## 3. Component pressure testing (P and D measured separately; inputs to QPS matching)

Use `sglang.bench_serving --dataset-name random --random-range-ratio 1` against a
single instance:

- **P pressure**: `isl=ISL, osl=1` (pure prefill). Always `--disable-radix-cache`
  (concurrent equal-length random prompts hit the prefix cache and produce fantasy-low
  TTFT). Metrics: **P50 TTFT judges the SLA, Input TPS counts throughput**.
  P pressure is inherently high fidelity (prefill is the entire workload).
- **D pressure: exclusively the official pure-D fake-KV path (fully high fidelity;
  verified end-to-end on 0.5.15+)**:
  - D server adds `--disaggregation-mode decode --disaggregation-transfer-backend fake`
    on top of its normal decode flags;
  - bench_serving uses the real workload lengths `--random-input-len ISL
    --random-output-len OSL` plus
    `--extra-request-body '{"bootstrap_host":"2.2.2.2","bootstrap_room":0}'`
    ("2.2.2.2" = `FAKE_BOOTSTRAP_HOST` in the source; FakeKVReceiver's poll returns
    Success immediately).
  - Effect: prefill compute is skipped entirely, yet each request **allocates its own
    KV at the full ISL** (garbage contents, correct performance semantics) — every
    decode step reads true-length KV, KV memory is fully charged, and the concurrency
    ceiling is real. Upstream main wraps the same mechanism as
    `sglang.benchmark.one_batch_server --fake-prefill`.
  - Caveats: TTFT is meaningless in this mode (only TPOT / Output TPS count); garbage
    KV means **MTP acceptance may differ from real text** — sanity-check the gain
    multiple against the same-mode MTP-off baseline, and re-verify the final pick in a
    real PD setup; the server auto-skips radix insert for fake requests, no need to
    disable radix.
  - Fallback (only if the fake path breaks): `--dataset-name generated-shared-prefix`
    (gsp-system-prompt-len=ISL + radix on). Downside: the shared prefix is stored once,
    so memory footprint / concurrency ceiling read optimistic.
  - **Short-input proxies are banned** (isl=128 and the like): each decode step reads a
    sliver of KV, which grossly overestimates D capacity and can distort even relative
    rankings (memory-bound gains like DPA get underestimated) — removed from the flow.
- Metrics: **mean TPOT/ITL judges the SLA, Output TPS counts throughput**. Climb the
  concurrency ladder with two stop conditions: ① stop on FAIL (monotonicity
  assumption); ② **stop on throughput plateau** — TPOT still passing but outTPS gaining
  <5% means the instance is saturated and extra concurrency is pure queueing (a
  saturated instance keeps "passing" TPOT while the queueing piles into TTFT). The
  highest-throughput PASS rung is the config's "operating point" and feeds QPS matching
  directly.

Executor `scripts/runone.sh` (authoritative copy lives in the skill; before running,
write it to `$WORK_DIR/runone.sh` per site.md and `chmod +x`; export WORK_DIR /
MODELS_DIR from site.md):

```
env: IMG MODEL WORK_DIR MODELS_DIR TTFT_SLO TPOT_SLO [PORT CTX_LEN QUANT_ARGS]
usage: runone.sh <P|D> <label> <tp> <pp> <dpa> <ep> <chunk> <mtp> <isl> <osl> <conc_csv> -- <sglang extra args>
```

It handles: container lifecycle, killing stale sglang processes, waiting for VRAM to
drain, server start (fatal-log fast-fail), per-rung benchmarking, PASS/FAIL verdicts,
appending `auto/results.tsv`. The tp/pp/dpa/ep/chunk/mtp arguments are record-keeping
labels only — what actually takes effect are the sglang flags after `--`; keep them
consistent.

## 3.5 Seed derivation for a new model (conclusions are per-model; re-derive every time)

Tuning conclusions are **per-model** (they encode weight size, layer count, MoE
structure, MTP-head quality). For a new model, derive seeds from first principles:

1. **Memory feasibility**: W = weight bytes (nvfp4 ≈ params × 0.55); per-GPU memory
   GPU_MEM (see site.md). Min unit = ceil(W / (0.8 × GPU_MEM)) GPUs (reserve KV +
   activations); **KV headroom = GPU_MEM × GPUs − W** bounds decode concurrency;
   shapes with <20 GB/GPU headroom will saturate in decode — prune outright.
2. **P seeds**: pure PP first (platform law). PP depth candidates = the {shallow, mid,
   deep} feasible tiers (if the model fits one GPU, start at TP1/PP1 and scale P by
   replicas instead of PP depth); chunk starting point ≈ SLO_ms × estimated per-GPU
   prefill TPS / 1000 / PP-depth, then hill-climb.
3. **D seeds**: unit size = smallest with "weights fit + KV headroom ≥30 GB/GPU"; MoE →
   EP=TP plus a DPA=2 contrast; dense → DPA is the only lever. MTP from 3/1/4 when the
   model ships a draft head, else off (NGRAM as fallback).
4. Then run the same search algorithm (§4) — **the methodology never changes, only the
   seeds do**. Same-hardware cross-model laws (PP>TP for P, low-TP+DPA+EP for D, big-TP
   poison) carry over; specific numbers do not.

## 4. Search: coordinate descent + global hardening (LLM in the loop, rules explicit)

Don't run a fixed candidate table; advance point by point, reading results.tsv after
each measurement:

```
1. Seeds:    2-4 per phase, derived by §3.5 from model properties + hardware first principles
2. Screen:   quick short-ladder pass (D also via fake pure-D, 1-2 rungs only); drop
             clear losers (per-GPU throughput < 60% of leader); keep top-K (K≈2)
3. Descend:  hill-climb the champion one dimension at a time, in sensitivity order:
             P: pp depth → chunk → concurrency    D: mtp depth → dpa → ep → mem-fraction → concurrency
             ±1 step along one dim; improvement (per-GPU ↑ and PASS) → same direction;
             regression → next dimension; a full loop with no improvement → local optimum, stop
4. Boundary: champion sitting on any scanned edge → extend one step (anti-miss);
             generic convergence: consecutive gains but <5% per step → converged, stop
             extending (log as inference-based pruning)
5. Finals:   full ladders on the top-2 to fix operating points + marginal points
             (|metric-SLO| < 5%) re-measured 3× (median), including adjacent rungs —
             the operating point may shift to a neighbor
6. Budget:   short ladder ~5-10 min/point, full ladder ~15-25 min/point; past half
             budget, shrink steps and only advance the champion branch
```

This is successive halving + coordinate descent: short ladders for cheap screening,
full ladders for finals, one measurement standard throughout (fake pure-D); coordinate
descent measures O(dims × steps) points instead of the Cartesian product; boundary
extension guards against over-pruned priors.

### Global hardening (mandatory after descent converges; budget ≈ 50% of main search)

Coordinate descent only certifies "per-dimension local optimality"; the global optimum
can hide in dimension interactions, unswept knobs, noise, and narrow peaks. Four
tightening steps, by cost-effectiveness:

```
G1 Interaction probes: 2-3 points around the champion where TWO dims move together
             (diagonals single-dim climbing can't see), e.g. chunk↑×conc↑, MTP-depth×conc.
             Any probe wins → restart descent from it.
G2 Untouched-knob roll call: try one step of every §2 knob the champion never moved
             (MTP topk>1, attention backend, mem-fraction↑, A2A backend…). One shot
             each; SRVFAIL is information too.
G3 Far-point ε-exploration: 1-2 shapes structurally different from the champion
             (different unit size / different parallel family) — guards against the
             seed locking onto the wrong shape family.
G4 Fine-grid audit: half-step densification around the champion (e.g. conc ±8, chunk
             midpoints) to confirm it isn't a narrow-peak edge; when top-2 are within
             5% (noise scale), re-measure both once more before ranking.
```

Scheduling note: G1 and G4 probes are defined relative to the finalized champion's
coordinates — they MUST wait for descent to converge (probing a moving target wastes
points; a champion shift invalidates them). G3 far points and champion-independent G2
probes (new shape families, version/backend roll calls) may be folded into earlier
batches to save wall-clock when the machine window is tight.

Reports gain a **confidence section**: ① local-optimality certificates (interior /
converged evidence per dimension); ② G1-G4 outcomes; ③ **roofline upper-bound
comparison** — estimate utilization from the model's activated FLOPs vs hardware peak
(state assumptions). High utilization mathematically bounds the remaining global
headroom; low utilization means the big wins live outside the flag space
(kernels/versions). State honestly that global optimality is unprovable — report
known-best + a list of where the remainder could hide.

### Phase P — Prefill (judge TTFT_SLO, rank by Input TPS/GPU)
1. Seeds from §3.5: pure PP first, {shallow, mid, deep} feasible depths; single-GPU
   models start at TP1/PP1 (scale by replicas). Add 1-2 TP-bearing shapes as contrast.
2. Chunk hill-climb: start mid (~2048 or the SLO-derived point), one step each way;
   the sweet spot varies by PP depth, and at higher concurrency a larger chunk can
   even lower TTFT — let data decide, don't presume direction.
3. Concurrency low rungs (1,2,3,4); find the TTFT knee.
4. Prune: no cross-NUMA large TP; EP rarely helps prefill — defer when budget is tight.
5. **KV-margin candidate**: additionally keep one P operating point whose TTFT margin
   ≥ (KV-transfer residual + jitter), for the conservative deployment option.

### Phase D — Decode (judge TPOT_SLO, rank by Output TPS/GPU)
1. Seeds from §3.5: smallest unit with "weights fit + KV headroom ≥30 GB/GPU";
   MoE → EP=TP plus DPA=2 contrast; dense → DPA only.
2. MTP depth scan (when the model has a draft head): from 3/1/4 climb both ways
   (1/1/2 and 4/1/5); stop when TPOT rises (acceptance collapse); no head → off,
   NGRAM as fallback.
3. Concurrency ladder 32,48,64,96,128…; stop on FAIL or plateau (+<5%).
4. If an EP>1 shape wins, try FlashInfer A2A once (needs dp==tp + DPA). Keep DP-MLP
   at its default skip; never enable strong sync.
5. ⚠️ In colocated (non-disagg) mode, the DPA + MTP + chunked-prefill joint path has
   crashed at kernel level (Triton illegal memory access) on some versions; the fake
   pure-D mode has no such path. Single-point-verify before scanning that combo
   colocated.

### Anti-miss mechanisms (run at the end of every phase)

Prior-based pruning can cut the true optimum. Two backstops:

1. **Boundary detection**: if the winner sits on a **scanned boundary** (deepest PP,
   largest EP, deepest MTP, highest passing rung with nothing above it), the true
   optimum may lie outside — extend one step along that dimension until the winner is
   interior.
2. **Marginal re-measurement**: rows with |metric − SLO| < 5%×SLO can't be trusted from
   one shot (e.g. TPOT in 14.25-15.75 ms at a 15 ms SLO) — re-run 3×, take the median;
   same when two candidates differ by <5% in throughput.

Secondary dimensions accepted as unswept (must be listed in the report's coverage
statement): MTP topk>1 where supported, deepep/megamoe A2A, `enable_dynamic_chunking`,
moe_dp_size combos, full chunk×PP interaction (only the winner gets chunk tuning),
mem-fraction/scheduler params beyond the formal coordinate. Pruning has two grades —
**data-backed** (a measured FAIL, safe) and **inference-based** (extrapolation — state
the basis in the report).

### Phase R — QPS matching → PD ratio → QPS per 1k GPUs (`scripts/analyze.py`)
```
python3 analyze.py auto/results.tsv --isl ISL --osl OSL
```
- Each PASS config at its operating point: P-unit QPS = InputTPS/ISL, D-unit QPS =
  OutputTPS/OSL.
- Enumerate xP:yD (reduced ratios, x,y ≤ 8): system QPS = min(x·qps_P, y·qps_D);
  per-1k-GPU QPS = 1000 × QPS ÷ (x·gpus_P + y·gpus_D).
- Output top-3 combos with the bottleneck side (P-bound / D-bound). This is a
  steady-state approximation; KV transfer is not included.
- **Deployability constraint**: a ratio is only as good as its tiling onto real
  nodes. Prefer groups whose GPU count **equals or divides the node size** (e.g. 8) —
  a node-exact group (like 5P:3D = 8 GPUs) is a self-contained PD box: KV transfer
  stays intra-node, one router per node, homogeneous ops. Ratios that don't tile
  (e.g. 13 GPUs/group) deploy as **homogeneous node pools** at fleet level (whole
  nodes of P replicas : whole nodes of D replicas at the target ratio) — never split
  a group across nodes.
- **Replica-count caveat**: many single-GPU replicas per node multiply host-side
  overhead (per-server tokenizer/scheduler processes, PCIe/memory-bandwidth sharing).
  Pin each replica's CPUs to its GPU's NUMA node and give each its NUMA-local NIC.
  Component numbers assume no cross-replica interference — **validate with an
  all-replicas-loaded run before quoting production capacity** (expect a few % loss).
- **The report must offer three options**: ① steady-state optimum; ② **node-aligned
  ops pick** — the best ratio whose group tiles the node exactly (prefer it when
  within ~5%); ③ **KV-margin conservative pick** — swap in the P operating point with
  TTFT headroom (real PD adds tens of ms of KV-transfer residual to TTFT — measure
  it; borderline P configs will violate the SLA in production).

### Phase V — (optional) real PD-disaggregation validation
A single 8-GPU box can host e.g. 1P(4)+1D(4) to measure KV-transfer overhead:
- P server: `--disaggregation-mode prefill --disaggregation-transfer-backend nixl`
  (or mooncake/mooncake_tcp) + `--disaggregation-ib-device <NIC on the same NUMA>`;
  D server: `--disaggregation-mode decode ...`.
- Routing: `python -m sglang_router.launch_router --pd-disaggregation --prefill
  http://127.0.0.1:P1 [bootstrap_port] --decode http://127.0.0.1:P2`, then drive the
  router with a Poisson load (`--request-rate` ≈ 90% of the system QPS from
  analyze.py).
- Judge the final validation on **P90**: closed-loop → Poisson typically inflates tail
  latency substantially; plan production capacity on P90.

## 5. Reporting

- results.tsv → HTML report (**white background**), timestamped filename
  `<name>-sweep-YYYYMMDD-HHMM.html`.
- Headline: best P config, best D config, PD ratio, per-1k-GPU QPS (three options);
  full result table (PASS/FAIL colored), image tag+digest, SLA parameters, confidence
  section, coverage statement.
- FAILs that miss by <3 ms are borderline — Poisson P90 will amplify them; leave
  margin when selecting.

## 6. Operational red lines (every one learned the hard way)

1. **Kill clean between configs**: `pkill -9 -f sglang` inside the container, then poll
   `nvidia-smi` until VRAM < 2 GB before the next server (PP-mode schedulers release
   slowly; leftovers = phantom OOM).
2. **Server log must use the container-side path**; redirecting to a host path fails
   silently and the server never comes up.
3. **Never wait blind**: poll `/health` while grepping the server log for
   `Traceback|OOM|NCCL error` — judge SRVFAIL immediately on a hit; investigate any
   wait > 1 minute.
4. PP cold start is slow (35 s+); give health checks up to 600 s.
5. Timestamp every log/result filename; prefix output lines with `[HH:MM:SS]` to avoid
   reading stale data.
6. `docker pull $IMG` at start and record the resolved digest; after any image change,
   smoke-test one known config before trusting cross-campaign comparisons.
7. Shared machine: confirm no one else's job via `nvidia-smi` first. Other users'
   automated pipelines may **chain-queue** campaigns (the next starts the moment one
   ends) — a gap between campaigns ≠ idle; wait until their scheduler/driver processes
   are all gone, or both sides' data get polluted.
8. `IMG=latest` is a rolling tag: record
   `python3 -c "import sglang; print(sglang.__version__)"` + digest at start; check
   versions before comparing across campaigns.
9. Background tasks launched over the MCP SSH channel must be wrapped in a subshell:
   `( setsid nohup ... & )` — otherwise the channel hangs 120 s and reports a fake
   failure (the task actually runs; verify via a fresh connection).
10. Don't build "wait for the previous batch" watchers on `pgrep -f <batch-name>` —
    the watcher's own command line contains the name and self-matches. Poll the
    BATCH_DONE marker in driver.log instead.
11. Methodology generalization is validated: the same seed-derivation + search rules
    converge to the correct shape family in both the "multi-GPU split regime" (large
    models, PP/EP) and the "single-GPU replica regime" (small models, TP1/PP1) —
    switching models re-derives seeds only, never the rules.
