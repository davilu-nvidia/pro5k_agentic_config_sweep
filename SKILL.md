---
name: sglang-pd-sweep
description: Automatically sweep SGLang prefill/decode parallel configs (TP/PP/DPA/EP/MTP/chunk) under any SLA (TTFT/TPOT) and workload (ISL/OSL) on PCIe-only multi-GPU nodes, then QPS-match prefill and decode units to pick the PD ratio and report max throughput per 1000 GPUs. Use when asked to tune or optimize SGLang serving performance or find the best config under an SLA.
---

# SGLang PD parallel-strategy auto-sweep

Goal: given any SLA (TTFT / TPOT) + workload profile (ISL / OSL) + model, automatically
find the SLA-compliant throughput-optimal prefill config and decode config, QPS-match
them into a PD ratio, and report **QPS per 1000 GPUs**.

## 0. Inputs (confirm before running)

| Param | Meaning | Typical |
|---|---|---|
| `TTFT_SLO` | prefill SLA, P50 TTFT ceiling (ms) | 500 |
| `TPOT_SLO` | decode SLA, mean TPOT/ITL ceiling (ms) | 15 |
| `ISL` / `OSL` | input/output length profile | 10000 / 700 |
| `MODEL` | model path inside the container | — |
| `IMG` | SGLang image; record the resolved version + digest (`latest` is a rolling tag) | latest |

## 0.5 Sweep plan (present for confirmation before every run)

Before any sweep action, produce a plan covering: environment snapshot (GPU idleness,
image tag+digest, model, disk), SLA echo, candidate table with the prior justifying each
entry, execution order with decision points and pruning rules, time budget with what gets
cut first, and output artifacts. Deviations during execution must be flagged; if a plan
assumption breaks (image behavior change, OOM boundary moved), stop and revise.

## 1. Hardware priors for PCIe-only nodes (no NVLink)

- **Large TP is poison**: per-layer all-reduce rides PCIe. Keep TP within one NUMA node's
  GPU set (typically ≤4) and prefer PXB-paired GPUs for TP2.
- Cross-NUMA scaling only via PP (activations only, one P2P hop per microbatch) or
  DP-attention / EP (communication amenable to RDMA / A2A backends).
- Prefill is compute-dense and hides PP bubbles → PP-first. Decode is latency-sensitive
  and memory-bound → PP hurts TPOT; spend parallelism on DPA + EP + speculative decoding.

## 2. Knob → flag map (verify against your SGLang version's server_args)

`tp_size × pp_size = GPUs per unit`. Hard constraints (asserted in source):
DPA requires `tp % dp == 0`; EP requires `ep × moe_dp == tp`; FlashInfer A2A requires
`dp == tp` with DPA on; decode-side radix cache is incompatible with speculative decoding.

| Knob | Flags | Prefill | Decode |
|---|---|---|---|
| TP / PP | `--tp-size N --pp-size M` | ✅ PP is the prefill lever | PP hurts TPOT |
| chunk | `--chunked-prefill-size N` | ✅ core knob | n/a |
| DPA | `--enable-dp-attention --dp-size N` | little gain at long ISL | ✅ key throughput lever |
| EP | `--ep-size N` | no gain (measured) | ✅ spreads MoE experts |
| MTP | `--speculative-algorithm NEXTN --speculative-num-steps a --speculative-eagle-topk b --speculative-num-draft-tokens c` | n/a | ✅ top lever; depth has a sweet spot |
| A2A | `--moe-a2a-backend flashinfer` | – | compare once when EP>1 |

## 3. Component pressure testing (inputs to QPS matching)

- **Prefill**: `bench_serving --dataset-name random`, `isl=ISL, osl=1`, and **always**
  `--disable-radix-cache` (equal-length prompts otherwise hit the prefix cache → fake TTFT).
  Judge P50 TTFT; count Input TPS. Naturally high fidelity.
- **Decode — no free lunch, use dual fidelity**:
  - *Low fidelity (relative ranking only)*: `isl=128, osl=OSL`. Prefill pollution is
    minor, but per-step attention reads a short KV instead of ~ISL — absolute capacity is
    overestimated; ranking across configs is roughly preserved.
  - *High fidelity (absolute numbers)* — SGLang's official pure-decode path: server adds
    `--disaggregation-mode decode --disaggregation-transfer-backend fake`; each request
    carries `{"bootstrap_host": "2.2.2.2", "bootstrap_room": 0}`
    (`FAKE_BOOTSTRAP_HOST` in the source; `FakeKVReceiver.poll()` returns Success
    immediately). Prefill compute is skipped entirely, yet KV (and linear-attention
    state) slots are allocated per request at full ISL — decode kernels read
    true-length KV, memory pressure and max concurrency are real. Upstream wraps the
    same mechanism as `benchmark.one_batch_server --fake-prefill`. Caveats: TTFT is
    meaningless here; KV contents are garbage so speculative acceptance rates may
    deviate from real text — sanity-check against the low-fidelity MTP gain.
- Judge mean TPOT/ITL; count Output TPS. Ladder stops on FAIL **or** throughput plateau
  (<5% gain — a saturated instance keeps "passing" TPOT while queueing shows up in TTFT).

`scripts/runone.sh` executes one config: container lifecycle, kill-and-drain between
configs, server health polling with fatal-log fast-fail, concurrency ladder, verdicts,
TSV append. `FAKE_KV=1` enables the pure-decode mode.

## 3.5 Seed derivation for a new model (priors are per-model)

1. **Memory feasibility**: W = weight bytes. Min unit = ceil(W / ~80% GPU mem); prune
   decode shapes whose KV headroom is < ~30 GB/GPU (they saturate at tiny running batches).
2. **Prefill seeds**: pure-PP first; candidate depths = {min, mid, deep} of what fits.
   If the model fits one GPU, scale by replicas, not PP. Chunk sweet spots are
   shape-dependent — hill-climb per shape.
3. **Decode seeds**: smallest unit with weights + ≥30 GB/GPU KV headroom; MoE → EP=TP
   plus a DPA=2 comparison; dense → DPA is the only lever. MTP from 3/1/4 if the model
   ships a draft head, else off (NGRAM as fallback).
4. The search algorithm below is model-independent; only seeds change. Qualitative
   hardware laws transfer; specific numbers do not.

## 4. Search: multi-fidelity coordinate descent (LLM in the loop, rules explicit)

```
1. Seeds:    2–4 per phase from priors
2. Screen:   low-fidelity quick pass, drop configs <60% of leader (per-GPU), keep top-K
3. Descend:  hill-climb champion one dimension at a time
             P: pp depth → chunk → concurrency    D: mtp depth → dpa → ep → concurrency
             improvement → keep direction; regression → next dimension; full loop w/o
             improvement → local optimum, stop
4. Boundary: champion on any scanned edge → extend one step (anti-miss)
5. Finals:   top-2 re-measured at high fidelity; marginal points (within 5% of SLA)
             re-run 3×, take median
6. Budget:   prune low-priority branches first when over budget
```

Coverage statement in every report: distinguish **data-backed pruning** (a measured FAIL)
from **inference-based pruning** (extrapolation — state the basis), and list unswept
dimensions.

## 5. QPS matching → PD ratio → per-1k-GPU QPS (`scripts/analyze.py`)

- Per config at its operating point: `qps_P = inputTPS/ISL`, `qps_D = outputTPS/OSL`.
- Enumerate reduced ratios xP:yD: system QPS = `min(x·qps_P, y·qps_D)`,
  per-1k-GPU QPS = `1000 × QPS / (x·gpus_P + y·gpus_D)`.
- **Report three options**: ① steady-state optimum; ② small-group ops-friendly
  alternative (prefer it when within ~5%); ③ KV-margin conservative pick — real PD adds
  a KV-transfer residual to TTFT (tens of ms), so a prefill operating point near the SLA
  line will violate it in production; pick a P point with headroom.
- This is a steady-state approximation (no KV transfer, router, or Poisson queueing).
  Closed-loop → Poisson typically inflates P90 substantially; plan capacity on P90.

## 6. Operational red lines (each learned the hard way)

1. Kill **all** sglang processes (schedulers included) between configs and poll
   `nvidia-smi` until VRAM drains, or the next server sees phantom OOM.
2. Server log redirection must target a path valid **inside** the container.
3. Never wait blind: poll `/health` while grepping the server log for
   `Traceback|OOM|NCCL error` — fail fast on fatal lines; investigate any wait >1 min.
4. PP cold-start is slow; give health checks a generous timeout.
5. Timestamp every log/result filename; prefix output lines with `[HH:MM:SS]`.
6. Record the resolved image version + digest at start; smoke-test one known config
   after any image change before trusting comparisons.
7. Shared machines: check GPU idleness first; a gap between another pipeline's jobs is
   **not** idle — wait for its driver processes to exit.
8. Colocated (non-disaggregated) DPA + MTP + chunked-prefill has triggered kernel-level
   crashes on some versions; the fake pure-decode path avoids that code path. Verify on
   your version before scanning that combination colocated.
