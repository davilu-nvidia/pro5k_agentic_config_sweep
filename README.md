# sglang-pd-sweep

An agent skill (Claude Code / any LLM-in-the-loop harness) that automatically finds the
best SGLang serving configuration under an arbitrary SLA on PCIe-only multi-GPU nodes,
for disaggregated prefill/decode (PD) deployments.

Given:

- an SLA — P50 TTFT ceiling for prefill, mean TPOT ceiling for decode,
- a workload profile — input/output sequence lengths (ISL/OSL),
- a model path,

it sweeps prefill parallelism (TP/PP × chunked-prefill size), decode parallelism
(TP × DP-attention × EP × speculative depth), then **QPS-matches** the two sides to pick
the PD ratio and reports the max **QPS per 1000 GPUs** with three deployment options
(steady-state optimum, small-group ops-friendly, KV-transfer-margin conservative).

## Why this exists

On PCIe-only nodes (no NVLink), tensor parallelism pays per-layer all-reduce over the
slowest interconnect, so the optimal parallel strategy differs sharply between prefill
(compute-bound → pipeline parallelism wins) and decode (memory-bound → low TP + DP
attention + expert parallel + speculative decoding wins). Component-level measurement +
QPS matching finds the jointly-optimal deployment without benchmarking every end-to-end
combination.

## Key methodology (see `references/methodology.md`)

- **Pure-prefill pressure**: `osl=1` + radix cache disabled (equal-length prompts
  otherwise hit the prefix cache and fake low TTFT).
- **Pure-decode pressure**: SGLang's official fake KV path —
  `--disaggregation-mode decode --disaggregation-transfer-backend fake` on the server,
  `bootstrap_host="2.2.2.2"` injected per request. Zero prefill compute, but KV is
  allocated per request at the full ISL, so decode kernels read true-length KV and the
  memory footprint (hence max concurrency) is real.
- **Dual fidelity**: a cheap short-context proxy for relative ranking, the fake-KV
  full-context path for absolute numbers.
- **Search**: multi-fidelity coordinate descent — priors as seeds, hill-climb one
  dimension at a time, extend when the winner sits on a scan boundary, re-measure
  marginal (within 5% of SLA) points 3× median, stop a concurrency ladder on FAIL *or*
  throughput plateau (<5% gain).
- **QPS matching**: `qps_P = inputTPS/ISL`, `qps_D = outputTPS/OSL`, system QPS of an
  `xP:yD` group is `min(x·qps_P, y·qps_D)`; enumerate reduced ratios and maximize
  `1000 × QPS / total GPUs`.

## Layout

- `SKILL.md` — canonical methodology (Chinese; byte-identical with the internal working copy). Drop into `.claude/skills/<name>/` together with a private `references/site.md` holding your host/paths/GPU_MEM/model inventory.
- `scripts/runone.sh` — single-config executor (server lifecycle + concurrency ladder + verdicts)
- `scripts/analyze.py` — QPS matching / PD-ratio enumeration
- `PIPELINE.md` — optimization-path diagrams (mermaid)

Numbers and machine specifics from the campaigns that produced this method have been
removed; qualitative laws that transfer across models on PCIe-only hardware are kept.
