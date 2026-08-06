# Measurement methodology

## Why component-level measurement + QPS matching

A disaggregated PD deployment has a huge joint config space (P shape × D shape × ratio ×
router). End-to-end benchmarking of every combination is quadratic; component-level
measurement is linear and the composition is arithmetic:

- a P unit's request rate is `inputTPS / ISL`
- a D unit's request rate is `outputTPS / OSL`
- an xP:yD group sustains `min(x·qps_P, y·qps_D)`

Maximize `1000 × QPS / total GPUs` over reduced integer ratios. The winner's bottleneck
side (P-bound vs D-bound) and match percentage tell you where the next optimization
dollar goes. Note the two sides' per-GPU champions don't automatically compose into the
global optimum — integer-ratio granularity loses throughput when the sides don't divide
evenly, so enumerate top-K candidates per side, not just the winners.

## Measuring prefill in isolation

Easy case: `osl=1` makes the request all-prefill. One trap: benchmark generators emit
equal-length random prompts, and with the radix cache on, concurrent equal-length
prompts hit the prefix cache and TTFT collapses to a fantasy number. Always
`--disable-radix-cache` for prefill pressure runs.

## Measuring decode in isolation — the interesting problem

A decode instance in production holds each request's full-ISL KV. Naive proxies fail in
opposite directions:

| Proxy | What's wrong |
|---|---|
| `isl=ISL, osl=OSL` colocated | ISL/(ISL+OSL) of all tokens are prefill — you measured an aggregated instance, not decode |
| `isl=small, osl=OSL` | per-step attention reads a tiny KV instead of ~ISL → capacity overestimated, TPOT underestimated |
| shared-prefix dataset (radix on) | prefill amortized to ~0 and context length is right, but the shared prefix is stored **once** → memory footprint (and max concurrency) too optimistic; shared pages are also cache-friendlier than distinct KV |

The clean answer ships in SGLang: the **fake KV-transfer backend**.

- Server: `--disaggregation-mode decode --disaggregation-transfer-backend fake`
- Client: add `{"bootstrap_host": "2.2.2.2", "bootstrap_room": 0}` to each request —
  either via `bench_serving --extra-request-body`, or on newer versions
  `python -m sglang.benchmark.one_batch_server --fake-prefill`.

The decode server runs its standard PD flow: the prealloc queue reserves KV (and, for
hybrid-attention models, linear-state) slots for the request's full input length, then
polls the receiver — which, in the fake backend, reports Success immediately instead of
waiting for RDMA. Result:

- **Real**: KV allocation at full ISL per request (memory pressure, allocator behavior,
  max concurrency), every decode kernel (attention over true-length KV, MoE, speculative
  draft+verify, sampling), scheduler batching.
- **Fake/skipped**: all prefill compute, all KV movement.
- **Meaningless/biased**: TTFT (just alloc+queue); KV contents are garbage, so
  speculative-decoding acceptance rates may differ from real text — sanity-check the
  MTP gain ratio against the low-fidelity run, and confirm the final pick in a real PD
  setup when available.

## Dual fidelity, not single

Full-context runs are slower per point (long allocations, big batches). Use the cheap
short-context proxy to *rank* configs and prune, then re-measure the top-2 at high
fidelity for the numbers that enter QPS matching. Ranking generally survives the proxy;
absolute values do not — in one campaign the DPA gain more than doubled when measured at
true context length versus the short-context proxy.

## Search algorithm

Multi-fidelity coordinate descent with explicit anti-miss rules (boundary extension,
marginal re-measurement, plateau stop) — see SKILL.md §4. Two design points worth
stating:

- **Ladder stop needs two conditions.** Decode TPOT stays flat while an instance
  saturates (queueing appears in TTFT, which decode SLA doesn't gate), so a FAIL-only
  rule climbs forever. Stop on throughput plateau too.
- **Pruning has two grades.** A measured FAIL prunes safely; an extrapolated prune
  ("next rung would exceed the SLA by trend") must be labeled as inference in the
  report, because it is exactly where a miss can hide.

## Reporting

White-background HTML, timestamped filename. Headline: best P config, best D config, PD
ratio, per-1k-GPU QPS — as **three options** (steady-state optimum / ops-friendly small
group / KV-margin conservative). Full result table with PASS/FAIL, the image version +
digest, the coverage statement, and the caveats (fake-KV MTP acceptance, KV-transfer
residual, Poisson P90 inflation).
