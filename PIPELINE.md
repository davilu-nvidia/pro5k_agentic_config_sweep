# Optimization-path diagram

Given any SLA + workload + model, the skill's full search pipeline (SKILL.md is the
authoritative definition; this is the guided tour):

```mermaid
flowchart TB
    A["Input: SLA (TTFT/TPOT) + ISL/OSL + model"] --> PLAN["§0.5 Sweep Plan<br/>env snapshot / candidate table / budget / decision points<br/>user confirms before execution"]
    PLAN --> SEED["§3.5 Seed derivation (zero numeric priors)<br/>memory feasibility: min unit = ceil(W/(0.8×GPU_MEM)), KV headroom ≥30GB/GPU<br/>P: pure PP {shallow,mid,deep} / single-GPU models scale by replicas<br/>D: smallest feasible unit, MoE→EP=TP + DPA contrast, MTP from 3/1/4"]

    SEED --> SCREEN["Screening (fake pure-D, short ladders)<br/>drop per-GPU throughput < 60% of leader, keep top-K"]

    SCREEN --> CD["Coordinate descent (LLM in the loop, reads results.tsv per point)<br/>P: pp depth→chunk→concurrency<br/>D: mtp depth→dpa→ep→mem-fraction→concurrency"]
    CD --> CDRULE{"±1 step on one dim"}
    CDRULE -->|"improves & PASS"| CD2["keep direction"] --> CDRULE
    CDRULE -->|"regresses"| CD3["next dimension"] --> CDRULE
    CDRULE -->|"no improvement on any dim"| LOCAL["per-dimension local optimum"]

    CD -.stop rules.-> STOPS["① stop ladder on SLA FAIL<br/>② stop on throughput plateau (+<5%)<br/>③ boundary extension: winner on scan edge → extend<br/>④ consecutive gains <5%/step → converged"]

    LOCAL --> FINAL["Finals (full ladders)<br/>D via official pure-D: fake KV backend + FAKE_BOOTSTRAP_HOST<br/>true-ISL context / fully-charged KV allocation<br/>marginal points (|metric-SLO|<5%): 3× median + adjacent rungs"]

    FINAL --> G["G1-G4 global hardening (budget ≈ 50% of main search)<br/>G1 interaction probes: two-dim diagonals<br/>G2 untouched-knob roll call: one shot each, SRVFAIL is information<br/>G3 far-point ε-exploration: different shape families<br/>G4 fine-grid audit: half steps + noise-aware runoff"]
    G -->|"a probe wins > noise (5%)"| CD
    G -->|"nothing wins"| CERT["local-optimality certificate (G-hardened)"]

    CERT --> MATCH["QPS matching<br/>qps_P=inpTPS/ISL, qps_D=outTPS/OSL<br/>enumerate xP:yD, system QPS=min(x·qps_P, y·qps_D)<br/>per-1k-GPU QPS = 1000×QPS/total GPUs"]

    MATCH --> REPORT["Report (white-background HTML, timestamped)<br/>three options: steady-state / small-group ops pick / KV-margin conservative<br/>confidence: certificates + G outcomes + roofline bound<br/>coverage: data-backed vs inference-based pruning vs unswept dims"]

    style SEED fill:#ede9fe
    style CD fill:#dbeafe
    style G fill:#fef3c7
    style MATCH fill:#d1fae5
    style REPORT fill:#f3f4f6
    style STOPS fill:#fee2e2
```

## Measurement design at a glance

```mermaid
flowchart LR
    subgraph P["P pressure (inherently high fidelity)"]
      P1["isl=ISL, osl=1 pure prefill<br/>+ --disable-radix-cache<br/>judge P50 TTFT, count Input TPS"]
    end
    subgraph D["D pressure (always high-fidelity fake pure-D)"]
      D2["--disaggregation-transfer-backend fake<br/>+ bootstrap_host=2.2.2.2<br/>zero prefill, full-ISL KV per request<br/>short ladders for screening / full ladders for finals<br/>judge mean TPOT, count Output TPS"]
    end
```

Validated across two regimes: the multi-GPU split regime (large MoE, PP/EP shape
family) and the single-GPU replica regime (small MoE, TP1/PP1 family) converge to the
correct shape family under the same rules; the G stage has caught a real knob
improvement that coordinate descent missed.
