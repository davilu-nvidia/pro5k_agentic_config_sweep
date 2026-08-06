#!/usr/bin/env python3
"""QPS matching: pick best P/D configs from results.tsv, enumerate PD ratios,
report throughput per 1000 GPUs.

P unit QPS = InputTPS / ISL   (measured at its best SLA-passing operating point)
D unit QPS = OutputTPS / OSL
System QPS for xP:yD = min(x*qps_P, y*qps_D); per-1k-GPU QPS = 1000*QPS/(x*gpus_P+y*gpus_D)
Steady-state approximation: KV-transfer overhead not included — validate the
winner with a real PD-disaggregated Poisson run (see SKILL.md Phase V).

usage: analyze.py results.tsv --isl 10000 --osl 700 [--top-p 3 --top-d 3 --max-units 8]
"""
import argparse, csv, sys
from fractions import Fraction


COLS = ["phase", "label", "tp", "pp", "dpa", "ep", "chunk", "mtp", "conc",
        "ttft_p50", "tpot_mean", "req_tps", "out_tps", "inp_tps", "status", "verdict"]


def load(path):
    rows = []
    with open(path) as f:
        first = f.readline()
        if not first.startswith("phase\t"):  # headerless legacy file
            f.seek(0)
        for vals in csv.reader(f, delimiter="\t"):
            r = dict(zip(COLS, vals))
            if r.get("phase") == "M":  # legacy label for decode-with-MTP runs
                r["phase"] = "D"
            if r.get("status") != "DONE" or r.get("verdict") != "PASS":
                continue
            try:
                rows.append(dict(
                    phase=r["phase"], label=r["label"],
                    gpus=int(r["tp"]) * int(r["pp"]),
                    conc=int(r["conc"]), ttft=float(r["ttft_p50"]),
                    tpot=float(r["tpot_mean"]),
                    inp_tps=float(r["inp_tps"]), out_tps=float(r["out_tps"]),
                ))
            except (ValueError, KeyError):
                continue
    return rows


def best_ops(rows, phase, key):
    """Best SLA-passing operating point per config label."""
    ops = {}
    for r in rows:
        if r["phase"] != phase:
            continue
        cur = ops.get(r["label"])
        if cur is None or r[key] > cur[key]:
            ops[r["label"]] = r
    return sorted(ops.values(), key=lambda r: r[key] / r["gpus"], reverse=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv")
    ap.add_argument("--isl", type=int, required=True)
    ap.add_argument("--osl", type=int, required=True)
    ap.add_argument("--top-p", type=int, default=3)
    ap.add_argument("--top-d", type=int, default=3)
    ap.add_argument("--max-units", type=int, default=8)
    a = ap.parse_args()

    rows = load(a.tsv)
    ps = best_ops(rows, "P", "inp_tps")[: a.top_p]
    ds = best_ops(rows, "D", "out_tps")[: a.top_d]
    if not ps or not ds:
        sys.exit(f"need PASS rows for both phases (P={len(ps)} D={len(ds)})")

    print(f"== P candidates (QPS = inp_tps/{a.isl}) ==")
    for p in ps:
        print(f"  {p['label']:30s} {p['gpus']}gpu c={p['conc']:<3d} "
              f"inpTPS={p['inp_tps']:.0f} TTFT={p['ttft']:.0f}ms "
              f"qps/unit={p['inp_tps']/a.isl:.3f}")
    print(f"== D candidates (QPS = out_tps/{a.osl}) ==")
    for d in ds:
        print(f"  {d['label']:30s} {d['gpus']}gpu c={d['conc']:<3d} "
              f"outTPS={d['out_tps']:.0f} TPOT={d['tpot']:.2f}ms "
              f"qps/unit={d['out_tps']/a.osl:.3f}")

    combos = []
    for p in ps:
        for d in ds:
            qp, qd = p["inp_tps"] / a.isl, d["out_tps"] / a.osl
            for x in range(1, a.max_units + 1):
                for y in range(1, a.max_units + 1):
                    fr = Fraction(x, y)
                    if (fr.numerator, fr.denominator) != (x, y):
                        continue  # keep reduced ratios only
                    qps = min(x * qp, y * qd)
                    gpus = x * p["gpus"] + y * d["gpus"]
                    combos.append(dict(
                        p=p["label"], d=d["label"], x=x, y=y, qps=qps, gpus=gpus,
                        kqps=1000 * qps / gpus,
                        bound="P" if x * qp <= y * qd else "D",
                        util=min(x * qp, y * qd) / max(x * qp, y * qd),
                    ))
    combos.sort(key=lambda c: c["kqps"], reverse=True)
    print("\n== Top PD combos (per-1k-GPU QPS) ==")
    for c in combos[:10]:
        print(f"  {c['x']}P:{c['y']}D  P={c['p']}  D={c['d']}  "
              f"gpus/group={c['gpus']}  sysQPS={c['qps']:.3f}  "
              f"per-1k-GPU QPS={c['kqps']:.1f}  {c['bound']}-bound  match={c['util']:.0%}")


if __name__ == "__main__":
    main()
