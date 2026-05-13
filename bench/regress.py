#!/usr/bin/env python3
"""Regression gate for per-operator throughput.

Reads two JSONL files (baseline + candidate) where each line is a single
bench record with at least these fields:

    {"op": ..., "rows": ..., "throughput_b_v_s": ..., "p99_ns": ...,
     "label": ...}

For every (op, rows) tuple that exists in both files, compute the relative
drift of throughput and p99 latency. Fail if either exceeds the configured
drift threshold (default 30%).

The candidate is allowed to be *better* (higher throughput, lower p99) by
any margin; only regressions trip the gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict


def load(path):
    records = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = (obj["op"], int(obj["rows"]))
            records[key].append(obj)
    out = {}
    for key, recs in records.items():
        best = max(recs, key=lambda r: float(r.get("throughput_b_v_s", 0.0)))
        out[key] = best
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--max-drift", type=float, default=0.30)
    args = ap.parse_args()

    import os.path
    if not os.path.exists(args.baseline):
        print(f"baseline {args.baseline} missing - first CI run, accept as PASS")
        return 0
    base = load(args.baseline)
    cand = load(args.candidate)

    fails = []
    rows_checked = 0
    for key, c in cand.items():
        if key not in base:
            continue
        rows_checked += 1
        b = base[key]
        b_tput = float(b["throughput_b_v_s"])
        c_tput = float(c["throughput_b_v_s"])
        b_p99 = float(b["p99_ns"])
        c_p99 = float(c["p99_ns"])

        if b_tput > 0:
            tput_drift = (b_tput - c_tput) / b_tput
        else:
            tput_drift = 0.0
        if b_p99 > 0:
            p99_drift = (c_p99 - b_p99) / b_p99
        else:
            p99_drift = 0.0

        op, rows = key
        status = "ok"
        if tput_drift > args.max_drift:
            status = "fail"
            fails.append(f"{op} rows={rows}: throughput drift {tput_drift*100:.1f}% "
                         f"(base={b_tput:.3f} cand={c_tput:.3f} B v/s)")
        if p99_drift > args.max_drift:
            status = "fail"
            fails.append(f"{op} rows={rows}: p99 drift {p99_drift*100:.1f}% "
                         f"(base={b_p99:.0f} cand={c_p99:.0f} ns)")
        print(f"  {op:20s} rows={rows:>10}  tput drift={tput_drift*100:+5.1f}%  "
              f"p99 drift={p99_drift*100:+5.1f}%  [{status}]")

    if rows_checked == 0:
        print("no overlapping (op, rows) pairs between baseline and candidate")
        return 1
    if fails:
        print()
        print(f"FAIL: {len(fails)} regression(s) over {args.max_drift*100:.0f}%:")
        for f in fails:
            print(f"  - {f}")
        return 1
    print(f"PASS: {rows_checked} (op, rows) pairs within {args.max_drift*100:.0f}% drift")
    return 0


if __name__ == "__main__":
    sys.exit(main())
