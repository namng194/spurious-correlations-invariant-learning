"""Aggregate sweep results into figures + markdown table.

Usage: uv run python scripts/make_plots.py --in outputs/sweep/results.json --out outputs/figs
"""
import argparse
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="outputs/sweep/results.json")
    ap.add_argument("--out", default="outputs/figs")
    args = ap.parse_args()
    rows = json.load(open(args.inp))
    os.makedirs(args.out, exist_ok=True)
    by_m = defaultdict(list)
    for r in rows:
        by_m[r["method"]].append(r)
    # bar: train vs test acc (mean over seeds)
    methods = sorted(by_m)
    train_m = [sum(r["train_acc"] for r in by_m[m]) / len(by_m[m]) for m in methods]
    test_m = [sum(r["test_acc"] for r in by_m[m]) / len(by_m[m]) for m in methods]
    x = range(len(methods))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([i - 0.2 for i in x], train_m, width=0.4, label="train (ID avg)")
    ax.bar([i + 0.2 for i in x], test_m, width=0.4, label="test (OOD p_e=0.9)")
    ax.set_xticks(list(x), methods)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("ERM collapses OOD; invariant methods recover")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "acc_bar.png"), dpi=150)
    # markdown table
    lines = ["| method | train_acc | test_acc | ood_gap | worst_group | ece |",
             "|---|---|---|---|---|---|"]
    for m in methods:
        rs = by_m[m]
        n = len(rs)
        avg = lambda k: sum(r[k] for r in rs) / n
        lines.append(f"| {m} | {avg('train_acc'):.3f} | {avg('test_acc'):.3f} | "
                     f"{avg('ood_gap'):.3f} | {avg('test_worst'):.3f} | {avg('test_ece'):.3f} |")
    open(os.path.join(args.out, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
