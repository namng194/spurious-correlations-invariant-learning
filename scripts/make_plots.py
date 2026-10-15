"""Aggregate sweep results into figures + markdown/LaTeX tables.

Supports the honest model-selection protocol field:
  protocol in {last, trainval, oracle}. Defaults to trainval when present,
  falling back to last for legacy files.

Usage:
  uv run python scripts/make_plots.py --in outputs/sweep/results.json --out outputs/figs
  uv run python scripts/make_plots.py --in outputs/ablate/ablate.json --out outputs/figs --ablate
"""
import argparse
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def mean_std(xs):
    xs = np.asarray(xs, dtype=float)
    return float(xs.mean()), float(xs.std(ddof=1)) if len(xs) > 1 else 0.0


def _fmt(x, nd=3):
    if x != x:  # nan (e.g. single-env has no ID/OOD gap)
        return "---"
    return f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="outputs/sweep/results.json")
    ap.add_argument("--out", default="outputs/figs")
    ap.add_argument("--ablate", action="store_true")
    args = ap.parse_args()
    rows = json.load(open(args.inp, encoding="utf-8"))
    os.makedirs(args.out, exist_ok=True)

    if not args.ablate:
        # Main sweep: group by (method, protocol); prefer trainval.
        protos = {r.get("protocol", "last") for r in rows}
        proto = "trainval" if "trainval" in protos else "last"
        sel = [r for r in rows if r.get("protocol", "last") == proto]
        if not sel:
            sel = rows
        by_m = defaultdict(list)
        for r in sel:
            by_m[r["method"]].append(r)
        methods = sorted(by_m)
        train_m = [np.mean([r["train_acc"] for r in by_m[m]]) for m in methods]
        test_m = [np.mean([r["test_acc"] for r in by_m[m]]) for m in methods]
        test_s = [mean_std([r["test_acc"] for r in by_m[m]])[1] for m in methods]
        x = np.arange(len(methods))
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.bar(x - 0.2, train_m, width=0.4, label="train (ID avg)")
        ax.bar(x + 0.2, test_m, width=0.4, yerr=test_s, capsize=4,
               label="test (OOD $p_e=0.9$)")
        ax.set_xticks(list(x), methods)
        ax.set_ylabel("accuracy")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.set_title(f"ERM collapses OOD; invariant methods recover [{proto}, mean±std]")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "acc_bar.png"), dpi=150)
        # markdown + latex tables
        lines = ["| method | train_acc | test_acc | ood_gap | worst_group | ece |",
                 "|---|---|---|---|---|---|"]
        tex = []
        for m in methods:
            rs = by_m[m]
            n = len(rs)
            avg = lambda k: sum(r[k] for r in rs) / n
            sd = lambda k: mean_std([r[k] for r in rs])[1]
            lines.append(
                f"| {m} | {_fmt(avg('train_acc'))} | {_fmt(avg('test_acc'))}±{_fmt(sd('test_acc'))} | "
                f"{_fmt(avg('ood_gap'))} | {_fmt(avg('test_worst'))}±{_fmt(sd('test_worst'))} | "
                f"{_fmt(avg('test_ece'))} |")
            tex.append(
                f"{m} & {_fmt(avg('train_acc'))} & {_fmt(avg('test_acc'))} $\\pm$ {_fmt(sd('test_acc'))} & "
                f"{_fmt(avg('ood_gap'))} & {_fmt(avg('test_worst'))} $\\pm$ {_fmt(sd('test_worst'))} & "
                f"{_fmt(avg('test_ece'))} \\\\")
        open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
        # NOTE: last table row must NOT end with \\ (breaks booktabs rules after
        # \input inside tabular); the main .tex adds the final \\ after \input.
        tex[-1] = tex[-1].removesuffix(" \\\\")
        open(os.path.join(args.out, "summary.tex"), "w", encoding="utf-8").write("\n".join(tex) + "\n")
        print("\n".join(lines))
    else:
        # Ablations: aggregate cases with _s{seed} suffix into mean±std per base case.
        base = defaultdict(list)
        for r in rows:
            case = r["case"]
            # strip trailing _s{seed}
            if "_s" in case:
                stem = case.rsplit("_s", 1)[0]
            else:
                stem = case
            base[stem].append(r)
        lines = ["| case | train_acc | test_acc | ood_gap | worst_group | ece |",
                 "|---|---|---|---|---|---|"]
        tex = []
        def _case(c):
            return c.replace("_", "\\_")

        for case in sorted(base):
            rs = base[case]
            n = len(rs)
            avg = lambda k: sum(r[k] for r in rs) / n
            sd = lambda k: mean_std([r[k] for r in rs])[1]
            lines.append(
                f"| {case} | {_fmt(avg('train_acc'))} | {_fmt(avg('test_acc'))}±{_fmt(sd('test_acc'))} | "
                f"{_fmt(avg('ood_gap'))} | {_fmt(avg('test_worst'))}±{_fmt(sd('test_worst'))} | "
                f"{_fmt(avg('test_ece'))} |")
            tex.append(
                f"\\texttt{{{_case(case)}}} & {_fmt(avg('train_acc'))} & "
                f"{_fmt(avg('test_acc'))} $\\pm$ {_fmt(sd('test_acc'))} & "
                f"{_fmt(avg('ood_gap'))} & {_fmt(avg('test_worst'))} $\\pm$ {_fmt(sd('test_worst'))} & "
                f"{_fmt(avg('test_ece'))} \\\\")
        open(os.path.join(args.out, "ablate_summary.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
        tex[-1] = tex[-1].removesuffix(" \\\\")
        open(os.path.join(args.out, "ablate_summary.tex"), "w", encoding="utf-8").write("\n".join(tex) + "\n")
        print("\n".join(lines))


if __name__ == "__main__":
    main()
