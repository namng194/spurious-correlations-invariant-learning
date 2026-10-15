"""Audit completed runs and export report-ready replacements without editing the report."""
import json
import math
import shutil
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")

def pm(values, digits=3):
    return f"{statistics.mean(values):.{digits}f} $\\pm$ {statistics.stdev(values):.{digits}f}"

def stats(rows):
    assert len(rows) == 3 and {r["seed"] for r in rows} == {0, 1, 2}
    return [f"{statistics.mean(r['train_acc'] for r in rows):.3f}",
            pm([r['test_acc'] for r in rows]),
            f"{statistics.mean(r['ood_gap'] for r in rows):.3f}",
            pm([r['test_worst'] for r in rows]),
            f"{statistics.mean(r['test_ece'] for r in rows):.3f}"]

def table(path, rows):
    # Match the report's convention: it supplies the final row terminator.
    path.write_text(" \\\\\n".join(" & ".join(row) for row in rows) + "\n", encoding="utf-8")

def audit_run(path):
    record = read(path)
    splits = record["splits"]
    for key, train in splits.items():
        if key.startswith("train_"):
            val = splits[key.replace("train_", "val_", 1)]
            assert not set(train) & set(val), path
            assert len(set(train + val)) == len(train) + len(val), path
    history = record["history"]
    best = max(h["id_avg"] for h in history)
    first = next(h["step"] for h in history if h["id_avg"] == best)
    assert record["evaluations"]["trainval"]["step"] == first, path
    for h in history:
        vals = [v for k, v in h.items() if k.startswith("acc_val_")]
        assert vals and math.isclose(h["id_avg"], statistics.mean(vals)), path
    for protocol, evaluation in record["evaluations"].items():
        assert (path.parent / f"{protocol}.pt").is_file(), path
        test = evaluation["metrics"]["test"]
        assert all(0 <= test[k] <= 1 for k in ("acc", "worst", "ece")), path
        assert sum(v for k, v in test.items() if k.startswith("n_")) == 10000, path
        if "patch_groups" in test:
            assert sum(v["n"] for v in test["patch_groups"].values()) == 10000, path
    return record

if "--wait" in sys.argv:
    print("Waiting for the experiment pipeline to finish before auditing.", flush=True)
    while True:
        state = read(OUT / "status.json")
        if state["state"] == "complete":
            break
        if state["state"] == "failed":
            raise RuntimeError(f"Experiment pipeline failed: {state}")
        time.sleep(15)

main = read(OUT / "sweep/results.json")
ablate = read(OUT / "ablate/ablate.json")
assert len(main) == 27 and len(ablate) == 54
assert len({(r["method"], r["seed"], r["protocol"]) for r in main}) == 27
assert len({r["case"] for r in ablate}) == 54
by_method = defaultdict(list)
by_protocol = defaultdict(list)
for row in main:
    by_protocol[(row["method"], row["protocol"])].append(row)
    if row["protocol"] == "trainval":
        by_method[row["method"]].append(row)
cases = defaultdict(list)
audited = set()
reused = 0
for row in ablate:
    stem, suffix = row["case"].rsplit("_s", 1)
    assert int(suffix) == row["seed"]
    cases[stem].append(row)
    source = Path(row["reused_from"]) if "reused_from" in row else OUT / "ablate" / row["case"] / "run.json"
    reused += "reused_from" in row
    audited.add(source.resolve())
    record = audit_run(source)
    selected = record["evaluations"]["trainval"]
    assert row["checkpoint_step"] == selected["step"]
    for key, metric in (("test_acc", "acc"), ("test_worst", "worst"), ("test_ece", "ece")):
        assert math.isclose(row[key], selected["metrics"]["test"][metric]), row["case"]
for method in ("erm", "irm", "rex"):
    for seed in (0, 1, 2):
        source = OUT / "sweep" / f"{method}_s{seed}" / "run.json"
        audit_run(source)
        audited.add(source.resolve())
assert len(cases) == 18 and reused == 12 and len(audited) == 51
assets = OUT / "report_assets"
tables = assets / "tables"
tables.mkdir(parents=True, exist_ok=True)
names = {"erm": "ERM", "irm": "IRMv1", "rex": "V-REx"}
table(tables / "main.tex", [[names[m], *stats(by_method[m])] for m in names])
table(tables / "protocol.tex", [[names[m], *[pm([r["test_acc"] for r in by_protocol[m, p]])
      for p in ("trainval", "oracle", "last")]] for m in names])
sweeps = [("irm", "lam", p) for p in (100, 1000, 10000, 30000)] + [
          ("rex", "beta", p) for p in (100, 1000, 10000)]
table(tables / "sweep.tex", [[names[m], f"$\\{'lambda' if m == 'irm' else 'beta'}={p}$",
      *stats(cases[f"{m}_{kind}{p}"])] for m, kind, p in sweeps])
seed_rows = []
for m, kind, p in sweeps:
    rows = sorted(cases[f"{m}_{kind}{p}"], key=lambda r: r["seed"])
    stats(rows)
    seed_rows.append([names[m], f"$\\{'lambda' if m == 'irm' else 'beta'}={p}$",
                      *[f"{r['test_acc']:.4f} / {r['test_worst']:.4f}" for r in rows]])
table(tables / "sweep_seeds.tex", seed_rows)
backbones = [("erm_mlp", "ERM", "MLP"), ("erm_cnn", "ERM", "CNN"),
             ("irm_mlp", "IRMv1", "MLP"), ("irm_cnn", "IRMv1", "CNN"),
             ("irm_single_env", "IRMv1", "Single training environment")]
transfers = [(f"{m}_{case}", names[m], label) for case, label in (
             ("patch2spur", "Additional $3\\times3$ patch"),
             ("smalln", "$n=6{,}250$ per environment"),
             ("fashion", "Colored-FashionMNIST")) for m in ("erm", "irm")]
for filename, items in (("backbone_env.tex", backbones), ("robust_transfer.tex", transfers)):
    table(tables / filename, [[method, label, *stats(cases[case])] for case, method, label in items])
figs = assets / "figs"
figs.mkdir(exist_ok=True)
for name in ("acc_bar.png", "reliability_erm.png", "reliability_irm.png",
             "ig_erm_vs_irm.png", "saliency_erm_vs_irm.png"):
    shutil.copy2(OUT / "figs" / name, figs / name)
for seed in (0, 1, 2):
    path = OUT / "figs/diagnostics.json" if seed == 0 else OUT / "diagnostics" / f"seed{seed}" / "diagnostics.json"
    diagnostics = read(path)
    assert diagnostics["protocol"] == "trainval" and diagnostics["seed"] == seed
    for method in names:
        assert diagnostics["checkpoint_steps"][method] == next(
            r["checkpoint_step"] for r in by_method[method] if r["seed"] == seed)
        assert math.isclose(diagnostics["ece"][method], next(
            r["test_ece"] for r in by_method[method] if r["seed"] == seed))
save(OUT / "audit.json", {"passed": True, "main_protocol_rows": 27,
     "ablation_entries": 54, "unique_training_runs": 51, "reused_ablation_entries": 12,
     "diagnostic_seeds": [0, 1, 2], "report_table_files": [p.name for p in tables.glob("*.tex")],
     "report_figures": [p.name for p in figs.glob("*.png")],
     "note": "Assets are replacements; report prose and PDFs still require updating."})
print("Audit passed: 51 unique runs, 54 ablation entries, 6 report tables, 5 images.")
