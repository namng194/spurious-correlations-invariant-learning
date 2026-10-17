"""Finish the corrected experiment suite after the already-running main sweep."""
import datetime
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
STATUS = OUT / "status.json"

def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def count_rows(path):
    try:
        return len(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return 0

def status(state, stage, **extra):
    try:
        ablations = json.loads((OUT / "ablate/ablate.json").read_text())
    except (OSError, ValueError):
        ablations = []
    value = {"state": state, "stage": stage, "updated_utc": timestamp(),
             "main_completed_runs": count_rows(OUT / "sweep/results.json") // 3,
             "main_expected_runs": 9,
             "ablation_completed_runs": count_rows(OUT / "ablate/ablate.json"),
             "ablation_expected_runs": 54,
             "ablation_reused_runs": sum("reused_from" in r for r in ablations),
             "ablation_expected_new_training_runs": 42, **extra}
    temp = STATUS.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(STATUS)

def run(stage, script, *args):
    command = [sys.executable, "-u", str(ROOT / script), *map(str, args)]
    log_path = OUT / (stage + ".log")
    with log_path.open("w", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        while child.poll() is None:
            status("running", stage, process_id=child.pid, log=str(log_path))
            time.sleep(10)
        if child.returncode:
            raise RuntimeError(f"{stage} failed with exit {child.returncode}; see {log_path}")

try:
    run("main_sweep", "scripts/run_sweep.py", "--config", "configs/base.yaml",
        "--out", OUT / "sweep", "--resume")
    if count_rows(OUT / "sweep/results.json") != 27:
        raise RuntimeError("Expected 27 main-sweep protocol rows")
    run("main_plots", "scripts/make_plots.py", "--in", OUT / "sweep/results.json",
        "--out", OUT / "figs")
    diagnostics = []
    for seed in (0, 1, 2):
        target = OUT / "figs" if seed == 0 else OUT / "diagnostics" / f"seed{seed}"
        run(f"diagnostics_seed{seed}", "scripts/make_diagnostics.py", "--config",
            "configs/base.yaml", "--out", target, "--seed", seed,
            "--checkpoint-root", OUT / "sweep")
        diagnostics.append(json.loads((target / "diagnostics.json").read_text()))
    aggregates = {}
    for method in ("erm", "irm", "rex"):
        measures = {"ece": [d["ece"][method] for d in diagnostics]}
        for metric in ("mean_probability_change", "prediction_flip_rate"):
            measures[metric] = [d["recoloring"][method][metric] for d in diagnostics]
        aggregates[method] = {key: {"mean": statistics.mean(vals),
                                   "std": statistics.stdev(vals), "values": vals}
                              for key, vals in measures.items()}
    (OUT / "diagnostics_summary.json").write_text(json.dumps(aggregates, indent=2))
    run("ablations", "scripts/ablate.py", "--config", "configs/heavy.yaml",
        "--out", OUT / "ablate", "--resume", "--reuse-sweep", OUT / "sweep")
    if count_rows(OUT / "ablate/ablate.json") != 54:
        raise RuntimeError("Expected 54 completed ablation runs")
    run("ablation_tables", "scripts/make_plots.py", "--in", OUT / "ablate/ablate.json",
        "--out", OUT / "figs", "--ablate")
    status("complete", "finished", diagnostic_seeds=[0, 1, 2])
except Exception as error:
    status("failed", "pipeline", error=str(error))
    raise
