"""
Two checks in one, about 5 minutes on a Colab GPU:

1. Determinism: the same three runs (seed 42) are executed twice into two
   separate folders and every reported number and every saved score is
   compared. PASS means bit-identical results.
2. Reproduction: the global-graph test AP of those runs is compared with the
   value recorded for the same variant and seed in the paper's run A
   (v6a, non-deterministic GPU code). Small differences are expected; the
   v6a FedAvg/FedProx pair, which is mathematically identical, differed by
   a median of about 3e-4 AP and at most 5.7e-3 AP across seeds.

    python tools/determinism_check.py --work /content/detcheck
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = "Centralized_GCN,FedAvg_GCN,FedGNN_Soft_Virtual_GCN"
V6A_SEED42_AP = {   # from the paper's run A records (config 97bf30443fd2)
    "Centralized_GCN": 0.5453457230251026,
    "FedAvg_GCN": 0.5486609760549729,
    "FedGNN_Soft_Virtual_GCN": 0.5319951047274181,
}


def run(out: Path, config: str, data_dir=None):
    if out.exists():
        shutil.rmtree(out)
    cmd = [sys.executable, "-u", str(ROOT / "main.py"), "--config", config,
           "--output-dir", str(out), "--seeds", "42", "--variants", VARIANTS,
           "--skip-analysis"]
    if data_dir:
        cmd += ["--data-dir", data_dir]
    print("Running:", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=str(ROOT))
    if rc != 0:
        sys.exit(f"run failed with exit code {rc}")
    return json.loads((out / "artifacts.json").read_text(encoding="utf-8"))["runs"]


def numbers(r):
    keys = ["test_metrics", "test_metrics_local", "test_metrics_tuned",
            "test_metrics_local_tuned", "test_metrics_local_no_virtual"]
    out = {"best_val_ap": r["best_val_ap"], "best_round_idx": r["best_round_idx"]}
    for k in keys:
        if k in r:
            for m in ("ap", "auc", "f1", "mcc"):
                out[f"{k}.{m}"] = r[k][m]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/content/detcheck")
    ap.add_argument("--config", default=str(ROOT / "configs" / "v7_A_louvain.yaml"))
    ap.add_argument("--data-dir", default=None, help="folder with the Elliptic CSV files")
    a = ap.parse_args()
    work = Path(a.work)
    r1 = run(work / "run1", a.config, a.data_dir)
    r2 = run(work / "run2", a.config, a.data_dir)

    ok = True
    print("\n=== 1. Determinism (run 1 vs run 2) ===")
    for key in sorted(r1):
        n1, n2 = numbers(r1[key]), numbers(r2[key])
        bad = [k for k in n1 if n1[k] != n2.get(k)]
        f1 = work / "run1" / "predictions" / r1[key].get("predictions_file", "")
        f2 = work / "run2" / "predictions" / r2[key].get("predictions_file", "")
        if f1.is_file() and f2.is_file():
            z1, z2 = np.load(f1), np.load(f2)
            bad += [f"scores:{k}" for k in z1.files if not np.array_equal(z1[k], z2[k])]
        print(f"  {key:34s} {'IDENTICAL' if not bad else 'DIFFERENT: ' + ', '.join(bad[:6])}")
        ok &= not bad

    print("\n=== 2. Reproduction against the paper's run A (seed 42) ===")
    for key, r in sorted(r1.items()):
        v = r["variant_name"]
        new, old = r["test_metrics"]["ap"], V6A_SEED42_AP.get(v)
        if old is not None:
            print(f"  {v:28s} v7 AP={new:.4f}  v6a AP={old:.4f}  diff={new - old:+.4f}")

    print("\nRESULT:", "PASS - runs are bit-identical" if ok else
          "FAIL - runs differ; see the notes in RUN_GUIDE.md (use a CPU runtime)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
