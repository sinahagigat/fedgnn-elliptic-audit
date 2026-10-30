"""
Third revision: three experiments that test the main claims under adequate
training and separate the client weighting from checkpoint selection.

  V0..V9   C0..C9 at budget 1 with separate validation halves for client
           weighting and checkpoint selection (FedAvg, Soft)        400 runs
  N        protocol D without early stopping (FedAvg GCN and MLP)   40 runs
  P0..P24  the 25 fixed partitions at budget 2 (FedAvg, Hard, Soft) 1,500 runs

    python tools/run_revision3.py --data-dir DATA --results-root RESULTS3 --workers 5 --threads 4

RESULTS3 should be a new, empty folder. Start the same command again after any
interruption; finished runs are skipped.
Output: RESULTS3/export/revision3_results.zip
"""
import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
from jobs import REVISION3_JOBS  # noqa: E402

FILES = ["elliptic_txs_features.csv", "elliptic_txs_classes.csv", "elliptic_txs_edgelist.csv"]
from code_versions import RELEASED as EXPECTED_CODE_HASH  # noqa: E402
HISTORY_JOBS = [j for j in REVISION3_JOBS if j[0] in "VP"]


def py(*args):
    cmd = [sys.executable, *map(str, args)]
    rc = subprocess.call(cmd, cwd=str(ROOT))
    if rc != 0:
        sys.exit(f"step failed (exit code {rc}). Start the same command again; finished parts are skipped.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    for f in FILES:
        if not (Path(args.data_dir) / f).is_file():
            sys.exit(f"missing data file: {Path(args.data_dir) / f}")
    if (Path(args.data_dir) / "LOCAL_FEATURES_ONLY").is_file():
        sys.exit("--data-dir must be the normal data folder, not the local-features folder")
    from src.utils.common import code_fingerprint
    h = code_fingerprint()
    if h != EXPECTED_CODE_HASH:
        sys.exit(f"code hash is {h}, expected {EXPECTED_CODE_HASH}. Unzip the kit into a new, "
                 "empty folder instead of over an existing code folder.")
    print(f"code hash {h} OK", flush=True)
    root = Path(args.results_root)
    root.mkdir(parents=True, exist_ok=True)

    print(f"1/3  {len(REVISION3_JOBS)} jobs (V, then N, then P)", flush=True)
    py(ROOT / "tools" / "run_local.py", "--data-dir", args.data_dir, "--results-root", root,
       "--workers", args.workers, "--threads", args.threads, "--jobs", ",".join(REVISION3_JOBS))

    print("2/3  client-score histories", flush=True)
    out = root / "revision3_diagnostics"
    py(ROOT / "tools" / "revision_analysis.py", "history", "--results-root", root, "--out", out,
       "--jobs", ",".join(HISTORY_JOBS), "--name", "history3.json")

    print("3/3  export", flush=True)
    import yaml
    from jobs import JOBS
    lines, total, done_all = [], 0, 0
    for job in REVISION3_JOBS:
        spec = JOBS[job]
        cfg = yaml.safe_load((ROOT / spec["config"]).read_text(encoding="utf-8"))
        lo, hi = map(int, spec["seeds"].split("-"))
        expected = len(cfg["experiment"]["variants"]) * (hi - lo + 1)
        f = root / job / "artifacts.json"
        runs = json.loads(f.read_text(encoding="utf-8")).get("runs", {}) if f.is_file() else {}
        done = sum(1 for r in runs.values() if r.get("status") == "completed")
        hashes = sorted({str(r.get("code_hash")) for r in runs.values()})
        total += expected; done_all += done
        if done != expected or hashes != [EXPECTED_CODE_HASH]:
            lines.append(f"  {job}: {done}/{expected} runs, code hash {hashes}")
    report = (f"{done_all}/{total} runs completed" + ("" if not lines else ", incomplete jobs:\n" + "\n".join(lines))
              + ("\nALL COMPLETE" if not lines else "\nNOT COMPLETE: start the same command again"))
    zpath = root / "export" / "revision3_results.zip"
    zpath.parent.mkdir(exist_ok=True)
    n = 0
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("report.txt", report)
        for job in REVISION3_JOBS:
            f = root / job / "artifacts.json"
            if not f.is_file():
                continue
            d = json.loads(f.read_text(encoding="utf-8"))
            for r in d.get("runs", {}).values():
                r.pop("history", None)
                n += 1
            z.writestr(f"{job}/artifacts_slim.json", json.dumps(d, ensure_ascii=False))
        z.write(out / "history3.json", "diagnostics/history3.json")
    print(report)
    print(f"Exported {n} new runs to {zpath} ({zpath.stat().st_size / 1e6:.1f} MB)")
    print("DONE. Send the file RESULTS3/export/revision3_results.zip")


if __name__ == "__main__":
    main()
