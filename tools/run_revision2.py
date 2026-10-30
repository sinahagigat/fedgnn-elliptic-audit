"""
Second revision: complete the 20 x 20 matrix of partition seed x training
seed (jobs C10_1 ... C24_4) and vary the softmax temperature of Soft on the
ten fixed partitions (jobs T025C0 ... T2C9), then pack the results.

    python tools/run_revision2.py --data-dir DATA --results-root RESULTS --workers 5 --threads 4

RESULTS is the folder that already holds the finished jobs; they are not run
again. Start the same command again after any interruption.
Output: RESULTS/export/revision2_results.zip
"""
import argparse
import io
import json
import subprocess
import sys
import zipfile
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from jobs import REVISION2_JOBS  # noqa: E402

FILES = ["elliptic_txs_features.csv", "elliptic_txs_classes.csv", "elliptic_txs_edgelist.csv"]
HISTORY_JOBS = [f"C{k}_{i}" for k in range(5, 25) for i in range(1, 5)] + [j for j in REVISION2_JOBS if j.startswith("T")]


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
    root = Path(args.results_root)
    if not (root / "C5_1" / "artifacts.json").is_file():
        sys.exit(f"{root} does not look like the results folder of the study (no C5_1)")

    print("1/3  jobs C10-C24 and temperature sweep", flush=True)
    py(ROOT / "tools" / "run_local.py", "--data-dir", args.data_dir, "--results-root", root,
       "--workers", args.workers, "--threads", args.threads, "--jobs", ",".join(REVISION2_JOBS))

    print("2/3  client-score histories", flush=True)
    out = root / "revision2_diagnostics"
    py(ROOT / "tools" / "revision_analysis.py", "history", "--results-root", root, "--out", out,
       "--jobs", ",".join(HISTORY_JOBS), "--name", "history2.json")

    print("3/3  export", flush=True)
    import merge_results
    buf = io.StringIO()
    sys.argv = ["merge_results.py", str(root)]
    with redirect_stdout(buf):
        merge_results.main()
    zpath = root / "export" / "revision2_results.zip"
    zpath.parent.mkdir(exist_ok=True)
    n = 0
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("merge_report.txt", buf.getvalue())
        for job in REVISION2_JOBS:
            f = root / job / "artifacts.json"
            if not f.is_file():
                continue
            d = json.loads(f.read_text(encoding="utf-8"))
            for r in d.get("runs", {}).values():
                r.pop("history", None)
                n += 1
            z.writestr(f"{job}/artifacts_slim.json", json.dumps(d, ensure_ascii=False))
        z.write(out / "history2.json", "diagnostics/history2.json")
    print(buf.getvalue())
    print(f"Exported {n} new runs to {zpath} ({zpath.stat().st_size / 1e6:.1f} MB)")
    print("DONE. Send the file RESULTS/export/revision2_results.zip")


if __name__ == "__main__":
    main()
