"""
Run one job from tools/jobs.py.

    python tools/run_job.py A1 --results-root /content/drive/MyDrive/FedGNN_v7/results

Every job writes to its own folder <results-root>/<JOB>/ (artifacts.json,
run.log, predictions/). Re-running the same job resumes where it stopped.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from jobs import JOBS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job", help="job name, e.g. A1, C3_2, D7, or 'list'")
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--variants", default=None, help="optional subset (comma-separated)")
    ap.add_argument("--data-dir", default=None, help="folder with the Elliptic CSV files")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    args = ap.parse_args()

    if args.job == "list":
        for k, v in JOBS.items():
            print(k, v)
        return 0
    if args.job not in JOBS:
        print(f"Unknown job '{args.job}'. Valid: {', '.join(JOBS)}")
        return 2

    job = JOBS[args.job]
    out = Path(args.results_root) / args.job
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", str(ROOT / "main.py"),
           "--config", str(ROOT / job["config"]),
           "--output-dir", str(out),
           "--artifacts-name", "artifacts.json",
           "--seeds", job["seeds"],
           "--skip-analysis"]
    if args.variants:
        cmd += ["--variants", args.variants]
    if args.data_dir:
        cmd += ["--data-dir", args.data_dir]
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    if args.threads:
        cmd += ["--threads", str(args.threads)]
        env.update(OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
    print("Running:", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=str(ROOT), env=env)
    # final status line, easy to spot at the bottom of the Colab output
    import json
    f = out / "artifacts.json"
    runs = json.loads(f.read_text(encoding="utf-8")).get("runs", {}) if f.is_file() else {}
    done = sum(1 for r in runs.values() if r.get("status") == "completed")
    failed = sum(1 for r in runs.values() if r.get("status") == "failed")
    print("\n" + "=" * 60)
    if rc == 0 and failed == 0:
        print(f"JOB {args.job} FINISHED: {done} runs completed. Start the next job.")
    else:
        print(f"JOB {args.job} STOPPED: {done} completed, {failed} failed, exit code {rc}. "
              f"Run the same job again to resume, and send the output if it fails again.")
    print("=" * 60)
    return rc


if __name__ == "__main__":
    sys.exit(main())
