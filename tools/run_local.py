"""
Run many jobs in parallel on one machine (CPU), resumable.

    python tools/run_local.py --data-dir D:/FedGNN_v7/data --results-root D:/FedGNN_v7/results --workers 5 --threads 4

* Jobs are taken in the priority order of tools/jobs.py.
* Finished jobs are skipped, half-finished jobs resume, so the same command
  can simply be started again after a crash, a reboot or Ctrl+C.
* --jobs A1,A2,B1 runs only those jobs; --skip D1,D2 leaves jobs out
  (for example jobs that another machine is running).
* Each job's console output goes to <results-root>/<JOB>/console.log.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from jobs import JOBS, PRIORITY  # noqa: E402
sys.path.insert(0, str(ROOT))


def expected_count(job):
    cfg = yaml.safe_load((ROOT / JOBS[job]["config"]).read_text(encoding="utf-8"))
    a, b = (JOBS[job]["seeds"].split("-") + [None])[:2]
    n_seeds = int(b) - int(a) + 1 if b else 1
    return len(cfg["experiment"]["variants"]) * n_seeds


def completed_count(results_root, job):
    f = Path(results_root) / job / "artifacts.json"
    if not f.is_file():
        return 0
    try:
        runs = json.loads(f.read_text(encoding="utf-8")).get("runs", {})
    except Exception:
        return 0
    return sum(1 for r in runs.values() if r.get("status") == "completed")


def build_cache(data_dir):
    """Build the preprocessed graph cache once, before workers start, so
    parallel processes never write the cache file at the same time."""
    from src.compat import install_config_compat
    import src.config as cfgmod
    install_config_compat(cfgmod)
    from src.data.elliptic import load_elliptic_raw
    cfg = cfgmod.load_config(str(ROOT / "configs" / "v7_A_louvain.yaml"))
    cfg.data.raw_dir = str(Path(data_dir))
    cfg.data.cache_dir = str(Path(data_dir) / "cache")
    g = load_elliptic_raw(cfg.data)
    print(f"data ready: {g.num_nodes} nodes, {g.edge_index.shape[1]} arcs", flush=True)


lock = threading.Lock()


def log(msg):
    with lock:
        print(time.strftime("%H:%M:%S"), msg, flush=True)


def run_one(job, args):
    out = Path(args.results_root) / job
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log(f"START  {job}  ({completed_count(args.results_root, job)}/{expected_count(job)} already done)")
    cmd = [sys.executable, str(ROOT / "tools" / "run_job.py"), job,
           "--results-root", args.results_root, "--data-dir", args.data_dir,
           "--threads", str(args.threads)]
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    with open(out / "console.log", "a", encoding="utf-8") as fh:
        rc = subprocess.call(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT, env=env)
    done, exp = completed_count(args.results_root, job), expected_count(job)
    status = "DONE " if (rc == 0 and done == exp) else "ERROR"
    log(f"{status}  {job}  {done}/{exp} runs  {(time.time() - t0) / 60:.0f} min"
        + ("" if status == "DONE " else f"  (see {out / 'console.log'})"))
    return job, status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--jobs", default=None, help="comma-separated job names (default: all)")
    ap.add_argument("--skip", default="", help="comma-separated job names to leave out")
    args = ap.parse_args()

    wanted = [j.strip() for j in args.jobs.split(",")] if args.jobs else list(PRIORITY)
    bad = [j for j in wanted if j not in JOBS]
    if bad:
        sys.exit(f"unknown jobs: {bad}")
    skip = {j.strip() for j in args.skip.split(",") if j.strip()}
    queue = [j for j in wanted if j not in skip
             and completed_count(args.results_root, j) < expected_count(j)]
    finished = [j for j in wanted if j not in skip and j not in queue]
    log(f"{len(finished)} jobs already finished, {len(queue)} to run, "
        f"{args.workers} workers x {args.threads} threads")
    if not queue:
        return 0

    for name in ["elliptic_txs_features.csv", "elliptic_txs_classes.csv", "elliptic_txs_edgelist.csv"]:
        if not (Path(args.data_dir) / name).is_file():
            sys.exit(f"missing data file: {Path(args.data_dir) / name}")
    log("preparing the data cache (one time, about 1-2 minutes)...")
    build_cache(args.data_dir)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda j: run_one(j, args), queue))
    errors = [j for j, s in results if s != "DONE "]
    log("ALL JOBS FINISHED" if not errors else f"finished with errors in: {errors} "
        f"- start the same command again to retry them")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
