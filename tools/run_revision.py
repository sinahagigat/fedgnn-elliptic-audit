"""
Run everything needed for the revision, in order, and pack the results.

    python tools/run_revision.py --data-dir DATA --results-root RESULTS --workers 5 --threads 4

RESULTS is the folder that already holds the 42 finished jobs (A1 ... E4);
they are not run again. The script can be started again after any
interruption: finished steps and finished runs are skipped.

Steps
  1. diagnostics without training (partition composition, client histories,
     per-time-step AP from the saved scores, tabular references)
  2. jobs C5_1 ... C9_4 and F1 ... F4 (normal data folder)
  3. jobs G1 ... G4 with a separate data folder that holds only the 93 local
     features (its own graph cache; checked before and after the runs)
  4. RESULTS/export/revision_results.zip with the new run records and the
     diagnostics, to send back
"""
import argparse
import io
import json
import shutil
import subprocess
import sys
import zipfile
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
from jobs import JOBS  # noqa: E402

FILES = ["elliptic_txs_features.csv", "elliptic_txs_classes.csv", "elliptic_txs_edgelist.csv"]
CG_JOBS = [f"C{k}_{i}" for k in range(5, 10) for i in range(1, 5)] + [f"F{i}" for i in range(1, 5)]
H_JOBS = [f"G{i}" for i in range(1, 5)]


def step(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70, flush=True)


def py(*args):
    cmd = [sys.executable, *map(str, args)]
    rc = subprocess.call(cmd, cwd=str(ROOT))
    if rc != 0:
        sys.exit(f"step failed (exit code {rc}): {' '.join(cmd)}\n"
                 "Start the same command again; finished parts are skipped.")


def diagnostics(args, out):
    tool = ROOT / "tools" / "revision_analysis.py"
    for name, extra in [("partitions", ["--data-dir", args.data_dir]),
                        ("history", ["--results-root", args.results_root]),
                        ("timesteps", ["--data-dir", args.data_dir, "--results-root", args.results_root]),
                        ("tabular", ["--data-dir", args.data_dir, "--threads", str(args.workers * args.threads)])]:
        if (out / f"{name}.json").is_file():
            print(f"{name}: already done", flush=True)
            continue
        print(f"{name} ...", flush=True)
        py(tool, name, *extra, "--out", out)


def local_feature_dir(args):
    """Separate data folder for protocol G, with a graph cache of 93 features."""
    d = Path(args.data_dir).resolve().parent / (Path(args.data_dir).resolve().name + "_local_features")
    d.mkdir(exist_ok=True)
    for f in FILES:
        if not (d / f).is_file():
            shutil.copy2(Path(args.data_dir) / f, d / f)
    (d / "LOCAL_FEATURES_ONLY").write_text("graph cache in this folder has 93 local features\n")
    return d


def check_h_cache(d, build):
    import logging
    logging.disable(logging.WARNING)
    from src.compat import install_config_compat
    import src.config as cfgmod
    install_config_compat(cfgmod)
    from src.data.elliptic import load_elliptic_raw
    cfg = cfgmod.load_config(str(ROOT / "configs" / "v7_G_local_features_budget.yaml"))
    cfg.data.raw_dir, cfg.data.cache_dir = str(d), str(d / "cache")
    cache = d / "cache"
    if not build and not cache.is_dir():
        sys.exit("protocol G cache missing")
    g = load_elliptic_raw(cfg.data)
    n = int(g.x.shape[1])
    print(f"protocol G graph: {g.num_nodes} nodes, {n} features", flush=True)
    if n != 93:
        sys.exit(f"protocol G data folder {d} has a graph with {n} features instead of 93. "
                 f"Delete the folder {cache} and start again.")


def export(args, out):
    import merge_results
    root = Path(args.results_root)
    buf = io.StringIO()
    sys.argv = ["merge_results.py", str(root)]
    with redirect_stdout(buf):
        merge_results.main()
    report = buf.getvalue()
    zpath = root / "export" / "revision_results.zip"
    zpath.parent.mkdir(exist_ok=True)
    n = 0
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("merge_report.txt", report)
        for job in CG_JOBS + H_JOBS:
            f = root / job / "artifacts.json"
            if not f.is_file():
                continue
            d = json.loads(f.read_text(encoding="utf-8"))
            for r in d.get("runs", {}).values():
                r.pop("history", None)
                n += 1
            z.writestr(f"{job}/artifacts_slim.json", json.dumps(d, ensure_ascii=False))
        for f in sorted(out.glob("*.json")):
            z.write(f, f"diagnostics/{f.name}")
    print(report)
    print(f"Exported {n} new runs and {len(list(out.glob('*.json')))} diagnostic files "
          f"to {zpath} ({zpath.stat().st_size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--skip-diagnostics", action="store_true")
    args = ap.parse_args()
    for f in FILES:
        if not (Path(args.data_dir) / f).is_file():
            sys.exit(f"missing data file: {Path(args.data_dir) / f}")
    if (Path(args.data_dir) / "LOCAL_FEATURES_ONLY").is_file():
        sys.exit("--data-dir must be the normal data folder, not the local-features folder")
    if not (Path(args.results_root) / "A1" / "artifacts.json").is_file():
        sys.exit(f"{args.results_root} does not look like the results folder of the study (no A1)")
    out = Path(args.results_root) / "revision_diagnostics"
    out.mkdir(exist_ok=True)

    if not args.skip_diagnostics:
        step("1/4  diagnostics (no training)")
        diagnostics(args, out)

    step("2/4  jobs C5-C9 and F")
    py(ROOT / "tools" / "run_local.py", "--data-dir", args.data_dir, "--results-root", args.results_root,
       "--workers", args.workers, "--threads", args.threads, "--jobs", ",".join(CG_JOBS))

    step("3/4  jobs G (93 local features, separate data folder)")
    hdir = local_feature_dir(args)
    check_h_cache(hdir, build=True)
    py(ROOT / "tools" / "run_local.py", "--data-dir", hdir, "--results-root", args.results_root,
       "--workers", min(args.workers, 4), "--threads", args.threads, "--jobs", ",".join(H_JOBS))
    check_h_cache(hdir, build=False)

    step("4/4  export")
    export(args, out)
    print("\nDONE. Send the file RESULTS/export/revision_results.zip")


if __name__ == "__main__":
    main()
