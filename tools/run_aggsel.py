"""
Checkpoint selection on the aggregated model (fed.select_on_aggregate):
protocols Z20, Z50, Z100 (FedAvg GCN and MLP, budget 2, patience 20/50/100),
ZN and ZM (all federated designs, full budget); 280 runs.

    python tools/run_aggsel.py --data-dir DATA --results-root RESULTS8 --workers 5 --threads 4

RESULTS8 should be a new, empty folder. Start the same command again after any
interruption; finished runs are skipped.

The training trajectory does not depend on the selection criterion, so the
validation AP of the local models (kept in the history) must reproduce the
selection of the original full-budget runs exactly. Step 2 checks this for
every run of ZN and ZM against results/v7/N* and results/v7/M*.

Output: RESULTS8/export/aggsel_results.zip
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
from jobs import AGGSEL_JOBS, JOBS  # noqa: E402

FILES = ["elliptic_txs_features.csv", "elliptic_txs_classes.csv", "elliptic_txs_edgelist.csv"]
from code_versions import RELEASED as EXPECTED_CODE_HASH  # noqa: E402
SOURCE = {"ZN": "N", "ZM": "M"}  # protocols whose original runs must be reproduced


def py(*args):
    rc = subprocess.call([sys.executable, *map(str, args)], cwd=str(ROOT))
    if rc != 0:
        sys.exit(f"step failed (exit code {rc}). Start the same command again; finished parts are skipped.")


def local_selection(history, min_delta):
    """Best round and value under the local-model criterion, as the server applies it."""
    best, best_round = -1.0, -1
    for h in history:
        if h["avg_val_ap"] > best + min_delta:
            best, best_round = h["avg_val_ap"], h["round_idx"]
    return best_round, best


def load_runs(path):
    return json.loads(path.read_text(encoding="utf-8")).get("runs", {}) if path.is_file() else {}


def original_runs(protocol):
    """All runs of a protocol in results/v7, keyed by variant and seed (seed blocks differ between protocols)."""
    out = {}
    for f in sorted((ROOT / "results" / "v7").glob(f"{protocol}[0-9]*/artifacts_slim.json")):
        out.update(load_runs(f))
    return out


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

    print(f"1/3  {len(AGGSEL_JOBS)} jobs (protocols Z20, Z50, Z100, ZN, ZM)", flush=True)
    py(ROOT / "tools" / "run_local.py", "--data-dir", args.data_dir, "--results-root", root,
       "--workers", args.workers, "--threads", args.threads, "--jobs", ",".join(AGGSEL_JOBS))

    print("2/3  trajectory check against the original full-budget runs", flush=True)
    import yaml
    lines, total, done_all, checked, mismatched = [], 0, 0, 0, []
    for job in AGGSEL_JOBS:
        spec = JOBS[job]
        cfg = yaml.safe_load((ROOT / spec["config"]).read_text(encoding="utf-8"))
        lo, hi = map(int, spec["seeds"].split("-"))
        expected = len(cfg["experiment"]["variants"]) * (hi - lo + 1)
        runs = load_runs(root / job / "artifacts.json")
        done = sum(1 for r in runs.values() if r.get("status") == "completed")
        hashes = sorted({str(r.get("code_hash")) for r in runs.values()})
        total += expected; done_all += done
        if done != expected or hashes != [EXPECTED_CODE_HASH]:
            lines.append(f"  {job}: {done}/{expected} runs, code hash {hashes}")
        min_delta = float(cfg["fed"]["min_delta"])
        exp = job.split("_")[0]
        orig = original_runs(SOURCE[exp]) if exp in SOURCE else {}
        for key, r in runs.items():
            if r.get("status") != "completed":
                continue
            r["best_round_local"], r["best_val_ap_local"] = local_selection(r.get("history", []), min_delta)
            if key in orig:
                checked += 1
                o = orig[key]
                if (r["best_round_local"] != o["best_round_idx"] or abs(r["best_val_ap_local"] - o["best_val_ap"]) > 1e-12
                        or r["total_rounds_run"] != o["total_rounds_run"]):
                    mismatched.append(f"  {job} {key}: local best round {r['best_round_local']} vs {o['best_round_idx']}")
        (root / job / "checked.json").write_text(json.dumps({"runs": runs}, ensure_ascii=False), encoding="utf-8")
    report = (f"{done_all}/{total} runs completed" + ("" if not lines else ", incomplete jobs:\n" + "\n".join(lines))
              + f"\ntrajectory check: {checked - len(mismatched)}/{checked} runs reproduce the original selection"
              + ("" if not mismatched else "\n" + "\n".join(mismatched[:20]))
              + ("\nALL COMPLETE" if not lines else "\nNOT COMPLETE: start the same command again"))

    print("3/3  export", flush=True)
    zpath = root / "export" / "aggsel_results.zip"
    zpath.parent.mkdir(exist_ok=True)
    n = 0
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("report.txt", report)
        for job in AGGSEL_JOBS:
            f = root / job / "artifacts.json"
            if not f.is_file():
                continue
            d = json.loads(f.read_text(encoding="utf-8"))
            checked_runs = load_runs(root / job / "checked.json")
            for key, r in d.get("runs", {}).items():
                c = checked_runs.get(key, {})
                r["best_round_local"] = c.get("best_round_local")
                r["best_val_ap_local"] = c.get("best_val_ap_local")
                r.pop("history", None)
                n += 1
            z.writestr(f"{job}/artifacts_slim.json", json.dumps(d, ensure_ascii=False))
    print(report)
    print(f"Exported {n} new runs to {zpath} ({zpath.stat().st_size / 1e6:.1f} MB)")
    print(f"DONE. Send the file {zpath}")


if __name__ == "__main__":
    main()
