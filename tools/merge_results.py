"""
Collect the job folders, check them, and merge them into one file per
experiment. Safe to run at any time, also while jobs are still running
(it then reports progress).

    python tools/merge_results.py /content/drive/MyDrive/FedGNN_v7/results

Checks per experiment: one config hash, one code hash, no failed runs, no
duplicated (variant, seed), and completeness against the config.
Writes <results>/merged/artifacts_<EXP>.json for complete experiments.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from jobs import JOBS  # noqa: E402


def expected_runs(config_path):
    cfg = yaml.safe_load((ROOT / config_path).read_text(encoding="utf-8"))
    names = [v["name"] for v in cfg["experiment"]["variants"]]
    seeds = cfg["experiment"]["seeds"]
    return {f"{n}__seed{s}" for n in names for s in seeds}


def main():
    root = Path(sys.argv[1])
    by_exp = defaultdict(list)
    for job, spec in JOBS.items():
        f = root / job / "artifacts.json"
        if f.is_file():
            by_exp[spec["experiment"]].append((job, f))

    merged_dir = root / "merged"
    merged_dir.mkdir(exist_ok=True)
    configs = {spec["experiment"]: spec["config"] for spec in JOBS.values()}
    all_ok = True
    print(f"{'exp':4s} {'done':>9s}  {'config_hash':12s} {'code_hash':12s} status")
    for exp in sorted(configs):
        expected = expected_runs(configs[exp])
        runs, owners, problems = {}, defaultdict(list), []
        cfg_hashes, code_hashes, snapshot = set(), set(), None
        for job, f in sorted(by_exp.get(exp, [])):
            d = json.loads(f.read_text(encoding="utf-8"))
            snapshot = snapshot or d.get("config_snapshot")
            for k, r in d.get("runs", {}).items():
                if r.get("status") != "completed":
                    problems.append(f"{job}:{k} status={r.get('status')}")
                    continue
                owners[k].append(job)
                runs[k] = dict(r, job=job)
                cfg_hashes.add(r.get("config_hash"))
                code_hashes.add(r.get("code_hash"))
        dups = {k: v for k, v in owners.items() if len(v) > 1}
        extra = set(runs) - expected
        missing = expected - set(runs)
        if dups:
            problems.append(f"duplicated runs: {sorted(dups)[:5]}")
        if extra:
            problems.append(f"unexpected runs: {sorted(extra)[:5]}")
        if len(cfg_hashes) > 1:
            problems.append(f"several config hashes: {sorted(map(str, cfg_hashes))}")
        if len(code_hashes) > 1:
            problems.append(f"several code hashes: {sorted(map(str, code_hashes))}")
        status = "COMPLETE" if not missing and not problems else ("problems" if problems else "running")
        all_ok &= status == "COMPLETE"
        ch = next(iter(cfg_hashes)) if len(cfg_hashes) == 1 else "-"
        kh = next(iter(code_hashes)) if len(code_hashes) == 1 else "-"
        print(f"{exp:4s} {len(set(runs) & expected):4d}/{len(expected):<4d}  {str(ch):12s} {str(kh):12s} {status}")
        for p in problems:
            print("      !", p)
        if status == "COMPLETE":
            out = merged_dir / f"artifacts_{exp}.json"
            out.write_text(json.dumps({"runs": runs, "config_snapshot": snapshot,
                                       "config_hash": ch, "code_hash": kh},
                                      ensure_ascii=False), encoding="utf-8")
    print("\nAll experiments complete." if all_ok else "\nNot all experiments are complete yet.")


if __name__ == "__main__":
    main()
