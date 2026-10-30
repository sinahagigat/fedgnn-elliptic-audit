"""Load v7 results (and the v6a paper runs) into one tidy table."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from jobs import JOBS  # noqa: E402
JOB_EXP = {job: spec["experiment"] for job, spec in JOBS.items()}
V7 = RESULTS / "v7"


def _row(exp, r):
    tm = r["test_metrics"]
    row = dict(exp=exp, variant=r["variant_name"], seed=int(r["seed"]),
               ap=tm["ap"], auc=tm["auc"], f1=tm["f1"], precision=tm["precision"],
               recall=tm["recall"], mcc=tm["mcc"],
               best_val_ap=r["best_val_ap"], best_round=r["best_round_idx"],
               converged=bool(r["converged"]), rounds=r["total_rounds_run"],
               train_time=r.get("train_time_sec", np.nan), runtime=r.get("runtime_sec", np.nan),
               config_hash=r["config_hash"], code_hash=r.get("code_hash"),
               best_round_local=r.get("best_round_local", np.nan))
    for key, pre in [("test_metrics_local", "loc_"), ("test_metrics_tuned", "tun_"),
                     ("test_metrics_local_tuned", "loctun_"), ("test_metrics_local_no_virtual", "locnv_")]:
        if key in r:
            for m in ("ap", "auc", "f1", "precision", "recall", "mcc", "threshold"):
                row[pre + m] = r[key].get(m, np.nan)
    pdg = r.get("partition_diagnostics")
    if pdg:
        row["retention"] = pdg["edge_retention_rate"]
        pc = pdg["per_client"].values()
        row["label_skew"] = float(np.std([c["local_positive_rate"] for c in pc]))
        row["part_seed"] = pdg.get("effective_partition_seed")
    return row


def load_v7():
    rows = []
    for d in sorted(V7.iterdir()):
        f = d / "artifacts_slim.json"
        if not f.is_file():
            continue
        exp = JOB_EXP.get(d.name, d.name)
        for r in json.loads(f.read_text())["runs"].values():
            rows.append(_row(exp, r))
    return pd.DataFrame(rows)


def load_v6a():
    rows = []
    for n in "ABC":
        d = json.loads((RESULTS / "v6a" / f"{n}.json").read_text())["runs"]
        for r in d.values():
            rows.append(_row("v6a_" + n, r))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load_v7()
    print(df.groupby("exp").size())
    print(df.columns.tolist())
