"""
Sixth batch: protocol W, Louvain partitions redrawn per seed, 600 rounds x 5
local steps without early stopping (twice the full budget of protocols M/N),
for FedAvg, FedProto, Hard and Soft. Tests whether the full-budget ranking is
stable when training continues.

    python tools/make_revision6_configs.py
"""
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from make_revision_configs import load, variants, write  # noqa: E402


def main():
    d = load("v7_D_budget.yaml")
    cfg = copy.deepcopy(d)
    cfg["fed"]["early_stop_enabled"] = False
    cfg["fed"]["num_rounds"] = 600
    cfg["experiment"]["variants"] = variants(d, ["FedAvg_GCN", "FedProto_GCN", "FedGNN_Hard_GCN", "FedGNN_Soft_GCN"])
    write("v7_W_designs_600rounds.yaml", cfg, d)


if __name__ == "__main__":
    main()
