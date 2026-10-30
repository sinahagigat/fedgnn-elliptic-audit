"""
Fifth batch: protocol D (budget 2, Louvain redrawn per seed) without early
stopping for the six federated designs not yet run that way (FedAvg is in
protocol N, the centralized model is unaffected by the stopping rule).

  M   FedProx, FedProto, Hard, Soft, Soft-VN, Soft-VN-GAT; source v7_D_budget.yaml

    python tools/make_revision5_configs.py
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
    cfg["experiment"]["variants"] = variants(d, ["FedProx_GCN", "FedProto_GCN", "FedGNN_Hard_GCN", "FedGNN_Soft_GCN",
                                                 "FedGNN_Soft_Virtual_GCN", "FedGNN_Soft_Virtual_GAT"])
    write("v7_M_designs_full.yaml", cfg, d)


if __name__ == "__main__":
    main()
