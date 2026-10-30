"""
Seventh batch: patience of the federated early-stopping rule. Protocol D
(budget 2, Louvain redrawn per seed) for FedAvg GCN and FedAvg MLP with
patience 50 (S50) and 100 (S100) instead of 20.

    python tools/make_revision7_configs.py
"""
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from make_revision_configs import load, variants, write  # noqa: E402


def main():
    d, e = load("v7_D_budget.yaml"), load("v7_E_budget_mlp.yaml")
    for pat in (50, 100):
        cfg = copy.deepcopy(d)
        cfg["fed"]["patience"] = pat
        cfg["experiment"]["variants"] = variants(d, ["FedAvg_GCN"]) + variants(e, ["FedAvg_MLP"])
        write(f"v7_S{pat}_patience.yaml", cfg, d)


if __name__ == "__main__":
    main()
