"""
Generate the configs of the third revision from the existing v7 configs.

  P0..P24  the 25 fixed partitions of protocol C at budget 2 (300 rounds x 5
           local steps), FedAvg, Hard and Soft; source v7_C<k>_fixed.yaml
  V0..V9   the fixed partitions C0..C9 at budget 1 with separate validation
           data for client weighting and for checkpoint selection
           (fed.val_split_weighting), FedAvg and Soft
  N        protocol D (budget 2, Louvain redrawn per seed) without early
           stopping, so federated models train the full 300 rounds and keep
           their best checkpoint, as the centralized models do; FedAvg GCN and
           FedAvg MLP; source v7_D_budget.yaml / v7_E_budget_mlp.yaml

Each new config differs from its source only in the keys printed below.

    python tools/make_revision3_configs.py
"""
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from make_revision_configs import load, variants, write  # noqa: E402


def main():
    d, e = load("v7_D_budget.yaml"), load("v7_E_budget_mlp.yaml")
    for k in range(25):
        c = load(f"v7_C{k}_fixed.yaml")
        cfg = copy.deepcopy(c)
        cfg["fed"]["num_rounds"] = d["fed"]["num_rounds"]
        cfg["fed"]["local_epochs"] = d["fed"]["local_epochs"]
        cfg["experiment"]["variants"] = variants(c, ["FedAvg_GCN", "FedGNN_Hard_GCN", "FedGNN_Soft_GCN"])
        write(f"v7_P{k}_fixed_budget.yaml", cfg, c)
    for k in range(10):
        c = load(f"v7_C{k}_fixed.yaml")
        cfg = copy.deepcopy(c)
        cfg["fed"]["val_split_weighting"] = True
        cfg["experiment"]["variants"] = variants(c, ["FedAvg_GCN", "FedGNN_Soft_GCN"])
        write(f"v7_V{k}_valsplit.yaml", cfg, c)
    h = copy.deepcopy(d)
    h["fed"]["early_stop_enabled"] = False
    h["experiment"]["variants"] = variants(d, ["FedAvg_GCN"]) + variants(e, ["FedAvg_MLP"])
    write("v7_N_no_early_stop.yaml", h, d)


if __name__ == "__main__":
    main()
