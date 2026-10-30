"""
Generate the configs of the fourth revision batch: budget 2 without early
stopping, so that federated models train all 300 rounds and keep their best
checkpoint, as the centralized models do (protocol N showed that the stopping
rule ends federated training long before its best checkpoint).

  Q0..Q9  fixed partitions C0..C9 at budget 2, no early stopping, FedAvg and
          Soft; source v7_P<k>_fixed_budget.yaml
  R       random partition at budget 2, no early stopping, FedAvg GCN and MLP;
          source v7_F_random_budget.yaml
  L       93 local features at budget 2, no early stopping, FedAvg GCN and MLP
          (the centralized counterparts are those of protocol G); source
          v7_G_local_features_budget.yaml. Runs with the local-features data folder.

    python tools/make_revision4_configs.py
"""
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from make_revision_configs import load, variants, write  # noqa: E402


def main():
    for k in range(10):
        p = load(f"v7_P{k}_fixed_budget.yaml")
        cfg = copy.deepcopy(p)
        cfg["fed"]["early_stop_enabled"] = False
        cfg["experiment"]["variants"] = variants(p, ["FedAvg_GCN", "FedGNN_Soft_GCN"])
        write(f"v7_Q{k}_fixed_budget_full.yaml", cfg, p)
    f = load("v7_F_random_budget.yaml")
    cfg = copy.deepcopy(f); cfg["fed"]["early_stop_enabled"] = False
    write("v7_R_random_budget_full.yaml", cfg, f)
    g = load("v7_G_local_features_budget.yaml")
    cfg = copy.deepcopy(g); cfg["fed"]["early_stop_enabled"] = False
    cfg["experiment"]["variants"] = variants(g, ["FedAvg_GCN", "FedAvg_MLP"])
    write("v7_L_local_features_full.yaml", cfg, g)


if __name__ == "__main__":
    main()
