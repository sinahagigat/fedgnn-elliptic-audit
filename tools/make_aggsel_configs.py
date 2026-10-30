"""
Checkpoint selection on the aggregated model (fed.select_on_aggregate).
Protocols Z20, Z50 and Z100 repeat FedAvg GCN and FedAvg MLP of protocols
D/E, S50 and S100 (budget 2, early stopping with patience 20, 50, 100); ZN and
ZM repeat the full-budget protocols N and M. Only the selection criterion
changes, so the training trajectory of every run equals that of its source run.

    python tools/make_aggsel_configs.py
"""
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from make_revision_configs import load, variants, write  # noqa: E402


def main():
    d, e = load("v7_D_budget.yaml"), load("v7_E_budget_mlp.yaml")
    for pat in (20, 50, 100):
        cfg = copy.deepcopy(d)
        cfg["fed"]["patience"] = pat
        cfg["fed"]["select_on_aggregate"] = True
        cfg["experiment"]["variants"] = variants(d, ["FedAvg_GCN"]) + variants(e, ["FedAvg_MLP"])
        write(f"v7_Z{pat}_aggsel.yaml", cfg, d)
    for src, name in (("v7_N_no_early_stop.yaml", "v7_ZN_aggsel_full.yaml"), ("v7_M_designs_full.yaml", "v7_ZM_aggsel_full.yaml")):
        cfg = load(src)
        cfg["fed"]["select_on_aggregate"] = True
        write(name, cfg, load(src))


if __name__ == "__main__":
    main()
