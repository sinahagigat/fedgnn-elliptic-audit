# Configurations

One file per protocol. Each file defines the full experiment; `tools/jobs.py`
splits it into seed blocks.

| file | protocol | partition | budget | designs |
|---|---|---|---|---|
| `v7_A_louvain.yaml` | A | Louvain, redrawn per seed | 1 (100 rounds × 1 local step) | 8 + 2 graph-free controls |
| `v7_B_random.yaml` | B | uniform random, redrawn per seed | 1 | 8 + 2 graph-free controls |
| `v7_C0_fixed.yaml` … `v7_C4_fixed.yaml` | C0–C4 | one Louvain partition (partition seed 0–4) for all seeds | 1 | 7 |
| `v7_D_budget.yaml` | D | Louvain, redrawn per seed | 2 (300 rounds × 5 local steps) | 8 GCN designs |
| `v7_E_budget_mlp.yaml` | E | Louvain, redrawn per seed | 2 | 2 graph-free controls |
| `legacy/run_v6a_20seeds.yaml` | earlier GPU execution | see `docs/CHANGES_V7.md` | 1 | 8 |

The files are kept exactly as executed. The `description` fields of the
variants are written in Persian; they are part of the configuration hash, so
translating them would change the hashes recorded in the run records. They
are informational only and do not affect training. English equivalents:

| variant | description |
|---|---|
| `Centralized_GCN` | Performance ceiling: centralized training on the full graph. |
| `FedAvg_GCN` | Standard FedAvg baseline. |
| `FedProx_GCN` | FedProx baseline for statistical heterogeneity. |
| `FedProto_GCN` | Prototype baseline, without quality weighting. |
| `FedGNN_Hard_GCN` | Hard selective aggregation (top-k) + prototypes. |
| `FedGNN_Soft_GCN` | Soft selective aggregation (weighted softmax) + prototypes. |
| `FedGNN_Soft_Virtual_GCN` | Soft aggregation + prototypes + virtual nodes (GCN encoder). |
| `FedGNN_Soft_Virtual_GAT` | Same design with a GAT encoder, to check dependence on the encoder. |
| `Centralized_MLP`, `FedAvg_MLP` | Graph-free controls (edges ignored). |
