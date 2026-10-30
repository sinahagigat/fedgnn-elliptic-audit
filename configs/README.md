# Configurations

One file per protocol. Each file defines the full experiment; `tools/jobs.py`
splits it into seed blocks.

| file | protocol | partition | budget | designs |
|---|---|---|---|---|
| `v7_A_louvain.yaml` | A | Louvain, redrawn per seed | 1 (100 rounds × 1 local step) | 8 + 2 graph-free controls |
| `v7_B_random.yaml` | B | uniform random, redrawn per seed | 1 | 8 + 2 graph-free controls |
| `v7_C0_fixed.yaml` … `v7_C4_fixed.yaml` | C0–C4 | one Louvain partition (partition seed 0–4) for all seeds | 1 | 7 |
| `v7_C5_fixed.yaml` … `v7_C24_fixed.yaml` | C5–C24 | one Louvain partition (partition seed 42–61, the partitions protocol A draws for its seeds; with A as diagonal they form a 20 × 20 matrix) | 1 | 7 |
| `v7_D_budget.yaml` | D | Louvain, redrawn per seed | 2 (300 rounds × 5 local steps) | 8 GCN designs |
| `v7_E_budget_mlp.yaml` | E | Louvain, redrawn per seed | 2 | 2 graph-free controls |
| `v7_F_random_budget.yaml` | F | uniform random, redrawn per seed | 2 | FedAvg with GCN and MLP encoders |
| `v7_G_local_features_budget.yaml` | G | Louvain, redrawn per seed, 93 local features only | 2 | centralized and FedAvg, GCN and MLP |
| `v7_T025_C*.yaml`, `v7_T1_C*.yaml`, `v7_T2_C*.yaml` | T | partitions of C0–C9, Soft with softmax temperature 0.25, 1 or 2 | 1 | Soft |
| `legacy/run_v6a_20seeds.yaml` | earlier GPU execution | see `docs/CHANGES_V7.md` | 1 | 8 |

The files are kept exactly as executed. The free-text `description` fields
of the variants are part of the configuration hash, so they are left as
written at run time; they are informational only and do not affect training.
Their English versions:

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

## Fields that are present but not used

`selection.participation_frac`, `selection.min_clients_per_round` and the
other partial-participation fields of the `selection` section are parsed and
validated but not read by the training loop: all ten clients train and are
aggregated in every round in every reported run. `rust.use_rust` selects an
optional compiled extension for a few graph utilities; it does not change any
computation, and the v7 runs use the Python implementation. The fields are
kept because they are part of the configuration hashes recorded in the run
records.

C5–C24, F, G and the T configurations were generated from the existing configs by
`tools/make_revision_configs.py`, which prints the keys that differ.

The later protocols were generated in the same way:

| configs | generator | difference from the source |
|---|---|---|
| `v7_V<k>_valsplit.yaml` (C0–C9) | `tools/make_revision3_configs.py` | `fed.val_split_weighting: true`; FedAvg and Soft |
| `v7_N_no_early_stop.yaml` | `tools/make_revision3_configs.py` | protocol D with `fed.early_stop_enabled: false`; FedAvg GCN and MLP |
| `v7_P<k>_fixed_budget.yaml` (C0–C24) | `tools/make_revision3_configs.py` | C<k> at budget 2 (300 rounds × 5 local steps); FedAvg, Hard, Soft |
| `v7_Q<k>_fixed_budget_full.yaml` (C0–C9) | `tools/make_revision4_configs.py` | P<k> without early stopping; FedAvg and Soft |
| `v7_R_random_budget_full.yaml` | `tools/make_revision4_configs.py` | F without early stopping |
| `v7_L_local_features_full.yaml` | `tools/make_revision4_configs.py` | G without early stopping; FedAvg GCN and MLP |
| `v7_M_designs_full.yaml` | `tools/make_revision5_configs.py` | D without early stopping; the six federated designs other than FedAvg |
| `v7_W_designs_600rounds.yaml` | `tools/make_revision6_configs.py` | D without early stopping and with 600 rounds; FedAvg, FedProto, Hard, Soft |
| `v7_S50_patience.yaml`, `v7_S100_patience.yaml` | `tools/make_revision7_configs.py` | D with patience 50 or 100; FedAvg GCN and MLP |
| `v7_Z20_aggsel.yaml`, `v7_Z50_aggsel.yaml`, `v7_Z100_aggsel.yaml` | `tools/make_aggsel_configs.py` | D/E, S50, S100 with `fed.select_on_aggregate: true`; FedAvg GCN and MLP |
| `v7_ZN_aggsel_full.yaml`, `v7_ZM_aggsel_full.yaml` | `tools/make_aggsel_configs.py` | N and M with `fed.select_on_aggregate: true` |
