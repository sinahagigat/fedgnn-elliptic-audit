# FedGNN v7: changes relative to v6a

v6a is the code that produced the earlier GPU runs A, B and C (results/v6a).
Their three config hashes (97bf30443fd2, 05349bb0e459, f43e2e2e6fe2) are
reproduced exactly by `configs/legacy/run_v6a_20seeds.yaml` with
`experiment.save_checkpoints=False` (run B: `partition.strategy=random_node`;
run C: `partition.repartition_per_seed=False`), once the options added in v7
(table below, all at their defaults) are removed before hashing.
`tools/verify_release.py` performs this check.

Under the legacy config, v7 produces results identical to v6a on CPU
(checked bit for bit on synthetic data of the same size). All new behaviour
is behind config flags that the v7 configs switch on.

## New behaviour (config flags)

| flag | default | effect |
|---|---|---|
| `experiment.deterministic` | False | `torch.use_deterministic_algorithms(True)`; v6a GPU runs were not deterministic |
| `experiment.eval_local` | False | adds `test_metrics_local`: per-client inference, each client scores its own nodes on its own subgraph with its boundary nodes passed to the virtual-node module. v6a evaluated only on the full graph, where the virtual-node module was never called. For virtual-node variants `test_metrics_local_no_virtual` gives the same model with the module bypassed |
| `experiment.eval_tuned_threshold` | False | adds `test_metrics_tuned` (and `_local_tuned`): F1/P/R/MCC at the max-F1 threshold chosen on validation |
| `experiment.save_predictions` | False | per-run `predictions/<variant>__seed<k>.npz` with validation and test scores |
| `partition.fixed_partition_seed` | 0 | Louvain seed when `repartition_per_seed=False` (v6a hard-coded 0) |
| `fed.centralized_early_stop` | False | optional patience-based early stop for the centralized baseline |
| `model.arch: mlp` | - | graph-free encoder (same width/depth/LayerNorm/dropout as GCN, edges ignored) |

## Corrected bookkeeping (always on)

* Centralized `converged` is computed with the federated patience rule; v6a hard-coded `True`.
* Every run records `code_hash` (SHA-256 of `main.py` and `src/**/*.py`), `environment` (library versions, GPU/CPU, determinism flag) and `train_time_sec` (training only, excluding evaluation).
* An artifacts file refuses runs from a different config hash or code hash.
* `main.py --seeds --variants` run one shard; the config hash is always taken from the full config.

## Facts about v6a that the paper must state (unchanged in v7)

* All 10 clients train in every round; `participation_frac` is not used by the server.
* With `local_epochs=1` the FedProx proximal gradient is exactly zero, so FedProx equals FedAvg.
* FedProto, FedGNN_Hard and FedGNN_Soft all use the prototype loss; Hard keeps the top 5 clients by score, Soft weights clients by softmax(score/0.5) times data share; score = client validation AP x log(1 + training positives).
* Trainable parameters: GCN 15,300; GCN + virtual nodes 32,260; GAT + virtual nodes 45,188.
* The virtual-node payload is the mean raw feature vector of each client's boundary nodes, averaged over clients by boundary degree, projected by a frozen random linear layer, identical for all clients and constant over rounds.
* The Rust extension was never built (module name mismatch); the numpy fallback ran.
* Secure aggregation was configured but every call failed and fell back to the raw payload.
* Many `selection.*` and `fed.w_exp_*` keys are not read by any code path.
