# Option for protocols Z: checkpoint selection on the aggregated model (code hash c73ef066fa6e)

In every earlier run, the validation AP that selects the federated checkpoint
and drives the patience rule is the mean, over clients, of the validation AP of
each client's local model after its local steps (`src/fed/client.py`,
`local_train`). The checkpoint that is saved and tested is the aggregated model
of the same round (`src/fed/server.py`, `run_round`). With one local step the two
models are close; with five local steps they can differ.

The code version that ran protocols Z (code hash `c73ef066fa6e`, archived in `executed_code/`) adds one option, kept in the released code:

* `fed.select_on_aggregate` (default `false`). When `true`, after aggregation
  every client evaluates the aggregated model on its own checkpoint-selection
  validation nodes (`FederatedClient.evaluate_aggregate`, on a copy of the local
  model, in evaluation mode and without gradients), and the mean of these APs
  replaces the local-model mean for checkpoint selection and for the patience
  rule. The local-model value is still recorded as `avg_val_ap` in the history.
  Files: `src/config.py`, `src/fed/client.py`, `src/fed/server.py`.

The evaluation does not change the training: training always continues from the
latest aggregate, and evaluation uses no random numbers. A run with the option
on therefore follows exactly the trajectory of the same run with the option
off; only the selected round, and with early stopping the last round, can
differ. `tools/run_aggsel.py` checks this for every run of ZN and ZM: the
local-model criterion recomputed from the history must reproduce the selected
round and validation AP of the original runs of N and M.

With the option at its default the code is unchanged in behaviour:

* the option is left out of the configuration hash while it is `false`, so all
  earlier configuration hashes are reproduced (`tools/verify_release.py`, check 4);
* `tools/verify_release.py`, check 2, removes these additions from the syntax
  tree and requires the rest to equal both archived versions;
* running the same configuration with this version and with `598ac132e99f`
  gives bit-identical run records.

All 160 runs of ZN and ZM passed the trajectory check.

Protocols: Z20, Z50 and Z100 repeat FedAvg GCN and FedAvg MLP under budget 2
with patience 20 (protocols D and E), 50 (S50) and 100 (S100); ZN and ZM repeat
the full-budget protocols N and M. Configurations: `tools/make_aggsel_configs.py`;
runs: `tools/run_aggsel.py`.
