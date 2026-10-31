# fedgnn-elliptic-audit

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23825265.svg)](https://doi.org/10.5281/zenodo.23825265)

Code, configurations and run records for the article

> **When the protocol picks the winner: a controlled variance audit of federated graph neural networks for illicit Bitcoin transaction detection**
> Sina Haghighat and Mohammad Ali Balafar, Faculty of Electrical and Computer Engineering, University of Tabriz.

The study runs eight federated and centralized graph neural network designs and parameter-matched graph-free controls on the Elliptic Bitcoin dataset, under protocols that vary the client partition (Louvain redrawn per seed, random, and 25 fixed Louvain partitions that include a complete 20 × 20 matrix of partition seed by training seed), the training budget and stopping rule (100 rounds × 1 local step; 300 rounds × 5 local steps with early stopping; the same for the full budget without early stopping, and 600 rounds; patience 50 and 100), the checkpoint-selection criterion (local models or aggregated model), the input features, the softmax temperature of the score-weighted aggregation and the use of separate validation data for client weighting, with twenty paired seeds per condition: 7,800 deterministic runs in total. Non-graph reference models (random forest, gradient-boosted trees, logistic regression) are trained on the same split. Every number, table and figure in the article is computed by `analysis/build_numbers.py` from the run records and diagnostics in `results/`.

## Repository layout

| path | content |
|---|---|
| `main.py`, `src/` | federated training and evaluation code |
| `configs/v7_*.yaml` | one configuration per protocol (A, B, C0–C24, D, E, F, G, the temperature sweep T, and V, N, P0–P24, Q0–Q9, R, L, M, W, S50, S100, Z20, Z50, Z100, ZN, ZM); see `configs/README.md` |
| `configs/legacy/` | configuration of the earlier non-deterministic GPU execution |
| `tools/` | job table, job runners, determinism check, result merging, release verification |
| `colab/` | optional notebook for running single jobs on Google Colab (the reported runs were executed on one CPU workstation) |
| `results/v7/` | records of all 7,800 runs, one file per job (`artifacts_slim.json`) |
| `results/diagnostics/` | partition composition, per-round client scores (`history*.json.gz`), per-time-step AP and the reference-model results |
| `results/v6a/` | records of the 480 runs of the earlier GPU execution (protocols A, B, C0) |
| `analysis/` | scripts that produce every number, table and figure of the article |
| `executed_code/` | byte-exact archives of the three code versions that produced runs: `de756535ee20` (4,820 runs), `598ac132e99f` (2,700) and `c73ef066fa6e` (280) |
| `docs/CHANGES_V7.md` | changes relative to the code of the earlier GPU execution |
| `docs/CHANGES_REV3.md` | the option added for protocol V and how its neutrality is verified |
| `docs/CHANGES_AGGSEL.md` | the option added for protocols Z (checkpoint selection on the aggregated model) |

## Installation

Python 3.14 was used for all runs (CPU only; no GPU is needed).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Install torch from the CPU index (`pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu`) and keep networkx at 3.6.1: networkx 3.7 returns different Louvain communities for the same seed, so every Louvain partition, and every federated result, would change. Run with `--threads 4`, as the reported runs did.

The run records store the exact environment of every run (`environment` field): Python 3.14.6, PyTorch 2.14.1 (CPU), PyTorch Geometric 2.8.1, with `torch.use_deterministic_algorithms(True)`.

## Data

The Elliptic dataset (Weber et al., 2019) is not redistributed here. Download the three files

```
elliptic_txs_features.csv
elliptic_txs_classes.csv
elliptic_txs_edgelist.csv
```

from Kaggle (`ellipticco/elliptic-data-set`) or through PyTorch Geometric:

```python
from torch_geometric.datasets import EllipticBitcoinDataset
EllipticBitcoinDataset(root="elliptic_pyg")   # files are placed in elliptic_pyg/raw
```

and put them in one folder, for example `data/`.

## Reproducing the analysis (minutes, no training)

```bash
python analysis/build_numbers.py
```

This reads `results/` and writes `analysis/output/numbers.json`, `analysis/output/tables.json` and the figures of the article and the supplement to `analysis/output/figures/`.

## Reproducing the runs

A single protocol and seed block:

```bash
python main.py --config configs/v7_A_louvain.yaml --data-dir data --output-dir runs/A1 --seeds 42-46 --threads 4
```

All 400 jobs of the study (`python tools/jobs.py` prints the table), several in parallel on one machine, resumable after an interruption:

```bash
python tools/run_local.py --data-dir data --results-root runs --workers 5 --threads 4
python tools/merge_results.py runs
```

Protocol G uses only the 93 local features and therefore runs from its own data folder, so that its graph cache is kept apart from the full-feature graph; `tools/run_revision.py` creates that folder, runs C5–C9, F and G, computes the diagnostics in `results/diagnostics` (`tools/revision_analysis.py`) and checks the feature count before and after the runs:

```bash
python tools/run_revision.py --data-dir data --results-root runs --workers 5 --threads 4
python tools/run_revision2.py --data-dir data --results-root runs --workers 5 --threads 4   # C10-C24 and the temperature sweep
python tools/run_revision3.py --data-dir data --results-root runs3 --workers 5 --threads 4  # V, N, P0-P24
python tools/run_revision4.py --data-dir data --results-root runs4 --workers 5 --threads 4  # Q0-Q9, R, L (full budget)
python tools/run_revision5.py --data-dir data --results-root runs5 --workers 5 --threads 4  # M (full budget)
python tools/run_revision6.py --data-dir data --results-root runs6 --workers 5 --threads 4  # W (600 rounds)
python tools/run_revision7.py --data-dir data --results-root runs7 --workers 5 --threads 4  # S50, S100 (patience) and the logistic-regression check
python tools/run_aggsel.py --data-dir data --results-root runs8 --workers 5 --threads 4     # Z20, Z50, Z100, ZN, ZM (selection on the aggregated model)
```

Determinism check (the same runs executed twice must give identical numbers and scores):

```bash
python tools/determinism_check.py --work detcheck --data-dir data
```

With five parallel workers on a 20-thread desktop CPU, the 28 jobs of protocols C5–C9, F and G took about two and a quarter hours, the 90 jobs of C10–C24 and the temperature sweep about four and a half hours, V, N and P0–P24 about five hours, Q0–Q9, R and L about seven hours, M about two and a half hours and W about three hours.

## Verifying the released code

Three code versions produced the runs, each identified by the code hash stored in its run records: `de756535ee20` (protocols A–G, C0–C24, T), `598ac132e99f` (V, N, P, Q, R, L, M, W, S50, S100; adds the option of protocol V, `docs/CHANGES_REV3.md`) and `c73ef066fa6e` (Z20, Z50, Z100, ZN, ZM; adds selection on the aggregated model, `docs/CHANGES_AGGSEL.md`). All three are archived byte for byte in `executed_code/`. The released code (`112f9c4c9616`, `tools/code_versions.py`) combines both options in one code base, with edited comments, docstrings and log messages. An unused hook of the virtual-node module (`virtual.protection`, set to `none` in every configuration) was simplified. Because comments and messages enter the code hash, the released hash differs from the three executed ones; the computation does not. The script

```bash
python tools/verify_release.py
```

checks that

1. each archive has the code hash recorded in its runs;
2. every released Python file has the same syntax tree as the corresponding file of each archive, after the options added since that version are removed; comments, docstrings and message text are not compared, and the `virtual.protection` hook is compared only together with check 3 (any other change to executable code, identifiers, numbers or strings fails the check);
3. every configuration sets `virtual.protection: none`, so the simplified hook is never entered;
4. the released configuration loader reproduces the configuration hash recorded for every protocol in `results/v7` and for the three earlier GPU runs in `results/v6a`.

Test runs of all ten variants (eight graph-encoder designs and two graph-free controls) with the released code and with the archived versions produce identical run records and identical saved scores, bit for bit. To rerun with exactly the archived files, unzip an archive from `executed_code/` and run `main.py` from there.

## Checking a copy of the repository

```bash
python tools/check_manifest.py       # every file identical to the released version (SHA-256)
python tools/check_reproduction.py   # every number and table of the article recomputed from the records
python tools/verify_release.py       # code and configuration hashes
```

## Run records

Each run in `results/v7/<job>/artifacts_slim.json` contains the variant, seed, configuration hash, code hash, environment, selected round, convergence flag, training time, partition diagnostics and the test metrics under global inference (`test_metrics`), per-client inference (`test_metrics_local`) and validation-tuned thresholds (`test_metrics_tuned`). The per-run validation and test scores (`predictions/*.npz`) are attached to the repository release as a zip file, packed with `tools/pack_predictions.py`.

## Citation

See `CITATION.cff`. The archived release (code, run records and prediction scores) is on Zenodo: https://doi.org/10.5281/zenodo.23825265. Please also cite the Elliptic dataset:
Weber, M. et al. (2019). Anti-money laundering in Bitcoin: Experimenting with graph convolutional networks for financial forensics. KDD Workshop on Anomaly Detection in Finance.

## License

Code: MIT (see `LICENSE`). Run records in `results/`: CC BY 4.0.
