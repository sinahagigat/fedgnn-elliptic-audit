# fedgnn-elliptic-audit

Code, configurations and run records for the article

> **When the partition picks the winner: a controlled variance audit of federated graph neural networks for illicit Bitcoin transaction detection**
> Sina Haghighat and Mohammad Ali Balafar, Faculty of Electrical and Computer Engineering, University of Tabriz.

The study runs eight federated and centralized graph neural network designs and two graph-free controls on the Elliptic Bitcoin dataset, under seven client-partition protocols and two training budgets, with twenty paired seeds per condition: 1,300 deterministic runs in total. Every number, table and figure in the article is computed by `analysis/build_numbers.py` from the run records in `results/`.

## Repository layout

| path | content |
|---|---|
| `main.py`, `src/` | federated training and evaluation code |
| `configs/v7_*.yaml` | one configuration per protocol (A, B, C0–C4, D, E); see `configs/README.md` |
| `configs/legacy/` | configuration of the earlier non-deterministic GPU execution |
| `tools/` | job table, job runners, determinism check, result merging, release verification |
| `colab/` | notebook used to run jobs on Google Colab |
| `results/v7/` | records of all 1,300 runs, one file per job (`artifacts_slim.json`) |
| `results/v6a/` | records of the 480 runs of the earlier GPU execution (protocols A, B, C0) |
| `analysis/` | scripts that produce every number, table and figure of the article |
| `executed_code/` | byte-exact archive of the code that produced the runs (code hash `de756535ee20`) |
| `docs/CHANGES_V7.md` | changes relative to the code of the earlier GPU execution |

## Installation

Python 3.14 was used for all runs (CPU only; no GPU is needed).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

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

This reads `results/` and writes `analysis/output/numbers.json`, `analysis/output/tables.json` and the five figures of the article to `analysis/output/figures/`.

## Reproducing the runs

A single protocol and seed block:

```bash
python main.py --config configs/v7_A_louvain.yaml --data-dir data --output-dir runs/A1 --seeds 42-46 --threads 4
```

All 42 jobs of the study (`python tools/jobs.py` prints the table), several in parallel on one machine, resumable after an interruption:

```bash
python tools/run_local.py --data-dir data --results-root runs --workers 5 --threads 4
python tools/merge_results.py runs
```

Determinism check (the same runs executed twice must give identical numbers and scores):

```bash
python tools/determinism_check.py --work detcheck --data-dir data
```

The recorded run times add up to about 13 hours of sequential computation for all 1,300 runs (about 4 hours of it for protocol D, which uses 300 rounds of five local steps); with several workers in parallel the wall-clock time is correspondingly shorter.

## Verifying the released code

The source files in this repository are the files that produced the runs, with comments, docstrings and log messages translated from Persian to English. Translation changes the code hash, so the exact executed code is archived in `executed_code/`. The script

```bash
python tools/verify_release.py
```

checks that

1. the archived code has the code hash `de756535ee20` recorded in every run;
2. every released Python file has the same syntax tree as the executed file, except for translated string constants (any change to executable code, identifiers or numbers fails the check);
3. no untranslated text remains in the released Python files;
4. the released configuration loader reproduces the configuration hash recorded for every protocol in `results/v7` and for the three earlier GPU runs in `results/v6a`.

Because the comments differ, runs made with the released code record the code hash `38ae2fc0d12a` instead of `de756535ee20`. Their results are the same: running all ten designs of protocol A with both versions produces identical run records and identical saved scores, bit for bit. To rerun with exactly the archived files, unzip `executed_code/fedgnn_v7_executed_de756535ee20.zip` and run `main.py` from there.

## Run records

Each run in `results/v7/<job>/artifacts_slim.json` contains the variant, seed, configuration hash, code hash, environment, selected round, convergence flag, training time, partition diagnostics and the test metrics under global inference (`test_metrics`), per-client inference (`test_metrics_local`) and validation-tuned thresholds (`test_metrics_tuned`). The per-run validation and test scores (`predictions/*.npz`) are archived separately (see the data availability statement of the article).

## Citation

See `CITATION.cff`. Please also cite the Elliptic dataset:
Weber, M. et al. (2019). Anti-money laundering in Bitcoin: Experimenting with graph convolutional networks for financial forensics. KDD Workshop on Anomaly Detection in Finance.

## License

Code: MIT (see `LICENSE`). Run records in `results/`: CC BY 4.0.
