"""
Diagnostics for the revision that need the data or the full run records but
no new training runs. Every subcommand writes one JSON file.

    python tools/revision_analysis.py partitions --data-dir DATA --out OUT
    python tools/revision_analysis.py tabular    --data-dir DATA --out OUT [--threads N]
    python tools/revision_analysis.py history    --results-root RESULTS --out OUT
    python tools/revision_analysis.py timesteps  --data-dir DATA --results-root RESULTS --out OUT

partitions  per-client composition of every partition used in the study
            (protocol A/D seeds 42-61, protocol B seeds 42-61, fixed partition
            seeds 0-4 and 42-46): nodes, labelled positives per split and the
            time-step mix of each client.
tabular     non-graph centralized references (random forest, gradient
            boosting, logistic regression) on the same temporal split, with
            all 165 features and with the 93 local features only.
history     per-round client validation AP and selection scores of every run,
            taken from the full artifacts.json files (the exported records
            omit the per-round history).
timesteps   AP and positive-class prevalence per time step, for validation
            and test, from the saved per-run scores (predictions/*.npz).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_cfg(name, data_dir=None):
    from src.compat import install_config_compat
    import src.config as cfgmod
    install_config_compat(cfgmod)
    cfg = cfgmod.load_config(str(ROOT / "configs" / name))
    if data_dir:
        cfg.data.raw_dir = str(Path(data_dir))
        cfg.data.cache_dir = str(Path(data_dir) / "cache")
    return cfg


def load_graph(data_dir, config="v7_A_louvain.yaml"):
    from src.data.elliptic import load_elliptic_raw
    cfg = load_cfg(config, data_dir)
    return cfg, load_elliptic_raw(cfg.data)


def arr(t):
    return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)


def write(out, name, obj):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / name).write_text(json.dumps(obj, indent=0), encoding="utf-8")
    print(f"wrote {out / name}", flush=True)


# ---------------------------------------------------------------------------
def cmd_partitions(args):
    import logging
    logging.disable(logging.WARNING)
    from src.data.partition import partition_graph
    cfg_a, data = load_graph(args.data_dir)
    cfg_b = load_cfg("v7_B_random.yaml", args.data_dir)
    y, ts = arr(data.y), arr(data.time_step)
    masks = {s: arr(getattr(data, f"{s}_mask")).astype(bool) for s in ("train", "val", "test")}
    steps = sorted(int(t) for t in np.unique(ts))
    jobs = ([("louvain_per_seed", cfg_a, s) for s in range(42, 62)]
            + [("louvain_fixed", cfg_a, s) for s in list(range(0, 5)) + list(range(42, 47))]
            + [("random_per_seed", cfg_b, s) for s in range(42, 62)])
    rows, seen = [], {}
    for kind, cfg, seed in jobs:
        key = (cfg.partition.strategy, seed)
        if key not in seen:
            t0 = time.time()
            assignment, diag = partition_graph(data, cfg.partition, seed)
            seen[key] = (np.asarray(assignment), diag)
            print(f"  {kind:18s} seed {seed:2d}  retention={diag['edge_retention_rate']:.4f}"
                  f"  ({time.time() - t0:.1f}s)", flush=True)
        assignment, diag = seen[key]
        clients = []
        for c in sorted(int(k) for k in np.unique(assignment)):
            m = assignment == c
            rec = {"client": c, "nodes": int(m.sum())}
            for s, mk in masks.items():
                rec[f"n_{s}"] = int((m & mk).sum())
                rec[f"pos_{s}"] = int(y[m & mk].sum())
            counts = np.array([int((ts[m] == t).sum()) for t in steps])
            p = counts / max(counts.sum(), 1)
            rec["steps_present"] = int((counts > 0).sum())
            rec["step_entropy"] = float(-(p[p > 0] * np.log(p[p > 0])).sum())
            rec["step_counts"] = counts.tolist()
            clients.append(rec)
        rows.append({"kind": kind, "strategy": cfg.partition.strategy, "seed": seed,
                     "edge_retention": diag["edge_retention_rate"], "clients": clients})
    write(args.out, "partitions.json", {"time_steps": steps, "partitions": rows})


# ---------------------------------------------------------------------------
def cmd_tabular(args):
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from src.evaluate import metrics_from_scores, tuned_metrics_from_scores
    _, data = load_graph(args.data_dir)
    x, y = arr(data.x), arr(data.y).astype(int)
    tr, va, te = (arr(getattr(data, f"{s}_mask")).astype(bool) for s in ("train", "val", "test"))
    assert x.shape[1] == 165, x.shape
    feature_sets = {"all_165": slice(None), "local_93": slice(0, 93)}
    models = {
        "random_forest": (lambda s: RandomForestClassifier(
            n_estimators=200, max_features=50, n_jobs=args.threads, random_state=s), range(42, 62)),
        "hist_gradient_boosting": (lambda s: HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, early_stopping=False, random_state=s), [42]),
        "logistic_regression": (lambda s: LogisticRegression(max_iter=5000), [42]),
    }
    runs = []
    for fs, cols in feature_sets.items():
        for name, (make, seeds) in models.items():
            for s in seeds:
                t0 = time.time()
                clf = make(s).fit(x[tr][:, cols], y[tr])
                pv = clf.predict_proba(x[va][:, cols])[:, 1]
                pt = clf.predict_proba(x[te][:, cols])[:, 1]
                rec = {"model": name, "features": fs, "seed": s,
                       "val": metrics_from_scores(y[va], pv),
                       "test": metrics_from_scores(y[te], pt),
                       "test_tuned": tuned_metrics_from_scores(y[va], pv, y[te], pt),
                       "fit_sec": round(time.time() - t0, 1)}
                runs.append(json.loads(json.dumps(rec, default=float)))
                print(f"  {fs:8s} {name:24s} seed {s}  test AP={rec['test']['ap']:.4f}"
                      f"  ({rec['fit_sec']}s)", flush=True)
    write(args.out, "tabular.json", {"runs": runs})


# ---------------------------------------------------------------------------
def cmd_logreg(args):
    """Convergence check of the logistic-regression reference and a small
    sensitivity analysis (penalty strength, class weights)."""
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from src.evaluate import metrics_from_scores
    _, data = load_graph(args.data_dir)
    x, y = arr(data.x), arr(data.y).astype(int)
    tr, va, te = (arr(getattr(data, f"{s}_mask")).astype(bool) for s in ("train", "val", "test"))
    runs = []
    for fs, cols in {"all_165": slice(None), "local_93": slice(0, 93)}.items():
        for C in (0.1, 1.0, 10.0):
            for cw in (None, "balanced"):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always", ConvergenceWarning)
                    clf = LogisticRegression(max_iter=5000, C=C, class_weight=cw).fit(x[tr][:, cols], y[tr])
                conv_warn = any(issubclass(m.category, ConvergenceWarning) for m in w)
                pv = clf.predict_proba(x[va][:, cols])[:, 1]; pt = clf.predict_proba(x[te][:, cols])[:, 1]
                rec = {"features": fs, "C": C, "class_weight": cw or "none", "solver": clf.solver,
                       "n_iter": int(clf.n_iter_[0]), "max_iter": 5000, "convergence_warning": conv_warn,
                       "val": metrics_from_scores(y[va], pv), "test": metrics_from_scores(y[te], pt)}
                runs.append(json.loads(json.dumps(rec, default=float)))
                print(f"  {fs:8s} C={C:<5} class_weight={str(cw):9s} n_iter={rec['n_iter']:5d} "
                      f"warning={conv_warn}  test AP={rec['test']['ap']:.4f} AUC={rec['test']['auc']:.4f}", flush=True)
    write(args.out, "logreg.json", {"runs": runs})


# ---------------------------------------------------------------------------
def cmd_history(args):
    root = Path(args.results_root)
    out = {}
    keep = [s for s in (args.jobs or "").split(",") if s]
    for f in sorted(root.glob("*/artifacts.json")):
        job = f.parent.name
        if keep and job not in keep:
            continue
        try:
            runs = json.loads(f.read_text(encoding="utf-8")).get("runs", {})
        except Exception as e:
            print(f"  skip {f}: {e}")
            continue
        for key, r in runs.items():
            h = r.get("history") or []
            if not h:
                continue
            out[f"{job}/{key}"] = {
                "best_round_idx": r.get("best_round_idx"),
                "rounds": [{"r": e.get("round_idx"),
                            "val": {k: round(float(v), 5) for k, v in (e.get("per_client_val_ap") or {}).items()},
                            "score": ({k: round(float(v), 5) for k, v in e["selection_scores"].items()}
                                      if isinstance(e.get("selection_scores"), dict) else e.get("selection_scores"))}
                           for e in h]}
        print(f"  {job}: {len(runs)} runs", flush=True)
    write(args.out, args.name or "history.json", out)


# ---------------------------------------------------------------------------
def cmd_timesteps(args):
    from sklearn.metrics import average_precision_score
    _, data = load_graph(args.data_dir)
    ts, y = arr(data.time_step), arr(data.y).astype(int)
    vm, tm = arr(data.val_mask).astype(bool), arr(data.test_mask).astype(bool)
    ts_val, ts_test = ts[vm], ts[tm]
    prevalence = {int(t): {"n": int((ts == t).sum()), "pos": int(y[ts == t].sum())}
                  for t in np.unique(ts)}
    rows = []
    for npz in sorted(Path(args.results_root).glob("*/predictions/*.npz")):
        d = np.load(npz)
        assert len(d["y_test"]) == len(ts_test) and (d["y_test"] == y[tm]).all(), npz
        rec = {"job": npz.parent.parent.name, "run": npz.stem}
        for split, tsx in (("val", ts_val), ("test", ts_test)):
            for kind in ("global", "local"):
                k = f"p_{split}_{kind}"
                if k not in d.files:
                    continue
                yy, pp = d[f"y_{split}"], d[k]
                rec[k] = {int(t): (round(float(average_precision_score(yy[tsx == t], pp[tsx == t])), 5)
                                   if yy[tsx == t].sum() > 0 else None)
                          for t in np.unique(tsx)}
        rows.append(rec)
    print(f"  {len(rows)} score files", flush=True)
    write(args.out, "timesteps.json", {"prevalence": prevalence, "runs": rows})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["partitions", "tabular", "history", "timesteps", "logreg"])
    ap.add_argument("--data-dir")
    ap.add_argument("--results-root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--jobs", default=None, help="history: only these jobs (comma-separated)")
    ap.add_argument("--name", default=None, help="history: output file name")
    args = ap.parse_args()
    {"partitions": cmd_partitions, "tabular": cmd_tabular,
     "history": cmd_history, "timesteps": cmd_timesteps, "logreg": cmd_logreg}[args.command](args)


if __name__ == "__main__":
    main()
