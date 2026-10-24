# -*- coding: utf-8 -*-
"""
train.py — orchestrator of the experiment matrix (variant × seed) — v5.0.1
==============================================================================
This version is written against the actual signatures of the existing modules:
  • load_elliptic_raw(cfg: DataConfig)            -> Data
  • preprocess_elliptic(data, cfg)                -> Data          (single value)
  • partition_graph_federated(data, cfg, seed)    -> (assignment, diagnostics)
  • build_federated_dataset(data, assignment, cfg)-> List[FederatedClientData]
  • FederatedClient(client_id, fcd, in_dim, model_cfg, virtual_cfg,
                    fed_cfg, use_virtual_nodes, device)
  • FedGNNModel.forward(x, edge_index, boundary_local_idx=None)

Principles: resume by default | atomic save after every run | graceful degradation
at loop level | fail fast at module level | output always JSON-safe.
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from src.config import Config, VariantSpec
from src.data.elliptic import load_elliptic_raw, preprocess_elliptic
from src.data.partition import partition_graph_federated
from src.data.federated_dataset import (
    build_federated_dataset,
    build_centralized_splits,
    build_global_test_split,
)
from src.models.build import build_model
from src.fed.client import FederatedClient
from src.fed.server import FederatedServer
from src.evaluate import (
    evaluate_variant_on_global_test,
    predict_scores,
    metrics_from_scores,
    tuned_metrics_from_scores,
)
from src.utils.common import (
    set_global_seed,
    load_json_if_exists,
    atomic_save_json,
    get_git_commit_hash,
    make_run_key,
    utc_timestamp,
    code_fingerprint,
    environment_info,
)
from src.compat import safe_forward

logger = logging.getLogger("fedgnn.train")


# ==============================================================================
# Defensive helpers: none of them changes the scientific logic, they only remove brittleness
# ==============================================================================
def _jsonable(obj: Any) -> Any:
    """Recursive conversion to a JSON-serializable structure (numpy/torch/Path/...)."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return {str(k): _jsonable(v) for k, v in vars(obj).items()
                if not str(k).startswith("_")}
    return str(obj)


def _first(ret: Any) -> Any:
    """If a function returns a tuple, take the first element; otherwise the value itself."""
    return ret[0] if isinstance(ret, tuple) else ret


def _forward(model, x, edge_index, boundary=None) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """
    Uniform model call. FedGNNModel usually returns (node_emb, logits),
    but a single-output model does not break here either.
    """
    return safe_forward(model, x, edge_index, boundary)
    out = None
    if isinstance(out, (tuple, list)):
        return out[0], out[-1]
    return None, out


def _get(res: Any, key: str, default: Any = None) -> Any:
    """Safe read from a dict or an object."""
    if isinstance(res, dict):
        return res.get(key, default)
    return getattr(res, key, default)


def _is_centralized(variant: VariantSpec) -> bool:
    """Detect a non-federated variant, whether is_centralized is a method or a property."""
    flag = getattr(variant, "is_centralized", None)
    if isinstance(flag, bool):
        return flag
    if callable(flag):
        try:
            return bool(flag())
        except TypeError:
            pass
    return (str(getattr(variant, "algorithm", "")).lower() in
            {"centralized", "central", "local_only"}
            or str(variant.name).lower().startswith("central"))


def _resolve_model_cfg(cfg: Config, variant: VariantSpec):
    """Effective ModelConfig for this variant (full override or arch swap only)."""
    if getattr(variant, "model_overrides", None) is not None:
        return variant.model_overrides
    arch = getattr(variant, "arch", None)
    if arch:
        return cfg.model.model_copy(update={"arch": arch})
    return cfg.model


def _resolve_fed_cfg(cfg: Config, variant: VariantSpec):
    """Effective FedConfig: the variant's algorithm plus any declared fed_overrides."""
    upd: Dict[str, Any] = {"algorithm": variant.algorithm}
    upd.update(dict(getattr(variant, "fed_overrides", {}) or {}))
    upd["use_virtual_nodes"] = bool(getattr(variant, "use_virtual_nodes", False))
    return cfg.fed.model_copy(update=upd)


def _class_weight(y: torch.Tensor, mask: torch.Tensor, fed_cfg, device: str):
    """Class weight for cross-entropy, capped by max_pos_weight."""
    if not getattr(fed_cfg, "use_pos_weight", True):
        return None
    y_tr = y[mask]
    n_pos = int((y_tr == 1).sum().item())
    n_neg = int((y_tr == 0).sum().item())
    if n_pos == 0:
        return None
    w = min(max(n_neg / n_pos, 1.0), float(fed_cfg.max_pos_weight))
    return torch.tensor([1.0, w], dtype=torch.float32, device=device)


# ==============================================================================
# Single run
# ==============================================================================
def run_single_experiment(
    cfg: Config,
    variant: VariantSpec,
    seed: int,
    global_data_bundle: Dict[str, object],
) -> Dict[str, object]:
    t_start = time.time()
    set_global_seed(seed, deterministic=bool(getattr(cfg.experiment, "deterministic", False)))

    full_data = global_data_bundle["full_data"]
    in_dim = int(global_data_bundle["in_dim"])

    preds_dir = global_data_bundle.get("preds_dir")
    if _is_centralized(variant):
        artifact = _run_centralized(cfg, variant, seed, full_data, in_dim, t_start)
    else:
        # partition is built only for federated variants
        assignment, part_diag = partition_graph_federated(full_data, cfg.partition, seed)
        client_datasets = build_federated_dataset(full_data, assignment, cfg.partition)
        artifact = _run_federated(
            cfg, variant, seed, full_data, in_dim, client_datasets, t_start
        )
        artifact["partition_diagnostics"] = _jsonable(part_diag)
        artifact["num_clients_effective"] = len(client_datasets)

    artifact["runtime_sec"] = round(time.time() - t_start, 2)
    artifact["variant_name"] = variant.name
    artifact["seed"] = seed
    artifact["timestamp_utc"] = utc_timestamp()
    artifact["git_commit"] = get_git_commit_hash()
    artifact["config_hash"] = global_data_bundle.get("config_hash") or cfg.config_hash()
    artifact["code_hash"] = global_data_bundle.get("code_hash")
    env = dict(global_data_bundle.get("environment") or {})
    env["deterministic_algorithms"] = bool(torch.are_deterministic_algorithms_enabled())
    artifact["environment"] = env

    # v7: optional per-run score file (val/test scores for later re-analysis)
    scores = artifact.pop("_scores", None)
    if preds_dir is not None and scores is not None and cfg.experiment.save_predictions:
        Path(preds_dir).mkdir(parents=True, exist_ok=True)
        fname = f"{make_run_key(variant.name, seed)}.npz"
        np.savez_compressed(Path(preds_dir) / fname, **scores)
        artifact["predictions_file"] = fname
    artifact["status"] = "completed"
    return artifact


# ==============================================================================
# v7 helpers
# ==============================================================================
def _patience_converged(history_vals: List[float], patience: int, min_delta: float) -> bool:
    """Same rule as FederatedServer: patience rounds without a min_delta gain."""
    best, since = -1.0, 0
    for v in history_vals:
        if v > best + min_delta:
            best, since = v, 0
        else:
            since += 1
        if since >= patience:
            return True
    return False


def _local_scores(model, client_datasets, device, use_boundary: bool):
    """
    Per-client inference: the global model is applied to each client's own
    subgraph (only intra-client edges), with that client's boundary nodes
    passed to the virtual-node module when use_boundary is True. Every
    labelled node belongs to exactly one client, so concatenating the clients
    gives one score per node of the global graph, indexed globally.
    """
    n = sum(int(c.num_nodes) for c in client_datasets)
    out = np.full(n, np.nan, dtype=np.float64)
    for fcd in client_datasets:
        d = fcd.data.to(device)
        b = None
        if use_boundary:
            b = torch.tensor(fcd.boundary_info.local_boundary_idx, dtype=torch.long, device=device)
        p = predict_scores(model, d.x, d.edge_index, b)
        out[np.asarray(fcd.global_node_idx)] = p
    assert not np.isnan(out).any(), "per-client inference did not cover every node"
    return out


# ==============================================================================
# Centralized path: non-federated baseline (upper reference)
# ==============================================================================
def _run_centralized(
    cfg: Config, variant: VariantSpec, seed: int, full_data, in_dim: int,
    t_start: Optional[float] = None,
) -> Dict[str, object]:
    device = cfg.experiment.device
    model_cfg = _resolve_model_cfg(cfg, variant)
    fed_cfg = _resolve_fed_cfg(cfg, variant)

    data = build_centralized_splits(full_data, seed=seed, cfg=cfg.data)
    data = _first(data).to(device)

    model = build_model(
        in_dim=in_dim,
        model_cfg=model_cfg,
        virtual_cfg=None,
        use_virtual_nodes=False,
        num_classes=int(getattr(model_cfg, "num_classes", 2)),
        device=device,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=fed_cfg.lr, weight_decay=fed_cfg.weight_decay
    )
    cls_w = _class_weight(data.y, data.train_mask, fed_cfg, device)

    # step budget equal to the total federated local steps -> fair comparison
    total_steps = int(fed_cfg.num_rounds) * int(fed_cfg.local_epochs)
    best_val_ap, best_state, best_step = -1.0, None, -1
    history: List[Dict[str, float]] = []

    for step in range(total_steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, logits = _forward(model, data.x, data.edge_index, None)
        loss = F.cross_entropy(
            logits[data.train_mask], data.y[data.train_mask], weight=cls_w
        )
        loss.backward()
        clip_grad_norm_(model.parameters(), fed_cfg.grad_clip_norm)
        optimizer.step()

        val_m = evaluate_variant_on_global_test(model, data, data.val_mask, device)
        val_ap = float(val_m.get("ap", float("nan")))
        history.append({
            "round_idx": step,
            "avg_train_loss": float(loss.item()),
            "avg_val_ap": val_ap,
            "avg_val_auc": float(val_m.get("auc", float("nan"))),
        })
        if val_ap > best_val_ap:
            best_val_ap, best_step = val_ap, step
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        # v7: optional early stop with the federated patience rule
        if getattr(fed_cfg, "centralized_early_stop", False) and _patience_converged(
            [h["avg_val_ap"] for h in history], int(fed_cfg.patience), float(fed_cfg.min_delta)
        ):
            break

    assert best_state is not None, "Centralized: no checkpoint was recorded."
    train_time = time.time() - (t_start if t_start is not None else time.time())
    model.load_state_dict(best_state)
    test_metrics = evaluate_variant_on_global_test(model, data, data.test_mask, device)

    # v6a hard-coded converged=True. v7 applies the federated patience rule.
    converged = _patience_converged(
        [h["avg_val_ap"] for h in history], int(fed_cfg.patience), float(fed_cfg.min_delta)
    )
    out = {
        "algorithm": "centralized",
        "best_val_ap": best_val_ap,
        "best_round_idx": best_step,
        "test_metrics": _jsonable(test_metrics),
        "converged": bool(converged),
        "total_rounds_run": len(history),
        "train_time_sec": round(train_time, 2),
        "history": history if cfg.experiment.save_per_round_history else [],
    }

    ex = cfg.experiment
    if ex.eval_local or ex.eval_tuned_threshold or ex.save_predictions:
        y = data.y.detach().cpu().numpy()
        vm = data.val_mask.detach().cpu().numpy()
        tm = data.test_mask.detach().cpu().numpy()
        p = predict_scores(model, data.x, data.edge_index, None)
        if ex.eval_tuned_threshold:
            out["test_metrics_tuned"] = _jsonable(tuned_metrics_from_scores(y[vm], p[vm], y[tm], p[tm]))
        if ex.eval_local:
            # no clients: per-client inference is identical to global inference
            out["test_metrics_local"] = _jsonable(metrics_from_scores(y[tm], p[tm]))
            out["test_metrics_local_note"] = "centralized: identical to global inference"
        out["_scores"] = {
            "y_val": y[vm].astype(np.int8), "y_test": y[tm].astype(np.int8),
            "p_val_global": p[vm].astype(np.float32), "p_test_global": p[tm].astype(np.float32),
        }
    return out


# ==============================================================================
# Federated path: FedAvg / FedProto / FedGNN_Hard / FedGNN_Soft
# ==============================================================================
def _run_federated(
    cfg: Config,
    variant: VariantSpec,
    seed: int,
    full_data,
    in_dim: int,
    client_datasets: List,
    t_start: Optional[float] = None,
) -> Dict[str, object]:
    device = cfg.experiment.device
    model_cfg = _resolve_model_cfg(cfg, variant)
    fed_cfg = _resolve_fed_cfg(cfg, variant)
    use_vn = bool(getattr(variant, "use_virtual_nodes", False))
    virtual_cfg = cfg.virtual if use_vn else None
    n_classes = int(getattr(model_cfg, "num_classes", 2))

    # global model only used to produce the shared initial state_dict (identical shapes)
    global_model = build_model(
        in_dim=in_dim,
        model_cfg=model_cfg,
        virtual_cfg=virtual_cfg,
        use_virtual_nodes=use_vn,
        num_classes=n_classes,
        device=device,
    )
    initial_state = {k: v.detach().clone() for k, v in global_model.state_dict().items()}

    clients: List[FederatedClient] = []
    for fcd in client_datasets:
        clients.append(
            FederatedClient(
                client_id=int(fcd.client_id),
                fcd=fcd,
                in_dim=in_dim,
                model_cfg=model_cfg,
                virtual_cfg=virtual_cfg,
                fed_cfg=fed_cfg,
                use_virtual_nodes=use_vn,
                device=device,
            )
        )
    assert len(clients) >= 2, f"❌ not enough clients: {len(clients)}"

    server = FederatedServer(
        clients=clients, initial_state_dict=initial_state, fed_cfg=fed_cfg
    )
    res = server.run()
    train_time = time.time() - (t_start if t_start is not None else time.time())

    # --- best checkpoint: explicit API first, then dict key ---
    best_state = None
    try:
        best_state = server.get_best_model_state()
    except Exception as _exc:
        logger.warning(f"⚠️ previously swallowed exception (P0-9): "
                       f"{type(_exc).__name__}: {_exc}", exc_info=True)
    if best_state is None:
        best_state = _get(res, "best_state_dict") or _get(res, "best_state")
    assert best_state is not None, "❌ the server returned no final state."

    # --- evaluation on the global test set, not on any client's local data ---
    gtest = _first(build_global_test_split(full_data, cfg=cfg.data, seed=seed)).to(device)
    global_model.load_state_dict(best_state)
    test_metrics = evaluate_variant_on_global_test(
        global_model, gtest, gtest.test_mask, device
    )

    raw_hist = _get(res, "history", []) or []
    history = [{
        "round_idx": _get(r, "round_idx"),
        "avg_train_loss": _get(r, "avg_train_loss"),
        "avg_val_ap": _get(r, "avg_val_ap"),
        "avg_val_auc": _get(r, "avg_val_auc"),
        "per_client_val_ap": _jsonable(_get(r, "per_client_val_ap", {})),
        "selection_scores": _jsonable(_get(r, "selection_scores", None)),
    } for r in raw_hist]

    out = {
        "algorithm": fed_cfg.algorithm,
        "best_val_ap": float(_get(res, "best_val_ap", float("nan"))),
        "best_round_idx": _get(res, "best_round_idx", -1),
        "test_metrics": _jsonable(test_metrics),
        "converged": bool(_get(res, "converged", False)),
        "total_rounds_run": int(_get(res, "total_rounds_run", len(history))),
        "train_time_sec": round(train_time, 2),
        "history": history if cfg.experiment.save_per_round_history else [],
    }

    ex = cfg.experiment
    if ex.eval_local or ex.eval_tuned_threshold or ex.save_predictions:
        y = gtest.y.detach().cpu().numpy()
        vm = gtest.val_mask.detach().cpu().numpy()
        tm = gtest.test_mask.detach().cpu().numpy()
        # global inference (server-side, full graph): the paper's v6a protocol
        p_glob = predict_scores(global_model, gtest.x, gtest.edge_index, None)
        scores = {
            "y_val": y[vm].astype(np.int8), "y_test": y[tm].astype(np.int8),
            "p_val_global": p_glob[vm].astype(np.float32),
            "p_test_global": p_glob[tm].astype(np.float32),
        }
        if ex.eval_tuned_threshold:
            out["test_metrics_tuned"] = _jsonable(
                tuned_metrics_from_scores(y[vm], p_glob[vm], y[tm], p_glob[tm]))
        if ex.eval_local:
            # per-client inference: each client scores its own nodes on its own
            # subgraph; virtual nodes are active here for virtual-node variants
            p_loc = _local_scores(global_model, client_datasets, device, use_boundary=use_vn)
            out["test_metrics_local"] = _jsonable(metrics_from_scores(y[tm], p_loc[tm]))
            if ex.eval_tuned_threshold:
                out["test_metrics_local_tuned"] = _jsonable(
                    tuned_metrics_from_scores(y[vm], p_loc[vm], y[tm], p_loc[tm]))
            scores["p_val_local"] = p_loc[vm].astype(np.float32)
            scores["p_test_local"] = p_loc[tm].astype(np.float32)
            if use_vn:
                # same model, virtual-node module bypassed: isolates its effect
                p_nv = _local_scores(global_model, client_datasets, device, use_boundary=False)
                out["test_metrics_local_no_virtual"] = _jsonable(metrics_from_scores(y[tm], p_nv[tm]))
                scores["p_test_local_no_virtual"] = p_nv[tm].astype(np.float32)
        out["_scores"] = scores
    return out


# ==============================================================================
# Full orchestrator — resume by default
# ==============================================================================
def run_all_experiments(
    cfg: Config,
    artifacts_path: Path,
    only_seeds: Optional[List[int]] = None,
    only_variants: Optional[List[str]] = None,
) -> Dict[str, object]:
    """
    v7: only_seeds / only_variants restrict which runs this process executes
    (one shard of the matrix). The config hash is always computed on the full
    config, so every shard of one experiment carries the same hash.
    """
    artifacts_path = Path(artifacts_path)
    artifacts = load_json_if_exists(artifacts_path, default={"runs": {}})
    runs: Dict[str, dict] = artifacts.get("runs", {})

    full_hash = cfg.config_hash()
    code_hash = code_fingerprint()
    env = environment_info()
    # refuse to mix results from a different config or code into this file
    for k, r in runs.items():
        if r.get("status") != "completed":
            continue
        if r.get("config_hash") != full_hash:
            raise RuntimeError(
                f"{artifacts_path} already holds run {k} with config_hash "
                f"{r.get('config_hash')}, current config is {full_hash}. "
                f"Use a new --artifacts-name or output folder.")
        if r.get("code_hash") not in (None, code_hash):
            raise RuntimeError(
                f"{artifacts_path} already holds run {k} produced by code "
                f"{r.get('code_hash')}, current code is {code_hash}.")
    logger.info(f"config_hash={full_hash} | code_hash={code_hash} | env={env}")

    variants = list(cfg.experiment.variants)
    seeds = list(cfg.experiment.seeds)
    if only_variants:
        unknown = set(only_variants) - {v.name for v in variants}
        assert not unknown, f"unknown variants requested: {sorted(unknown)}"
        variants = [v for v in variants if v.name in set(only_variants)]
    if only_seeds:
        unknown = set(only_seeds) - set(seeds)
        assert not unknown, f"seeds not in config: {sorted(unknown)}"
        seeds = [s for s in seeds if s in set(only_seeds)]
    assert variants and seeds, "❌ the experiment matrix is empty (variants/seeds)."
    total = len(variants) * len(seeds)
    done = sum(1 for r in runs.values() if r.get("status") == "completed")
    logger.info(f"📊 start state: {done}/{total} runs already completed (resume by default).")

    # --- config snapshot: built once and validated as JSON right here ---
    try:
        snapshot = _jsonable(cfg.canonical_dict())
        json.dumps(snapshot)
    except Exception as exc:
        logger.warning(f"⚠️ full config snapshot could not be serialized ({exc}); only the hash is saved.")
        snapshot = {"config_hash": cfg.config_hash()}

    # --- raw data loaded once for the whole matrix (cache enabled) ---
    raw = load_elliptic_raw(cfg.data)
    full_data = _first(preprocess_elliptic(raw, cfg.data))
    in_dim = int(full_data.x.shape[1])
    logger.info(
        f"📦 global data ready: N={full_data.num_nodes:,} | "
        f"E={full_data.edge_index.shape[1]:,} | F={in_dim}"
    )
    bundle = {
        "full_data": full_data, "in_dim": in_dim,
        "config_hash": full_hash, "code_hash": code_hash, "environment": env,
        "preds_dir": artifacts_path.parent / "predictions",
    }

    for variant in variants:
        for seed in seeds:
            key = make_run_key(variant.name, seed)
            if runs.get(key, {}).get("status") == "completed" and cfg.experiment.resume:
                logger.info(f"⏭️ skipping completed run: {key}")
                continue

            logger.info(f"🚀 starting run: {key} ({done + 1}/{total})")
            try:
                art = run_single_experiment(cfg, variant, seed, bundle)
                runs[key] = _jsonable(art)
                done += 1
                tm = art.get("test_metrics", {}) or {}
                logger.info(
                    f"✅ completed: {key} | val_ap={art.get('best_val_ap', float('nan')):.4f} | "
                    f"test_ap={float(tm.get('ap', float('nan'))):.4f} | "
                    f"converged={art.get('converged')} | {art.get('runtime_sec')}s"
                )
            except Exception as exc:
                logger.error(f"❌ run failed: {key} — {exc}", exc_info=True)
                runs[key] = {
                    "variant_name": variant.name,
                    "seed": seed,
                    "status": "failed",
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                    "timestamp_utc": utc_timestamp(),
                }

            try:
                atomic_save_json(artifacts_path, {
                    "runs": runs, "config_snapshot": snapshot,
                    "config_hash": full_hash, "code_hash": code_hash})
            except Exception as exc:  # saving must never kill the whole matrix
                logger.error(f"⚠️ saving artifacts failed: {exc}")

    n_ok = sum(1 for r in runs.values() if r.get("status") == "completed")
    n_bad = sum(1 for r in runs.values() if r.get("status") == "failed")
    logger.info(f"🏁 matrix finished: {n_ok} completed, {n_bad} failed, out of {total}.")
    return {"runs": runs}
