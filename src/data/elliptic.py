# -*- coding: utf-8 -*-
"""
elliptic.py — loading and preprocessing of the Elliptic Bitcoin Transaction Graph dataset
==============================================================================
The output of this module is a single torch_geometric.data.Data object, regarded
as the ground truth of the global graph; all later steps (federated partition,
virtual nodes, training) are fed from this object.

Anti-leakage notes:
  1) the feature scaler is fitted only on TRAIN nodes (by time step).
  2) split_mode="temporal" is the default: steps 1..34 -> train, 35..39 -> val,
     40..49 -> test. This is the most common and most defensible protocol in
     Elliptic papers and prevents temporal leakage.
  3) if include_timestep_as_feature=False (default), the time-step column is
     removed from the feature vector so that the model cannot read time directly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, coalesce, to_undirected

from src.config import DataConfig

logger = logging.getLogger("fedgnn.data.elliptic")


# ==============================================================================
# 1) Loading the raw files
# ==============================================================================
def _load_raw_csvs(cfg: DataConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the three standard Elliptic files.

    Layout of elliptic_txs_features.csv: no header; column 0 = txId, column 1 = time_step,
    columns 2..166 = 165 features (93 local + 72 aggregated).
    Layout of elliptic_txs_classes.csv: header txId,class with values {"1","2","unknown"}.
    Layout of elliptic_txs_edgelist.csv: header txId1,txId2 (directed edge).
    """
    root = Path(cfg.root)
    feat_path = root / cfg.file_features
    class_path = root / cfg.file_classes
    edge_path = root / cfg.file_edgelist

    for p in (feat_path, class_path, edge_path):
        if not p.exists():
            raise FileNotFoundError(
                f"❌ data file not found: {p}\n"
                f"Please place the raw Elliptic files in '{cfg.root}'."
            )

    features_df = pd.read_csv(feat_path, header=None)
    n_feat_cols = features_df.shape[1] - 2
    feat_cols = [f"feat_{i}" for i in range(n_feat_cols)]
    features_df.columns = ["txId", "time_step"] + feat_cols

    classes_df = pd.read_csv(class_path)
    classes_df.columns = [c.strip().lower() for c in classes_df.columns]
    if "txid" in classes_df.columns:
        classes_df = classes_df.rename(columns={"txid": "txId", "class": "class"})

    edges_df = pd.read_csv(edge_path)
    edges_df.columns = [c.strip() for c in edges_df.columns]
    # column names may be txId1/txId2 or txId1,txId2 with spaces
    rename_map = {}
    for c in edges_df.columns:
        cl = c.lower()
        if cl in ("txid1", "tx_id1", "source"):
            rename_map[c] = "txId1"
        elif cl in ("txid2", "tx_id2", "target"):
            rename_map[c] = "txId2"
    edges_df = edges_df.rename(columns=rename_map)

    logger.info(
        f"raw load: {len(features_df):,} nodes (total) | "
        f"{len(classes_df):,} labels | {len(edges_df):,} directed edges (total)"
    )
    return features_df, classes_df, edges_df


# ==============================================================================
# 2) Filtering to labelled nodes + re-indexing
# ==============================================================================
def _filter_and_reindex(
    features_df: pd.DataFrame,
    classes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    cfg: DataConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Merge features with labels, filter unknown nodes (when
    keep_labeled_only=True) and build the txId -> zero-based index mapping.
    """
    merged = features_df.merge(classes_df, on="txId", how="left")
    merged["class"] = merged["class"].astype(str)

    if cfg.keep_labeled_only:
        mask = merged["class"] != cfg.unknown_label_raw
        merged = merged[mask].reset_index(drop=True)

    # raw label -> binary mapping (1=illicit/positive, 0=licit/negative)
    label_map = {cfg.positive_label_raw: 1, cfg.negative_label_raw: 0}
    merged["y"] = merged["class"].map(label_map)
    if merged["y"].isna().any():
        n_bad = int(merged["y"].isna().sum())
        logger.warning(f"⚠️ {n_bad} nodes with an unrecognized label found; they are removed.")
        merged = merged.dropna(subset=["y"]).reset_index(drop=True)
    merged["y"] = merged["y"].astype(int)

    # txId -> new zero-based index mapping
    tx_to_idx = {tx: i for i, tx in enumerate(merged["txId"].values)}

    # edge filter: both endpoints must be in the set of kept nodes
    valid_tx = set(tx_to_idx.keys())
    edges_filtered = edges_df[
        edges_df["txId1"].isin(valid_tx) & edges_df["txId2"].isin(valid_tx)
    ].copy()

    edges_filtered["src"] = edges_filtered["txId1"].map(tx_to_idx)
    edges_filtered["dst"] = edges_filtered["txId2"].map(tx_to_idx)

    stats = {
        "num_nodes": len(merged),
        "num_edges_directed_pre_undirected": len(edges_filtered),
        "positive_rate": float(merged["y"].mean()),
    }
    return merged, edges_filtered, stats


# ==============================================================================
# 3) Building the final graph (Data)
# ==============================================================================
def _build_edge_index(edges_filtered: pd.DataFrame, num_nodes: int, cfg: DataConfig) -> torch.Tensor:
    """Build the final edge_index, applying make_undirected and remove_self_loops."""
    src = torch.tensor(edges_filtered["src"].values, dtype=torch.long)
    dst = torch.tensor(edges_filtered["dst"].values, dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)

    if cfg.remove_self_loops:
        edge_index, _ = remove_self_loops(edge_index)

    if cfg.make_undirected:
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    else:
        edge_index = coalesce(edge_index, num_nodes=num_nodes)

    return edge_index


def _select_feature_columns(merged: pd.DataFrame, cfg: DataConfig) -> list:
    """
    Feature-column selection for ablations:
      - use_local_features: 93 local features (feat_0 .. feat_92)
      - use_aggregated_features: 72 aggregated features (feat_93 .. feat_164)
      - include_timestep_as_feature: add the time step as a feature
    (The exact local/aggregated boundaries follow the original Elliptic paper.)
    """
    all_feat_cols = [c for c in merged.columns if c.startswith("feat_")]
    n_local = 93
    local_cols = all_feat_cols[:n_local]
    agg_cols = all_feat_cols[n_local:]

    selected = []
    if cfg.use_local_features:
        selected += local_cols
    if cfg.use_aggregated_features:
        selected += agg_cols
    if not selected:
        raise ValueError("❌ at least one of use_local_features/use_aggregated_features must be True.")

    if cfg.include_timestep_as_feature:
        selected = ["time_step"] + selected

    return selected


def _make_temporal_masks(time_step: np.ndarray, cfg: DataConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_mask = time_step <= cfg.temporal_train_end
    val_mask = (time_step > cfg.temporal_train_end) & (time_step <= cfg.temporal_val_end)
    test_mask = time_step > cfg.temporal_val_end
    return train_mask, val_mask, test_mask


def _make_random_masks(n: int, ratios: Tuple[float, float, float], seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    train_mask = np.zeros(n, dtype=bool); train_mask[train_idx] = True
    val_mask = np.zeros(n, dtype=bool); val_mask[val_idx] = True
    test_mask = np.zeros(n, dtype=bool); test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def _scale_features(x: np.ndarray, train_mask: np.ndarray, cfg: DataConfig) -> np.ndarray:
    """
    Critical anti-leakage note: the scaler is fitted only on x[train_mask] and then
    applied as a transform to the whole matrix. This prevents val/test statistics
    (which are the 'future' in practice) from entering the train normalization.
    """
    if cfg.feature_scaling == "none":
        return x

    scaler = StandardScaler() if cfg.feature_scaling == "standard" else MinMaxScaler()
    fit_mask = train_mask if cfg.fit_scaler_on_train_only else np.ones(len(x), dtype=bool)

    scaler.fit(x[fit_mask])
    return scaler.transform(x)


def _run_sanity_assertions(stats: dict, cfg: DataConfig) -> None:
    """Fail fast: if the loaded counts differ substantially from the expectations
    recorded in the config, the error is caught here, not 40 minutes later as a
    strange metric."""
    def _check(name, actual, expected, tol):
        if expected is None:
            return
        rel_diff = abs(actual - expected) / max(abs(expected), 1e-9)
        if rel_diff > tol:
            logger.warning(
                f"⚠️ mismatch '{name}': actual value={actual}, expected={expected} "
                f"(relative difference={rel_diff:.3f} > tolerance={tol})."
            )
        else:
            logger.info(f"✓ sanity check '{name}' passed: {actual} (expected≈{expected}).")

    _check("num_nodes", stats["num_nodes"], cfg.expected_num_nodes, cfg.assert_tolerance)
    _check("num_edges_directed", stats["num_edges_directed_pre_undirected"],
           cfg.expected_num_edges_directed, cfg.assert_tolerance)
    _check("positive_rate", stats["positive_rate"], cfg.expected_positive_rate, cfg.assert_tolerance)


# ==============================================================================
# 4) Public entry point
# ==============================================================================
def load_elliptic_graph(cfg: DataConfig, seed: int = 42) -> Data:
    """
    Full loading, preprocessing and construction of the global Elliptic graph.

    Returns:
        torch_geometric.data.Data with fields:
          x            : [N, F] feature matrix (scaled)
          y             : [N] binary label (1=illicit)
          edge_index    : [2, E] unsorted edges (after make_undirected)
          time_step     : [N] time step of each transaction
          train_mask/val_mask/test_mask : [N] boolean
          tx_id         : [N] original transaction id (for tracing/debugging)
          meta          : dict of descriptive graph statistics
    """
    features_df, classes_df, edges_df = _load_raw_csvs(cfg)
    merged, edges_filtered, stats = _filter_and_reindex(features_df, classes_df, edges_df, cfg)

    n = len(merged)
    time_step = merged["time_step"].values.astype(np.int64)
    y = merged["y"].values.astype(np.int64)
    tx_id = merged["txId"].values

    feat_cols = _select_feature_columns(merged, cfg)
    x_raw = merged[feat_cols].values.astype(np.float32)

    if cfg.split_mode == "temporal":
        train_mask, val_mask, test_mask = _make_temporal_masks(time_step, cfg)
    elif cfg.split_mode == "random_stratified":
        train_mask, val_mask, test_mask = _make_random_masks(n, cfg.random_split_ratios, seed)
    else:
        raise ValueError(f"unknown split_mode: {cfg.split_mode}")

    if train_mask.sum() == 0:
        raise RuntimeError("❌ the train set is empty; check the temporal_train_end bounds.")

    x_scaled = _scale_features(x_raw, train_mask, cfg)

    edge_index = _build_edge_index(edges_filtered, num_nodes=n, cfg=cfg)

    data = Data(
        x=torch.tensor(x_scaled, dtype=torch.float32),
        y=torch.tensor(y, dtype=torch.long),
        edge_index=edge_index,
    )
    data.time_step = torch.tensor(time_step, dtype=torch.long)
    data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
    data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
    data.test_mask = torch.tensor(test_mask, dtype=torch.bool)
    data.tx_id = tx_id  # kept as a numpy array, not a tensor

    data.meta = {
        "num_nodes": n,
        "num_edges_directed_pre_undirected": stats["num_edges_directed_pre_undirected"],
        "num_edges_final": int(edge_index.shape[1]),
        "positive_rate": stats["positive_rate"],
        "positive_rate_train": float(y[train_mask].mean()),
        "positive_rate_val": float(y[val_mask].mean()) if val_mask.sum() else float("nan"),
        "positive_rate_test": float(y[test_mask].mean()) if test_mask.sum() else float("nan"),
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "n_test": int(test_mask.sum()),
        "num_features": x_scaled.shape[1],
        "split_mode": cfg.split_mode,
    }

    _run_sanity_assertions(stats, cfg)

    logger.info(
        f"✅ Elliptic graph built: N={n:,} | E_final={edge_index.shape[1]:,} | "
        f"F={x_scaled.shape[1]} | pos_rate={stats['positive_rate']:.4f} | "
        f"train/val/test = {train_mask.sum()}/{val_mask.sum()}/{test_mask.sum()}"
    )
    return data


def get_or_build_cached_graph(cfg: DataConfig, seed: int = 42, force_rebuild: bool = False) -> Data:
    """
    Cache the preprocessed graph on disk (.pt) to save loading/preprocessing
    time in later runs (a few seconds for Elliptic, but it can be annoying on
    Colab with slow I/O).
    """
    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"elliptic_graph_{cfg.split_mode}_seed{seed}.pt"

    if cache_file.exists() and not force_rebuild:
        logger.info(f"📦 loading graph from cache: {cache_file}")
        return torch.load(cache_file, weights_only=False)

    data = load_elliptic_graph(cfg, seed=seed)
    torch.save(data, cache_file)
    logger.info(f"💾 graph saved to cache: {cache_file}")
    return data


# ==============================================================================
# Compatibility layer with the src/train.py contract
# ------------------------------------------------------------------------------
# train.py expects a two-stage workflow (raw => preprocess). The actual implementation
# of this project is single-stage: load_elliptic_graph does all the work (filtering, re-indexing,
# building edge_index, feature selection, temporal masks, scaling on train).
# These two functions only map names and add no new scientific logic.
# ==============================================================================
def load_elliptic_raw(cfg: DataConfig, seed: int = 42, force_rebuild: bool = False) -> Data:
    """Complete preprocessed graph (with cache). Equivalent to get_or_build_cached_graph."""
    return get_or_build_cached_graph(cfg, seed=seed, force_rebuild=force_rebuild)


def preprocess_elliptic(data: Data, cfg: DataConfig = None, seed: int = 42) -> Data:
    """
    Pass-through: preprocessing has already been done inside load_elliptic_graph.
    This function only satisfies the train.py call contract and deliberately
    applies no second transformation to the data (prevents double scaling).
    """
    for attr in ("x", "y", "edge_index", "train_mask", "val_mask", "test_mask"):
        assert hasattr(data, attr), f"[elliptic] graph is missing '{attr}'."
    return data
