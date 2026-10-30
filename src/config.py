# -*- coding: utf-8 -*-
"""Configuration: dataclass-style sections (data, partition, model, fed, virtual,
selection, privacy, blockchain, rust, experiment) and the experiment variants.

load_config() reads a YAML file and validates it; unknown keys are an error.
canonical_dict() / config_hash() give the SHA-256 hash recorded in every run
record. Options added after the first runs are left out of the hash while they
have their default value, so earlier hashes stay valid.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass, field, asdict, fields as dc_fields
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

PROJECT_VERSION: str = "5.0.1"

# Algorithms actually implemented in src/fed/ (from a grep audit).
# Adding a name to this set without an implementation => explicit error in validate().
SUPPORTED_ALGORITHMS: Tuple[str, ...] = (
    "centralized",
    "fedavg",
    "fedprox",
    "fedproto",
    "fedgnn_hard",
    "fedgnn_soft",
)


# 0) Compatibility layer with a Pydantic-style API
class PydanticCompat:
    """train.py is written in Pydantic style; this mixin provides the same API surface."""

    # serialization
    def model_dump(self) -> Dict[str, Any]:
        return asdict(self)  # type: ignore[arg-type]

    def dict(self) -> Dict[str, Any]:
        return self.model_dump()

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    # clone with override
    def model_copy(self, update: Optional[Dict[str, Any]] = None,
                   deep: bool = True) -> Any:
        obj = copy.deepcopy(self) if deep else copy.copy(self)
        for k, v in (update or {}).items():
            if not hasattr(obj, k):
                raise KeyError(
                    f"[config] unknown key for {type(obj).__name__}: '{k}'"
                )
            setattr(obj, k, v)
        return obj

    def copy(self, **kw) -> Any:
        return self.model_copy(update=kw or None)

    # safe access
    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def update(self, **kw) -> None:
        for k, v in kw.items():
            if not hasattr(self, k):
                raise KeyError(f"[config] unknown key: '{k}'")
            setattr(self, k, v)


# 1) Data — Elliptic Bitcoin Transaction Graph
@dataclass
class DataConfig(PydanticCompat):
    raw_dir: str = "/content/data"
    cache_dir: str = "/content/data/cache"

    file_features: str = "elliptic_txs_features.csv"
    file_classes: str = "elliptic_txs_classes.csv"
    file_edgelist: str = "elliptic_txs_edgelist.csv"

    # preprocessing
    keep_labeled_only: bool = True
    make_undirected: bool = True
    remove_self_loops: bool = True
    use_local_features: bool = True          # 93 local features
    use_aggregated_features: bool = True     # 72 aggregated features
    include_timestep_as_feature: bool = False
    feature_scaling: str = "standard"        # {"standard","minmax","none"}
    fit_scaler_on_train_only: bool = True    # prevents data leakage

    # splitting
    split_mode: str = "temporal"             # {"temporal","random_stratified"}
    temporal_train_end: int = 34
    temporal_val_end: int = 39
    random_split_ratios: Tuple[float, float, float] = (0.60, 0.15, 0.25)

    # raw labels
    positive_label_raw: str = "1"            # illicit
    negative_label_raw: str = "2"            # licit
    unknown_label_raw: str = "unknown"

    # reproducibility assertions (counts reported in v4.9.1)
    expected_num_nodes: Optional[int] = 46564
    expected_num_edges_directed: Optional[int] = 36624
    expected_positive_rate: Optional[float] = 0.0976
    assert_tolerance: float = 0.02

    use_cache: bool = True

    # legacy alias
    @property
    def root(self) -> str:
        return self.raw_dir

    @root.setter
    def root(self, v: str) -> None:
        self.raw_dir = v

    def path(self, which: str) -> str:
        m = {"features": self.file_features,
             "classes": self.file_classes,
             "edgelist": self.file_edgelist}
        return os.path.join(self.raw_dir, m[which])


# 2) Federated partition
@dataclass
class PartitionConfig(PydanticCompat):
    strategy: str = "louvain"     # {"louvain","random_node","dirichlet","metis"}
    num_clients: int = 10

    louvain_resolution: float = 1.0
    dirichlet_alpha: float = 0.5
    metis_ufactor: int = 30

    balance_objective: str = "nodes"
    max_size_imbalance: float = 0.25

    # feasibility conditions
    min_nodes_per_client: int = 200
    min_positives_per_client_train: int = 10
    reassign_if_infeasible: bool = True
    max_repartition_attempts: int = 20

    # per-seed partition: genuine structural variation between seeds
    repartition_per_seed: bool = True
    # v7: partition seed used when repartition_per_seed is False
    # (v6a hard-coded 0; run C of the paper used 0).
    fixed_partition_seed: int = 0

    sensitivity_strategies: List[str] = field(
        default_factory=lambda: ["louvain", "random_node"]
    )

    # legacy alias
    @property
    def scheme(self) -> str:
        return self.strategy

    @scheme.setter
    def scheme(self, v: str) -> None:
        self.strategy = v


# 3) Model (the code reads `cfg.arch.lower()`)
@dataclass
class ModelConfig(PydanticCompat):
    arch: str = "gcn"                 # {"gcn","gat","sage","mlp"}  <- main field
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.30
    num_classes: int = 2

    gat_heads: int = 4
    gat_concat: bool = True

    use_batchnorm: bool = False
    use_layernorm: bool = True
    activation: str = "relu"
    residual: bool = False

    # aliases
    @property
    def encoder(self) -> str:
        return self.arch

    @encoder.setter
    def encoder(self, v: str) -> None:
        self.arch = v

    @property
    def out_dim(self) -> int:
        return self.num_classes


# 4) FedConfig — the single reference for the federated training loop
#    All field names come from a grep over src/fed/client.py and src/fed/server.py.
@dataclass
class FedConfig(PydanticCompat):
    # algorithm selection
    algorithm: str = "fedgnn_soft"

    # optimization
    optimizer: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 5e-4
    num_rounds: int = 60
    local_epochs: int = 1
    grad_clip_norm: float = 5.0

    # class imbalance
    use_pos_weight: bool = True
    max_pos_weight: float = 50.0
    loss_type: str = "ce_weighted"     # {"ce_weighted","focal","ce"}
    focal_gamma: float = 2.0

    # early stopping / convergence
    model_select_metric: str = "val_ap"
    early_stop_enabled: bool = True
    patience: int = 20
    min_delta: float = 1e-4
    convergence_window: int = 10
    convergence_rel_tol: float = 5e-3

    # FedProx
    fedprox_mu: float = 0.01

    # contribution 1: prototypes
    prototype_loss_weight: float = 0.10
    prototype_temperature: float = 0.20
    prototype_momentum: float = 0.80
    prototype_distance: str = "cosine"
    prototype_warmup_rounds: int = 3
    proto_enabled: bool = True

    # contribution 1b: quality-based weighting
    quality_weighting: bool = True
    w_exp_size: float = 1.0
    w_exp_balance: float = 0.5
    w_exp_structure: float = 0.5
    w_exp_trust: float = 0.5
    weight_floor: float = 0.02
    weight_normalize: bool = True

    # contribution 2: virtual nodes (mirror of VirtualNodeConfig)
    use_virtual_nodes: bool = True
    virtual_loss_weight: float = 0.05

    # contribution 3: selective aggregation (mirror of SelectionConfig)
    top_k_ratio: float = 0.50
    aggregation_temperature: float = 0.50
    participation_frac: float = 0.50
    min_clients_per_round: int = 3

    amp: bool = False
    # v7: early stopping for the centralized baseline. False keeps the v6a
    # behaviour (always train the full step budget). The convergence flag
    # is now computed with the same patience rule either way.
    centralized_early_stop: bool = False
    # split each client's validation nodes into two stratified
    # halves; the selection score that weights clients is computed on one half
    # and the checkpoint-selection AP on the other. False (default) uses the
    # full validation set for both, exactly as before; the key is then left
    # out of the configuration hash so that earlier hashes are unchanged.
    val_split_weighting: bool = False
    # checkpoint selection and the patience rule use the mean validation AP of
    # the aggregated model on the clients' validation data, instead of the mean
    # validation AP of the local models after local training. Evaluation only:
    # the training trajectory is unchanged. False (default) is the behaviour of
    # every run before protocol Z, and the key is then left out of the hash.
    select_on_aggregate: bool = False

    # alias: parts of the code use cfg.agg.method
    @property
    def method(self) -> str:
        return self.algorithm

    @method.setter
    def method(self, v: str) -> None:
        self.algorithm = v


# legacy aliases — same class, no duplication
TrainConfig = FedConfig
AggregationConfig = FedConfig
AggConfig = FedConfig


# 5) Contribution 2 — Virtual Node Alignment (cross-client boundaries)
@dataclass
class VirtualNodeConfig(PydanticCompat):
    enabled: bool = True
    momentum: float = 0.9  # EMA coefficient for updating virtual-node features (0=memoryless, 1=unchanged)
    residual_weight: float = 0.1  # weight for mixing the virtual-node message into the node representation (residual mixing)
    num_virtual: int = 8
    align_loss_weight: float = 0.05        # <- source of truth; mirrored to fed.virtual_loss_weight

    exchange_payload: str = "mean_var"     # {"mean_var","sketch","embedding"}
    sketch_dim: int = 32

    protection: str = "secure_agg"         # {"none","secure_agg","dp"}
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    dp_clip_norm: float = 1.0

    attach_policy: str = "boundary_degree"
    attach_topk: int = 64
    learnable_virtual_features: bool = True

    run_attack_eval: bool = True
    attack_type: str = "feature_inference"


# 6) Contribution 3 — client selection and communication optimization
@dataclass
class SelectionConfig(PydanticCompat):
    mode: str = "soft"                     # {"all","random","hard","soft"}
    participation_frac: float = 0.50       # <- source of truth
    min_clients_per_round: int = 3         # <- source of truth
    soft_temperature: float = 0.50         # <- mirrored to fed.aggregation_temperature
    topk_ratio: float = 0.50               # <- mirrored to fed.top_k_ratio

    score_components: List[str] = field(
        default_factory=lambda: ["anomaly_density", "loss_gap", "staleness", "trust"]
    )
    score_weights: List[float] = field(
        default_factory=lambda: [0.40, 0.30, 0.20, 0.10]
    )
    staleness_penalty: float = 0.10

    # partial update (bandwidth reduction)
    partial_update: bool = True
    partial_mode: str = "topk_grad"        # {"none","topk_grad","layerwise","subgraph"}
    quantization_bits: Optional[int] = 8
    error_feedback: bool = True

    log_communication_bytes: bool = True
    count_downlink: bool = True


# 7) Privacy
@dataclass
class PrivacyConfig(PydanticCompat):
    dp_enabled: bool = False
    dp_clip_norm: float = 1.0
    dp_noise_multiplier: float = 0.50
    dp_accounting: str = "rdp"
    dp_target_delta: float = 1e-5

    secure_agg_enabled: bool = True
    secure_agg_seed_bits: int = 32
    verify_mask_cancellation: bool = True


# 8) Blockchain (trust and integrity of updates)
@dataclass
class BlockchainConfig(PydanticCompat):
    enabled: bool = False                  # off by default; a dedicated variant turns it on
    consensus: str = "poa"                 # {"poa","pow"}
    pow_difficulty: int = 3
    hash_algo: str = "sha256"
    merkle_enabled: bool = True

    verify_norm_bound: bool = True
    max_update_norm: float = 20.0
    verify_cosine_bound: bool = True
    min_cosine_to_median: float = -0.20
    reject_nan: bool = True

    trust_init: float = 1.0
    trust_reward: float = 0.05
    trust_penalty: float = 0.30
    trust_min: float = 0.05
    trust_max: float = 1.0

    simulate_attack: bool = False
    attack_type: str = "label_flip"
    num_malicious: int = 2

    ledger_path: str = "/content/results/ledger"
    persist_ledger: bool = True
    measure_overhead: bool = True


# 9) Rust / PyO3 extension
@dataclass
class RustConfig(PydanticCompat):
    use_rust: bool = True
    module_name: str = "fedgnn_rs"
    num_threads: int = 0                   # 0 = automatic
    strict_parity_check: bool = True       # numerical parity with the Python path
    parity_atol: float = 1e-6
    benchmark_on_start: bool = False
    benchmark_repeats: int = 5


# 10) VariantSpec — one row of the paper table
#     field names exactly as consumed in src/train.py lines 166–189
@dataclass
class VariantSpec(PydanticCompat):
    name: str
    algorithm: str = "fedavg"
    description: str = ""

    arch: Optional[str] = None             # if None => inherited from cfg.model
    use_virtual_nodes: bool = False
    use_privacy: bool = False
    use_blockchain: bool = False
    use_partial_update: bool = False

    # train.py:  model_cfg = variant.model_overrides or cfg.model
    # => must be a complete ModelConfig; Config._resolve_variants() builds it.
    model_overrides: Optional[ModelConfig] = None

    fed_overrides: Dict[str, Any] = field(default_factory=dict)
    overrides: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    # aliases
    @property
    def encoder(self) -> Optional[str]:
        return self.arch

    @encoder.setter
    def encoder(self, v: Optional[str]) -> None:
        self.arch = v

    @property
    def id(self) -> str:
        return self.name

    @property
    def label(self) -> str:
        return self.name

    @property
    def is_centralized(self) -> bool:
        return self.algorithm == "centralized" or self.name.startswith("Centralized")

    @property
    def is_federated(self) -> bool:
        return not self.is_centralized

    def __str__(self) -> str:  # pragma: no cover
        return f"VariantSpec({self.name} | {self.algorithm})"


# 11) Experiment
def _auto_device() -> str:
    """Device detection; lazy import so that config can be imported without torch."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _default_variants() -> List[VariantSpec]:
    """
    Ablation ladder. Only algorithms implemented in src/fed/
    (SUPPORTED_ALGORITHMS); no 'paper-only' variant is in the default list.
    """
    return [
        VariantSpec(
            name="Centralized_GCN", algorithm="centralized", arch="gcn",
            description="Performance ceiling: centralized training on the full graph.",
            tags=["baseline", "upper_bound"],
        ),
        VariantSpec(
            name="FedAvg_GCN", algorithm="fedavg", arch="gcn",
            description="Standard FedAvg baseline.",
            tags=["baseline"],
        ),
        VariantSpec(
            name="FedProx_GCN", algorithm="fedprox", arch="gcn",
            description="FedProx baseline for statistical heterogeneity.",
            tags=["baseline"],
        ),
        VariantSpec(
            name="FedProto_GCN", algorithm="fedproto", arch="gcn",
            description="Prototype baseline, without quality weighting.",
            tags=["baseline"],
        ),
        VariantSpec(
            name="FedGNN_Hard_GCN", algorithm="fedgnn_hard", arch="gcn",
            description="A1: hard selective aggregation (Top-K) + prototypes.",
            tags=["ablation", "innov3"],
        ),
        VariantSpec(
            name="FedGNN_Soft_GCN", algorithm="fedgnn_soft", arch="gcn",
            description="A2: soft selective aggregation (weighted softmax) + prototypes.",
            tags=["ablation", "innov1", "innov3"],
        ),
        VariantSpec(
            name="FedGNN_Soft_Virtual_GCN", algorithm="fedgnn_soft", arch="gcn",
            use_virtual_nodes=True,
            description="Full proposed model (GCN): soft + prototypes + virtual nodes.",
            tags=["proposed", "full"],
        ),
        VariantSpec(
            name="FedGNN_Soft_Virtual_GAT", algorithm="fedgnn_soft", arch="gat",
            use_virtual_nodes=True,
            description="Full proposed model (GAT): checks independence from the encoder architecture.",
            tags=["proposed", "full", "arch_sensitivity"],
        ),
    ]


@dataclass
class ExperimentConfig(PydanticCompat):
    project_name: str = "FedGNN-Fraud"
    version: str = PROJECT_VERSION
    tag: str = "v5.0.1"

    device: str = field(default_factory=_auto_device)
    seeds: List[int] = field(default_factory=lambda: [42, 1337, 2024])
    variants: List[VariantSpec] = field(default_factory=_default_variants)

    results_dir: str = "/content/results"
    log_dir: str = "/content/results/logs"
    ckpt_dir: str = "/content/results/checkpoints"
    figure_dir: str = "/content/results/figures"
    table_dir: str = "/content/results/tables"

    # Resume by default: completed runs are not executed again
    resume: bool = True
    overwrite: bool = False
    save_per_round_history: bool = True
    # v7 options. Defaults reproduce v6a behaviour; v7 configs switch them on.
    deterministic: bool = False          # torch.use_deterministic_algorithms
    eval_local: bool = False             # per-client inference on client subgraphs
    eval_tuned_threshold: bool = False   # F1/P/R/MCC at a val-tuned threshold
    save_predictions: bool = False       # per-run .npz with val/test scores
    save_checkpoints: bool = True
    log_level: str = "INFO"

    # statistical analysis
    alpha: float = 0.05
    multiple_comparison_correction: str = "holm"
    bootstrap_iters: int = 2000
    tost_margin: float = 0.01
    paired_test: str = "wilcoxon"

    def ensure_dirs(self) -> None:
        for d in (self.results_dir, self.log_dir, self.ckpt_dir,
                  self.figure_dir, self.table_dir):
            Path(d).mkdir(parents=True, exist_ok=True)


# 12) Root Config
@dataclass
class Config(PydanticCompat):
    data: DataConfig = field(default_factory=DataConfig)
    partition: PartitionConfig = field(default_factory=PartitionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    fed: FedConfig = field(default_factory=FedConfig)
    virtual: VirtualNodeConfig = field(default_factory=VirtualNodeConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    blockchain: BlockchainConfig = field(default_factory=BlockchainConfig)
    rust: RustConfig = field(default_factory=RustConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    # live aliases (no data copy)
    @property
    def agg(self) -> FedConfig:
        """cfg.agg.method / cfg.agg.proto_enabled / cfg.agg.quality_weighting"""
        return self.fed

    @property
    def train(self) -> FedConfig:
        return self.fed

    @property
    def version(self) -> str:
        return PROJECT_VERSION

    @property
    def device(self) -> str:
        return self.experiment.device

    @property
    def seeds(self) -> List[int]:
        return self.experiment.seeds

    @property
    def variants(self) -> List[VariantSpec]:
        return self.experiment.variants

    # build / clone
    def clone(self) -> "Config":
        return copy.deepcopy(self)

    # path access
    def get_by_path(self, path: str) -> Any:
        obj: Any = self
        for part in path.split("."):
            obj = getattr(obj, part)
        return obj

    def set_by_path(self, path: str, value: Any) -> None:
        parts = path.split(".")
        obj: Any = self
        for part in parts[:-1]:
            obj = getattr(obj, part)
        leaf = parts[-1]
        if not hasattr(obj, leaf):
            raise KeyError(f"[config] invalid path: '{path}'")
        setattr(obj, leaf, value)

    def apply_overrides(self, overrides: Optional[Dict[str, Any]]) -> "Config":
        for k, v in (overrides or {}).items():
            self.set_by_path(k, v)
        return self

    # synchronization of mirrored parameters
    def _sync(self) -> None:
        """
        server and client receive only fed_cfg, so values from other sections
        are mirrored into fed.
        """
        f, s, v = self.fed, self.selection, self.virtual

        # virtual nodes
        f.use_virtual_nodes = bool(v.enabled)
        f.virtual_loss_weight = float(v.align_loss_weight if v.enabled else 0.0)

        # selection / selective aggregation
        f.top_k_ratio = float(s.topk_ratio)
        f.aggregation_temperature = float(s.soft_temperature)
        f.participation_frac = float(s.participation_frac)
        f.min_clients_per_round = int(s.min_clients_per_round)

        # contribution flags by algorithm
        f.proto_enabled = f.algorithm in ("fedproto", "fedgnn_hard", "fedgnn_soft")
        f.quality_weighting = f.algorithm in ("fedgnn_hard", "fedgnn_soft")

        if f.algorithm == "centralized":
            f.use_virtual_nodes = False
            f.virtual_loss_weight = 0.0
            f.proto_enabled = False
            f.quality_weighting = False

        if not f.proto_enabled:
            f.prototype_loss_weight = 0.0

    # variant completion
    def _resolve_variants(self) -> None:
        """
        train.py:  model_cfg = variant.model_overrides or cfg.model
        => model_overrides must be a *complete* ModelConfig inheriting from cfg.model.
        """
        for var in self.experiment.variants:
            if var.model_overrides is None and var.arch is not None:
                var.model_overrides = self.model.model_copy(
                    update={"arch": var.arch}
                )
            elif var.model_overrides is not None:
                # fill None fields from the base
                base = self.model.model_dump()
                for k, val in var.model_overrides.model_dump().items():
                    if val is None:
                        setattr(var.model_overrides, k, base[k])
                if var.arch is None:
                    var.arch = var.model_overrides.arch

    # validation
    def validate(self) -> "Config":
        self._sync()
        self._resolve_variants()

        d, p, m, f = self.data, self.partition, self.model, self.fed
        s, v, ex = self.selection, self.virtual, self.experiment

        # device
        if ex.device in ("auto", "", None):
            ex.device = _auto_device()
        assert ex.device in ("cpu", "cuda") or ex.device.startswith("cuda:"), \
            f"[config] invalid device: {ex.device}"

        # data
        assert d.feature_scaling in ("standard", "minmax", "none"), \
            f"[config] invalid feature_scaling: {d.feature_scaling}"
        assert d.split_mode in ("temporal", "random_stratified"), \
            f"[config] invalid split_mode: {d.split_mode}"
        if d.split_mode == "temporal":
            assert 1 <= d.temporal_train_end < d.temporal_val_end <= 49, \
                "[config] temporal bounds must satisfy 1 ≤ train_end < val_end ≤ 49."
        else:
            assert abs(sum(d.random_split_ratios) - 1.0) < 1e-6, \
                "[config] random_split_ratios must sum to 1."
        assert d.use_local_features or d.use_aggregated_features, \
            "[config] at least one feature group must be enabled."

        # partition
        _ALIAS = {"random": "random_node", "rand": "random_node",
                  "community": "louvain", "louvain_community": "louvain"}
        p.strategy = _ALIAS.get(p.strategy, p.strategy)
        assert p.strategy in ("louvain", "random_node", "dirichlet", "metis"), \
            f"[config] invalid partition.strategy: {p.strategy}"
        assert p.num_clients >= 1, "[config] num_clients ≥ 1"
        assert p.min_nodes_per_client >= 1
        assert p.min_positives_per_client_train >= 1, \
            "[config] every client must have at least one positive training example."

        # model
        assert m.arch.lower() in ("gcn", "gat", "sage", "mlp"), \
            f"[config] invalid model.arch: {m.arch}"
        assert m.hidden_dim > 0 and m.num_layers >= 1, \
            "[config] hidden_dim > 0 and num_layers ≥ 1"
        assert 0.0 <= m.dropout < 1.0, \
            f"[config] dropout must be in [0,1): {m.dropout}"
        assert m.num_classes >= 2, "[config] num_classes ≥ 2"
        if m.arch.lower() == "gat":
            assert m.gat_heads >= 1, "[config] gat_heads ≥ 1"
            assert m.hidden_dim % m.gat_heads == 0, (
                f"[config] for GAT, hidden_dim ({m.hidden_dim}) must be divisible by "
                f"gat_heads ({m.gat_heads}); otherwise "
                f"the output dimension of the heads does not match hidden_dim."
            )
        assert m.activation in ("relu", "elu", "gelu", "leaky_relu"), \
            f"[config] invalid activation: {m.activation}"
        assert not (m.use_batchnorm and m.use_layernorm), \
            "[config] BatchNorm and LayerNorm must not be enabled at the same time."

        # federated loop
        assert f.algorithm in SUPPORTED_ALGORITHMS, (
            f"[config] algorithm '{f.algorithm}' is not implemented in src/fed/. "
            f"Allowed: {SUPPORTED_ALGORITHMS}"
        )
        assert f.optimizer in ("adam", "adamw", "sgd"), \
            f"[config] invalid optimizer: {f.optimizer}"
        assert f.lr > 0.0, "[config] lr > 0"
        assert f.weight_decay >= 0.0, "[config] weight_decay ≥ 0"
        assert f.num_rounds >= 1, "[config] num_rounds ≥ 1"
        assert f.local_epochs >= 1, "[config] local_epochs ≥ 1"
        assert f.grad_clip_norm > 0.0, "[config] grad_clip_norm > 0"
        assert f.max_pos_weight >= 1.0, (
            f"[config] max_pos_weight must be ≥ 1 (cap on the minority-class weight): "
            f"{f.max_pos_weight}"
        )
        assert f.loss_type in ("ce", "ce_weighted", "focal"), \
            f"[config] invalid loss_type: {f.loss_type}"
        assert f.model_select_metric in ("val_ap", "val_f1", "val_auc", "val_loss"), \
            f"[config] invalid model_select_metric: {f.model_select_metric}"
        assert f.patience >= 1, "[config] patience ≥ 1"
        assert f.min_delta >= 0.0, "[config] min_delta ≥ 0"
        assert f.convergence_window >= 2, "[config] convergence_window ≥ 2"
        assert f.convergence_rel_tol > 0.0, "[config] convergence_rel_tol > 0"
        assert f.patience <= f.num_rounds, (
            f"[config] patience ({f.patience}) is larger than num_rounds ({f.num_rounds}) "
            f"=> early stopping is never triggered."
        )
        if f.algorithm == "fedprox":
            assert f.fedprox_mu > 0.0, (
                "[config] FedProx with mu=0 is exactly FedAvg; "
                "a separate variant requires mu > 0."
            )
        assert f.fedprox_mu >= 0.0, "[config] fedprox_mu ≥ 0"

        # prototypes
        assert f.prototype_distance in ("cosine", "euclidean"), \
            f"[config] invalid prototype_distance: {f.prototype_distance}"
        assert f.prototype_temperature > 0.0, \
            "[config] prototype_temperature > 0 (denominator of the similarity)"
        assert 0.0 <= f.prototype_momentum < 1.0, (
            f"[config] prototype_momentum must be in [0,1): "
            f"{f.prototype_momentum}"
        )
        assert f.prototype_loss_weight >= 0.0, "[config] prototype_loss_weight ≥ 0"
        assert 0 <= f.prototype_warmup_rounds < f.num_rounds, (
            f"[config] prototype_warmup_rounds ({f.prototype_warmup_rounds}) must be "
            f"less than num_rounds ({f.num_rounds})."
        )
        if f.proto_enabled:
            assert f.prototype_loss_weight > 0.0, (
                f"[config] algorithm '{f.algorithm}' is prototype-based but "
                f"prototype_loss_weight is zero => the contribution has no effect."
            )

        # quality-based weighting
        for nm in ("w_exp_size", "w_exp_balance", "w_exp_structure", "w_exp_trust"):
            assert getattr(f, nm) >= 0.0, f"[config] {nm} ≥ 0"
        assert 0.0 <= f.weight_floor < 1.0 / max(p.num_clients, 1), (
            f"[config] weight_floor ({f.weight_floor}) must be less than "
            f"1/num_clients ({1.0 / max(p.num_clients, 1):.4f}), "
            f"otherwise the weight floors push the sum above 1."
        )
        if f.quality_weighting:
            assert (f.w_exp_size + f.w_exp_balance
                    + f.w_exp_structure + f.w_exp_trust) > 0.0, (
                "[config] quality_weighting is enabled but all weight exponents are zero "
                "=> equivalent to FedAvg."
            )

        # client selection
        assert s.mode in ("all", "random", "hard", "soft"), \
            f"[config] invalid selection.mode: {s.mode}"
        assert 0.0 < s.participation_frac <= 1.0, \
            f"[config] participation_frac must be in (0,1]: {s.participation_frac}"
        assert 1 <= s.min_clients_per_round <= p.num_clients, (
            f"[config] min_clients_per_round ({s.min_clients_per_round}) must be between 1 and "
            f"num_clients ({p.num_clients})."
        )
        assert s.soft_temperature > 0.0, "[config] soft_temperature > 0"
        assert 0.0 < s.topk_ratio <= 1.0, \
            f"[config] topk_ratio must be in (0,1]: {s.topk_ratio}"
        assert s.partial_mode in ("none", "topk_grad", "layerwise", "subgraph"), \
            f"[config] invalid partial_mode: {s.partial_mode}"
        assert 0.0 <= s.staleness_penalty <= 1.0, \
            "[config] staleness_penalty must be in [0,1]."
        assert len(s.score_components) == len(s.score_weights), (
            f"[config] length of score_components ({len(s.score_components)}) differs from "
            f"score_weights ({len(s.score_weights)})."
        )
        assert sum(s.score_weights) > 0.0, "[config] score_weights must sum to > 0."
        if s.quantization_bits is not None:
            assert s.quantization_bits in (1, 2, 4, 8, 16), \
                f"[config] invalid quantization_bits: {s.quantization_bits}"
        if s.partial_update:
            assert s.partial_mode != "none", \
                "[config] partial_update is enabled but partial_mode='none'."
        if s.mode in ("hard", "soft"):
            n_sel = max(int(round(s.participation_frac * p.num_clients)),
                        s.min_clients_per_round)
            assert n_sel >= 1, "[config] number of selected clients per round ≥ 1"
            assert n_sel <= p.num_clients, "[config] n_selected ≤ num_clients"

        # virtual nodes
        assert v.num_virtual >= 1, "[config] num_virtual ≥ 1"
        assert v.align_loss_weight >= 0.0, "[config] align_loss_weight ≥ 0"
        assert v.exchange_payload in ("mean_var", "sketch", "embedding"), \
            f"[config] invalid exchange_payload: {v.exchange_payload}"
        assert v.protection in ("none", "secure_agg", "dp"), \
            f"[config] invalid virtual.protection: {v.protection}"
        assert v.attach_policy in ("boundary_degree", "random", "all"), \
            f"[config] invalid attach_policy: {v.attach_policy}"
        assert v.attach_topk >= 1, "[config] attach_topk ≥ 1"
        if v.exchange_payload == "sketch":
            assert v.sketch_dim >= 1, "[config] sketch_dim ≥ 1"
        if v.protection == "dp":
            assert v.dp_epsilon > 0.0 and 0.0 < v.dp_delta < 1.0, \
                "[config] invalid virtual-node DP parameters."
            assert v.dp_clip_norm > 0.0, "[config] virtual.dp_clip_norm > 0"
        if v.enabled:
            assert v.align_loss_weight > 0.0, (
                "[config] virtual nodes are enabled but align_loss_weight is zero "
                "=> the alignment loss produces no gradient."
            )
            assert p.num_clients >= 2, (
                "[config] boundary virtual nodes with a single client are meaningless "
                "(there is no cross-client boundary)."
            )

        # privacy
        pv = self.privacy
        assert pv.dp_accounting in ("rdp", "gdp", "naive"), \
            f"[config] invalid dp_accounting: {pv.dp_accounting}"
        if pv.dp_enabled:
            assert pv.dp_clip_norm > 0.0, "[config] dp_clip_norm > 0"
            assert pv.dp_noise_multiplier > 0.0, (
                "[config] dp_enabled=True but dp_noise_multiplier is zero "
                "=> no protection is applied."
            )
            assert 0.0 < pv.dp_target_delta < 1.0, "[config] dp_target_delta ∈ (0,1)"
        if pv.secure_agg_enabled:
            assert pv.secure_agg_seed_bits >= 16, \
                "[config] secure_agg_seed_bits ≥ 16 (mask seed space)"

        # blockchain
        bc = self.blockchain
        assert bc.consensus in ("poa", "pow"), \
            f"[config] invalid consensus: {bc.consensus}"
        assert bc.hash_algo in ("sha256", "sha3_256", "blake2b"), \
            f"[config] invalid hash_algo: {bc.hash_algo}"
        assert 0.0 < bc.trust_min <= bc.trust_init <= bc.trust_max, (
            f"[config] 0 < trust_min ({bc.trust_min}) ≤ trust_init "
            f"({bc.trust_init}) ≤ trust_max ({bc.trust_max}) must hold."
        )
        assert bc.trust_reward >= 0.0 and bc.trust_penalty >= 0.0, \
            "[config] trust_reward and trust_penalty must be ≥ 0."
        assert bc.max_update_norm > 0.0, "[config] max_update_norm > 0"
        assert -1.0 <= bc.min_cosine_to_median <= 1.0, \
            "[config] min_cosine_to_median must be in [-1,1]."
        if bc.consensus == "pow":
            assert 1 <= bc.pow_difficulty <= 6, (
                "[config] pow_difficulty must be in [1,6] "
                "(higher values make the simulation run time impractical)."
            )
        if bc.simulate_attack:
            assert bc.attack_type in ("label_flip", "sign_flip", "gaussian", "scaling"), \
                f"[config] invalid attack_type: {bc.attack_type}"
            assert 0 < bc.num_malicious < p.num_clients, (
                f"[config] num_malicious ({bc.num_malicious}) must be less than "
                f"num_clients ({p.num_clients})."
            )

        # Rust
        rs = self.rust
        assert rs.num_threads >= 0, "[config] rust.num_threads ≥ 0 (0 = automatic)"
        assert rs.parity_atol > 0.0, "[config] parity_atol > 0"
        assert rs.benchmark_repeats >= 1, "[config] benchmark_repeats ≥ 1"

        # experiment
        assert len(ex.seeds) >= 1, "[config] at least one seed is required."
        assert len(set(ex.seeds)) == len(ex.seeds), \
            f"[config] duplicate seeds: {ex.seeds}"
        assert all(isinstance(sd, int) and sd >= 0 for sd in ex.seeds), \
            "[config] seeds must be non-negative integers."
        assert len(ex.variants) >= 1, "[config] at least one variant is required."
        names = [vr.name for vr in ex.variants]
        assert len(set(names)) == len(names), (
            f"[config] duplicate variant name: "
            f"{[n for n in set(names) if names.count(n) > 1]}"
        )
        for vr in ex.variants:
            assert vr.algorithm in SUPPORTED_ALGORITHMS, (
                f"[config] variant '{vr.name}' uses an unimplemented algorithm: "
                f"'{vr.algorithm}'"
            )
            assert vr.model_overrides is not None, (
                f"[config] model_overrides of variant '{vr.name}' were not completed "
                f"(_resolve_variants has not been run)."
            )
            assert vr.model_overrides.arch.lower() in ("gcn", "gat", "sage", "mlp"), \
                f"[config] invalid arch in variant '{vr.name}'"
            if vr.is_centralized:
                assert not vr.use_virtual_nodes, (
                    f"[config] centralized variant '{vr.name}' cannot have boundary "
                    f"virtual nodes."
                )
            if vr.use_virtual_nodes:
                assert p.num_clients >= 2, (
                    f"[config] variant '{vr.name}' requires virtual nodes but "
                    f"num_clients={p.num_clients}"
                )
            if vr.use_privacy:
                assert pv.dp_enabled or pv.secure_agg_enabled, (
                    f"[config] variant '{vr.name}' has use_privacy=True but "
                    f"neither dp_enabled nor secure_agg_enabled is on."
                )

        assert 0.0 < ex.alpha < 0.5, f"[config] invalid alpha: {ex.alpha}"
        assert ex.multiple_comparison_correction in ("none", "holm", "bonferroni", "bh"), \
            f"[config] invalid correction: {ex.multiple_comparison_correction}"
        assert ex.bootstrap_iters >= 200, (
            "[config] bootstrap_iters ≥ 200 is required for a stable confidence interval."
        )
        assert ex.tost_margin > 0.0, "[config] tost_margin > 0"
        assert ex.paired_test in ("wilcoxon", "ttest"), \
            f"[config] invalid paired_test: {ex.paired_test}"
        assert ex.log_level in ("DEBUG", "INFO", "WARNING", "ERROR"), \
            f"[config] invalid log_level: {ex.log_level}"
        if len(ex.seeds) < 3 and ex.paired_test == "wilcoxon":
            assert False, (
                "[config] a paired Wilcoxon test with fewer than 3 seeds has no power; "
                "add seeds or set paired_test='ttest'."
            )

        # final cross-consistency
        if p.num_clients == 1:
            assert all(vr.is_centralized for vr in ex.variants), (
                "[config] num_clients=1 but a federated variant is in the list."
            )
        if d.split_mode == "temporal" and p.min_positives_per_client_train > 0:
            pass  # actual feasibility is checked with the data in the partition layer

        return self

    # hash and identifier
    _VOLATILE_PATHS: ClassVar[Tuple[str, ...]] = (
        "experiment.device", "experiment.results_dir", "experiment.log_dir",
        "experiment.ckpt_dir", "experiment.figure_dir", "experiment.table_dir",
        "experiment.resume", "experiment.overwrite", "experiment.log_level",
        "data.raw_dir", "data.cache_dir", "blockchain.ledger_path",
        "rust.num_threads", "rust.benchmark_on_start",
    )

    def canonical_dict(self) -> Dict[str, Any]:
        """
        Hashable dictionary: environment fields (paths, device, execution flags) are
        removed so that the configuration hash stays identical across machines and
        represents the scientific meaning of the run.
        """
        d = self.model_dump()
        for path in self._VOLATILE_PATHS:
            sect, key = path.split(".", 1)
            d.get(sect, {}).pop(key, None)
        # options added after the first runs are omitted from the hash at their default value
        if not d.get("fed", {}).get("val_split_weighting", False):
            d.get("fed", {}).pop("val_split_weighting", None)
        if not d.get("fed", {}).get("select_on_aggregate", False):
            d.get("fed", {}).pop("select_on_aggregate", None)
        return d

    def config_hash(self, n: int = 12) -> str:
        blob = json.dumps(self.canonical_dict(), sort_keys=True,
                          ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:n]

    def run_id(self, variant_name: str, seed: int) -> str:
        """Unique run identifier: basis of resume-by-default and artifact naming."""
        return f"{variant_name}__seed{seed}__{self.config_hash(8)}"

    def artifact_path(self, variant_name: str, seed: int) -> Path:
        return Path(self.experiment.results_dir) / f"{self.run_id(variant_name, seed)}.json"

    # Config for a single variant
    def config_for_variant(self, variant: VariantSpec) -> "Config":
        """
        Complete, valid Config of that variant (clone => override => validate).
        In train.py you can either use this method or read
        variant.model_overrides / variant.use_virtual_nodes directly.
        """
        cfg = self.clone()
        cfg.fed.algorithm = variant.algorithm
        cfg.model = copy.deepcopy(variant.model_overrides or self.model)
        cfg.virtual.enabled = bool(variant.use_virtual_nodes)
        cfg.privacy.dp_enabled = bool(variant.use_privacy) and self.privacy.dp_enabled
        cfg.privacy.secure_agg_enabled = (
            bool(variant.use_privacy) and self.privacy.secure_agg_enabled
        )
        cfg.blockchain.enabled = bool(variant.use_blockchain)
        cfg.selection.partial_update = bool(variant.use_partial_update)
        cfg.selection.partial_mode = (
            self.selection.partial_mode if variant.use_partial_update else "none"
        )

        if variant.algorithm == "fedgnn_hard":
            cfg.selection.mode = "hard"
        elif variant.algorithm == "fedgnn_soft":
            cfg.selection.mode = "soft"
        else:
            cfg.selection.mode = "all"
            cfg.selection.participation_frac = 1.0

        if variant.is_centralized:
            cfg.partition.num_clients = 1
            cfg.partition.repartition_per_seed = False
            cfg.virtual.enabled = False
            cfg.blockchain.enabled = False
            cfg.selection.mode = "all"
            cfg.selection.participation_frac = 1.0
            cfg.selection.min_clients_per_round = 1
            cfg.selection.partial_update = False
            cfg.selection.partial_mode = "none"

        for k, val in variant.fed_overrides.items():
            if not hasattr(cfg.fed, k):
                raise KeyError(f"[config] invalid fed_overrides in '{variant.name}': {k}")
            setattr(cfg.fed, k, val)
        cfg.apply_overrides(variant.overrides)

        # restrict the variants to this variant only (so that the run hash is unique)
        cfg.experiment.variants = [copy.deepcopy(variant)]
        return cfg.validate()

    # report
    def summary(self) -> str:
        ex, p, m, f = self.experiment, self.partition, self.model, self.fed
        L = [
            "=" * 74,
            f"  {ex.project_name}  |  version {PROJECT_VERSION}  |  config hash {self.config_hash()}",
            "=" * 74,
            f"  device            : {ex.device}",
            f"  seeds             : {ex.seeds}",
            f"  variants          : {len(ex.variants)}  ×  {len(ex.seeds)} seeds "
            f"=  {len(ex.variants) * len(ex.seeds)} runs",
            f"  data              : {self.data.split_mode} | scaling="
            f"{self.data.feature_scaling} | labelled_only={self.data.keep_labeled_only}",
            f"  partition         : {p.strategy} | clients={p.num_clients} | "
            f"per-seed={p.repartition_per_seed}",
            f"  model             : {m.arch.upper()} | hidden={m.hidden_dim} | "
            f"L={m.num_layers} | dropout={m.dropout}"
            + (f" | heads={m.gat_heads}" if m.arch.lower() == "gat" else ""),
            f"  training          : rounds={f.num_rounds} | local_ep={f.local_epochs} | "
            f"lr={f.lr} | wd={f.weight_decay} | clip={f.grad_clip_norm}",
            f"  model selection   : {f.model_select_metric} | patience={f.patience}",
            f"  prototypes        : enabled={f.proto_enabled} | λ={f.prototype_loss_weight} | "
            f"τ={f.prototype_temperature} | m={f.prototype_momentum}",
            f"  quality weighting : {f.quality_weighting} | floor={f.weight_floor}",
            f"  virtual nodes     : {self.virtual.enabled} | K={self.virtual.num_virtual} | "
            f"λ={self.virtual.align_loss_weight} | protection={self.virtual.protection}",
            f"  client selection  : {self.selection.mode} | frac="
            f"{self.selection.participation_frac} | topk={self.selection.topk_ratio}",
            f"  privacy           : DP={self.privacy.dp_enabled} | "
            f"SecAgg={self.privacy.secure_agg_enabled}",
            f"  blockchain        : {self.blockchain.enabled} | {self.blockchain.consensus}",
            f"  Rust              : {self.rust.use_rust} | parity={self.rust.strict_parity_check}",
            f"  statistics        : α={ex.alpha} | correction={ex.multiple_comparison_correction} | "
            f"bootstrap={ex.bootstrap_iters}",
            "-" * 74,
            "  variants:",
        ]
        for i, vr in enumerate(ex.variants, 1):
            flags = []
            if vr.use_virtual_nodes:  flags.append("VN")
            if vr.use_privacy:        flags.append("PRIV")
            if vr.use_blockchain:     flags.append("BC")
            if vr.use_partial_update: flags.append("PART")
            tag = ("[" + ",".join(flags) + "]") if flags else ""
            L.append(f"   {i:2d}. {vr.name:<32s} {vr.algorithm:<14s} "
                     f"{vr.model_overrides.arch.upper():<5s} {tag}")
        L.append("=" * 74)
        return "\n".join(L)

    def __str__(self) -> str:  # pragma: no cover
        return self.summary()

    # save / load
    def to_yaml(self) -> str:
        import yaml
        return yaml.safe_dump(self.model_dump(), sort_keys=False,
                              allow_unicode=True, default_flow_style=False)

    def save(self, path: Optional[str] = None) -> Path:
        if path is None:
            self.experiment.ensure_dirs()
            path = str(Path(self.experiment.results_dir)
                       / f"config_{PROJECT_VERSION}_{self.config_hash()}.yaml")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.to_yaml(), encoding="utf-8")
        return out

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        return _coerce_config(d)

    @classmethod
    def load(cls, path: str) -> "Config":
        import yaml
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(raw).validate()


# 13) Rebuild from a dictionary (YAML -> dataclass), failing explicitly on unknown keys
_SECTION_TYPES: Dict[str, type] = {
    "data": DataConfig,
    "partition": PartitionConfig,
    "model": ModelConfig,
    "fed": FedConfig,
    "virtual": VirtualNodeConfig,
    "selection": SelectionConfig,
    "privacy": PrivacyConfig,
    "blockchain": BlockchainConfig,
    "rust": RustConfig,
    "experiment": ExperimentConfig,
}

_TUPLE_FIELDS = {("data", "random_split_ratios")}


def _init_section(cls: type, data: Optional[Dict[str, Any]],
                  section: str = "") -> Any:
    """Build a dataclass from a dict; an unknown key raises an error."""
    if data is None:
        return cls()
    valid = {fl.name for fl in dc_fields(cls)}
    unknown = set(data) - valid
    if unknown:
        raise KeyError(
            f"[config] unknown key(s) in section '{section or cls.__name__}': "
            f"{sorted(unknown)}; allowed: {sorted(valid)}"
        )
    kw = dict(data)
    for (sec, fname) in _TUPLE_FIELDS:
        if sec == section and fname in kw and isinstance(kw[fname], list):
            kw[fname] = tuple(kw[fname])
    return cls(**kw)


def _coerce_config(d: Dict[str, Any]) -> Config:
    unknown = set(d) - set(_SECTION_TYPES) - {"version", "PROJECT_VERSION"}
    if unknown:
        raise KeyError(f"[config] unknown section(s) in YAML: {sorted(unknown)}")

    kwargs: Dict[str, Any] = {}
    for sect, cls in _SECTION_TYPES.items():
        raw = d.get(sect)
        if sect != "experiment":
            kwargs[sect] = _init_section(cls, raw, sect)
            continue
        # ExperimentConfig: variants need nested reconstruction
        raw = dict(raw or {})
        var_raw = raw.pop("variants", None)
        exp = _init_section(ExperimentConfig, raw, "experiment")
        if var_raw is not None:
            variants: List[VariantSpec] = []
            for vd in var_raw:
                vd = dict(vd)
                mo = vd.pop("model_overrides", None)
                vs = _init_section(VariantSpec, vd, "variants[]")
                vs.model_overrides = (
                    _init_section(ModelConfig, mo, "model_overrides") if mo else None
                )
                variants.append(vs)
            exp.variants = variants
        kwargs["experiment"] = exp

    return Config(**kwargs)


# 14) Module-level factory — the only permitted entry point for the rest of the project
def get_config(overrides: Optional[Dict[str, Any]] = None,
               yaml_path: Optional[str] = None,
               validate: bool = True,
               **kw: Any) -> Config:
    """
    Build a valid configuration.

        cfg = get_config()
        cfg = get_config({"fed.num_rounds": 5, "partition.num_clients": 4})
        cfg = get_config(yaml_path="results/config_5.0.1_ab12cd34.yaml")
        cfg = get_config(**{"experiment.seeds": [42]})

    Keys use dotted path notation (`section.field`) and every invalid key
    raises KeyError immediately; no ineffective override is silently ignored.
    """
    cfg = Config.load(yaml_path) if yaml_path else Config()
    merged: Dict[str, Any] = {}
    merged.update(overrides or {})
    merged.update({k.replace("__", "."): v for k, v in kw.items()})
    cfg.apply_overrides(merged)
    cfg.experiment.ensure_dirs()
    return cfg.validate() if validate else cfg


# entry-point aliases
build_config = get_config
default_config = get_config


def load_config(path: str) -> Config:
    return Config.load(path)


def smoke_config(**kw: Any) -> Config:
    """Light configuration for a quick end-to-end smoke test (1 seed, 3 rounds)."""
    base = {
        "experiment.seeds": [42],
        "fed.num_rounds": 3,
        "fed.patience": 3,
        "fed.prototype_warmup_rounds": 1,
        "fed.convergence_window": 2,
        "partition.num_clients": 4,
        "selection.min_clients_per_round": 2,
        "experiment.paired_test": "ttest",
        "experiment.bootstrap_iters": 200,
        "rust.use_rust": False,
    }
    base.update(kw)
    cfg = get_config(base)
    cfg.experiment.variants = [
        vr for vr in cfg.experiment.variants
        if vr.name in ("Centralized_GCN", "FedAvg_GCN", "FedGNN_Soft_Virtual_GCN")
    ]
    return cfg.validate()


__all__ = [
    "PROJECT_VERSION", "SUPPORTED_ALGORITHMS", "PydanticCompat",
    "DataConfig", "PartitionConfig", "ModelConfig",
    "FedConfig", "TrainConfig", "AggregationConfig", "AggConfig",
    "VirtualNodeConfig", "SelectionConfig", "PrivacyConfig",
    "BlockchainConfig", "RustConfig",
    "VariantSpec", "ExperimentConfig", "Config",
    "get_config", "build_config", "default_config",
    "load_config", "smoke_config",
]


if __name__ == "__main__":  # pragma: no cover
    c = get_config()
    print(c.summary())
    print("\n— YAML round-trip test —")
    pth = c.save("/content/results/_config_selftest.yaml")
    c2 = Config.load(str(pth))
    assert c2.config_hash() == c.config_hash(), "hash changed after reload!"
    print(f"stable hash: {c.config_hash()}")
    print("\n— variant test —")
    for vr in c.experiment.variants:
        cv = c.config_for_variant(vr)
        print(f"  {vr.name:<32s} → run_id={cv.run_id(vr.name, 42)}")
