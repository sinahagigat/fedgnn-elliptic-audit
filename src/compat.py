# -*- coding: utf-8 -*-
"""
compat.py — compatibility bridge between attribute names used by consumer code and the actual config fields.
==============================================================================
Problem: modules such as virtual_nodes.py and privacy.py access attributes with the
section prefix (cfg.virtual_momentum) while the config class defines them without the
prefix (momentum). This file installs a fallback __getattr__ on the config classes
that maps the requested name to the actual field.

⚠️ This is a temporary bridge, not a final solution. Every mapping is logged as a
WARNING; the same mappings should eventually be applied as permanent renames in the
code so that the single-source-of-truth principle is not left violated. The function
compat_report() prints the full list of mappings used, for that purpose.
"""

from __future__ import annotations

import dataclasses
import difflib
import inspect
import logging
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("fedgnn.compat")

# Record of every mapping performed, for the final report: "ClassName.requested" -> "actual"
_RESOLVED: Dict[str, str] = {}
_ANNOUNCED: Set[str] = set()

# Words repeated as section-name prefixes/suffixes that can be removed
_SECTION_WORDS = {
    "virtual", "vn", "privacy", "dp", "model", "fed", "data", "partition",
    "selection", "sel", "agg", "aggregation", "blockchain", "chain", "rust",
    "experiment", "exp", "proto", "prototype", "cfg", "config",
}

# Boolean/count prefixes that may be present on one side and not on the other
_AFFIXES = ("num_", "n_", "use_", "enable_", "is_", "do_", "with_")

# Explicit mappings for cases where a structural guess is not enough.
# Order matters: the first name that exists in the class is chosen.
_EXPLICIT: Dict[str, List[str]] = {
    # --- VirtualNodeConfig ---
    "num_virtual_nodes":      ["num_nodes", "n_nodes", "num_virtual", "n_virtual", "count"],
    "virtual_momentum":       ["momentum", "ema_momentum", "update_momentum"],
    "virtual_residual_weight":["residual_weight", "alpha", "mix_weight", "beta"],
    "virtual_loss_weight":    ["loss_weight", "lambda_virtual", "weight"],
    "virtual_temperature":    ["temperature", "tau"],
    # --- PrivacyConfig ---
    "enabled":                ["enable", "enable_dp", "use_dp", "dp_enabled", "active"],
    "clip_norm":              ["dp_clip_norm", "max_grad_norm", "grad_clip_norm",
                               "l2_clip", "clipping_norm"],
    "noise_multiplier":       ["dp_noise_multiplier", "sigma", "noise_std", "noise_sigma"],
    "delta":                  ["dp_delta", "target_delta"],
    "epsilon":                ["dp_epsilon", "target_epsilon"],
    # --- ModelConfig ---
    "use_residual":           ["residual"],
    "residual":               ["use_residual"],
    "hidden":                 ["hidden_dim"],
    "heads":                  ["gat_heads", "num_heads"],
}


# ==============================================================================
# Extract the actual fields of a config object (pydantic v2 / dataclass / plain)
# ==============================================================================
def _field_names(obj: Any) -> List[str]:
    cls = type(obj)
    names: Set[str] = set()

    mf = getattr(cls, "model_fields", None)          # pydantic v2
    if isinstance(mf, dict):
        names |= set(mf.keys())

    if dataclasses.is_dataclass(cls):
        names |= {f.name for f in dataclasses.fields(cls)}

    if not names:                                     # plain class
        try:
            names |= {k for k in vars(obj) if not k.startswith("_")}
        except TypeError:
            pass
    return sorted(names)


# ==============================================================================
# Generate mapping candidates: deterministic and inspectable
# ==============================================================================
def _candidates(name: str) -> List[str]:
    out: List[str] = []

    def add(s: str) -> None:
        if s and s != name and s not in out:
            out.append(s)

    # 1) explicit mapping
    for c in _EXPLICIT.get(name, []):
        add(c)

    toks = name.split("_")

    # 2) remove section-name words from anywhere in the name
    kept = [t for t in toks if t not in _SECTION_WORDS]
    if kept and kept != toks:
        add("_".join(kept))

    # 3) remove boolean/count prefixes
    for p in _AFFIXES:
        if name.startswith(p):
            add(name[len(p):])
            rest = [t for t in name[len(p):].split("_") if t not in _SECTION_WORDS]
            if rest:
                add("_".join(rest))
                add(p + "_".join(rest))

    # 4) add prefixes
    for p in _AFFIXES:
        add(p + name)

    # 5) singular/plural
    if name.endswith("s"):
        add(name[:-1])
    else:
        add(name + "s")

    return out


def _resolve(obj: Any, requested: str) -> Optional[str]:
    """Name of the actual field matching requested, or None."""
    fields = _field_names(obj)
    if not fields:
        return None
    fset = set(fields)

    for cand in _candidates(requested):
        if cand in fset:
            return cand

    # last attempt: strict fuzzy matching (only for names of 6 or more characters)
    if len(requested) >= 6:
        near = difflib.get_close_matches(requested, fields, n=1, cutoff=0.78)
        if near:
            return near[0]
        stripped = "_".join(t for t in requested.split("_") if t not in _SECTION_WORDS)
        if stripped and len(stripped) >= 5:
            near = difflib.get_close_matches(stripped, fields, n=1, cutoff=0.78)
            if near:
                return near[0]
    return None


# ==============================================================================
# Install the fallback __getattr__
# ==============================================================================
def _make_fallback(cls_name: str, original):
    def __getattr__(self, item):                      # only when normal lookup fails
        if item.startswith("_") or item in ("model_fields", "model_config",
                                            "model_computed_fields"):
            if original is not None:
                return original(self, item)
            raise AttributeError(item)

        target = _resolve(self, item)
        if target is not None:
            key = f"{cls_name}.{item}"
            _RESOLVED[key] = target
            if key not in _ANNOUNCED:
                _ANNOUNCED.add(key)
                logger.warning(
                    f"🔗 compatibility mapping: {cls_name}.{item} → {cls_name}.{target} "
                    f"(rename it in the code before final release)"
                )
            return object.__getattribute__(self, target)

        if original is not None:
            try:
                return original(self, item)
            except AttributeError:
                pass

        raise AttributeError(
            f"'{cls_name}' has no field named '{item}' and no compatibility mapping "
            f"was found. Available fields: {_field_names(self)}"
        )

    return __getattr__


def install_config_compat(config_module) -> int:
    """Install the name fallback on all config classes of the given module."""
    n = 0
    for name, obj in vars(config_module).items():
        if not inspect.isclass(obj) or name.startswith("_"):
            continue
        is_cfg = (isinstance(getattr(obj, "model_fields", None), dict)
                  or dataclasses.is_dataclass(obj))
        if not is_cfg:
            continue
        if getattr(obj, "_fedgnn_compat_installed", False):
            continue
        original = obj.__dict__.get("__getattr__", None)
        obj.__getattr__ = _make_fallback(name, original)
        obj._fedgnn_compat_installed = True
        n += 1
    logger.info(f"🧩 config compatibility layer installed on {n} classes.")
    return n


def compat_report() -> Dict[str, str]:
    """List of the mappings actually used; these are the ones to make permanent."""
    if _RESOLVED:
        logger.info("=" * 70)
        logger.info(f"🧩 {len(_RESOLVED)} compatibility mappings used in this run:")
        for k, v in sorted(_RESOLVED.items()):
            cls = k.split(".")[0]
            logger.info(f"   • {k}  →  {cls}.{v}")
        logger.info("=" * 70)
    else:
        logger.info("✅ no compatibility mapping needed; names are aligned.")
    return dict(_RESOLVED)


# ==============================================================================
# Robust model call: reads the name of forward's third parameter from its signature
# ==============================================================================
_FWD_CACHE: Dict[type, Optional[str]] = {}


def _boundary_param(model) -> Optional[str]:
    cls = type(model)
    if cls in _FWD_CACHE:
        return _FWD_CACHE[cls]
    kw: Optional[str] = None
    try:
        params = list(inspect.signature(cls.forward).parameters.values())
        for p in params[3:]:                          # after self, x, edge_index
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY):
                kw = p.name
                break
        if kw is None and any(p.kind == p.VAR_KEYWORD for p in params):
            kw = "boundary_idx"
    except (TypeError, ValueError):
        kw = None
    _FWD_CACHE[cls] = kw
    return kw


def safe_forward(model, x, edge_index, boundary=None):
    """
    Calls model(x, edge_index, <boundary>) with the correct argument name.
    The output is always normalized to (node_emb or None, logits).
    """
    kw = _boundary_param(model)
    if boundary is not None and kw is not None:
        out = model(x, edge_index, **{kw: boundary})
    elif kw is not None:
        out = model(x, edge_index, **{kw: None})
    else:
        out = model(x, edge_index)

    if isinstance(out, (tuple, list)):
        return (out[0] if len(out) > 1 else None), out[-1]
    return None, out
