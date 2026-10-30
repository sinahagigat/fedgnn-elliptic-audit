"""
Verify that the released code computes exactly what the executed code computed.

Three code versions produced the runs in results/v7. Each is archived byte for
byte in executed_code/ and identified by its code hash (SHA-256 over main.py and
src/**/*.py, see src/utils/common.py:code_fingerprint):

  de756535ee20  protocols A-G, C0-C24 and T
  598ac132e99f  protocols V, N, P, Q, R, L, M, W, S50 and S100
                (adds the option fed.val_split_weighting, docs/CHANGES_REV3.md)
  c73ef066fa6e  protocols Z20, Z50, Z100, ZN and ZM
                (adds the option fed.select_on_aggregate, docs/CHANGES_AGGSEL.md)

The released code adds no option. Its comments, docstrings and messages were
edited, and one hook that no configuration reaches was simplified (see
check 2). This changes the code hash but not the computation. The script checks:

1. every archive reproduces its recorded code hash;
2. for every Python file, the syntax tree of the released file equals the tree
   of the file in each archive, after removing from the released tree the
   options added after that archive. Comments and docstrings are not compared,
   and neither is the text of messages (logging and print calls, assertion and
   exception messages, command-line help); other output text is compared after
   removing decorative symbols and text that was rewritten. The virtual-node
   protection hook in src/fed/client.py is compared only together with check 3;
   any other difference in executable code, identifiers, numbers or strings
   fails the check;
3. every configuration sets virtual.protection to "none", so the simplified
   hook is never entered in a reported run;
4. (if the dependencies are installed) loading every configs/v7_*.yaml with the
   released code gives the configuration hash recorded in results/v7, and the
   legacy configuration reproduces the hashes of the earlier GPU runs in
   results/v6a (docs/CHANGES_V7.md). Options at their default value are left
   out of the hash, so earlier hashes are unchanged.

    python tools/verify_release.py
"""
import ast
import hashlib
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "executed_code"
# archive, its code hash, options added after it
ARCHIVES = [
    (EXE / "fedgnn_v7_executed_de756535ee20.zip", "de756535ee20", ("val_split_weighting", "select_on_aggregate")),
    (EXE / "fedgnn_v7_valsplit_598ac132e99f.zip", "598ac132e99f", ("select_on_aggregate",)),
    (EXE / "fedgnn_v7_aggsel_c73ef066fa6e.zip", "c73ef066fa6e", ()),
]

REWRITTEN = re.compile("[" + chr(0x0600) + "-" + chr(0x06FF) + "]")  # original message text that was rewritten
SYMBOLS = re.compile("(?:[\U0001F000-\U0001FFFF\u2600-\u27BF\u23E9-\u23FA\u2139\u2B50\u2B55\u231A\u231B\u2705]"
                     "\uFE0F?\u200d?)+ ?|\uFE0F")
LOG_CALLS = {"debug", "info", "warning", "warn", "error", "exception", "critical", "print"}
HELP_KEYWORDS = {"help", "description", "epilog"}


class StripOptions(ast.NodeTransformer):
    """Remove the statements that implement the given options."""

    def __init__(self, options):
        self.options = set(options)

    @staticmethod
    def _mentions(node, *names):
        src = ast.unparse(node)
        return any(n in src for n in names)

    def _drop(self, s):
        o = self.options
        if "val_split_weighting" in o:
            if isinstance(s, ast.AnnAssign) and getattr(s.target, "id", "") == "val_split_weighting":
                return True
            if isinstance(s, ast.Assign) and any(getattr(tg, "id", "") == "VAL_SPLIT_SEED" for tg in s.targets):
                return True
            if isinstance(s, ast.FunctionDef) and s.name == "_split_validation_mask":
                return True
            if isinstance(s, ast.Assign) and any(isinstance(tg, ast.Attribute) and tg.attr in ("val_select_mask", "val_weight_mask")
                                                 for tg in s.targets):
                return True
            if isinstance(s, ast.If) and self._mentions(s.test, "val_split_weighting"):
                return True
        if "select_on_aggregate" in o:
            if isinstance(s, ast.AnnAssign) and getattr(s.target, "id", "") == "select_on_aggregate":
                return True
            if isinstance(s, ast.Assign) and any(isinstance(tg, ast.Attribute) and tg.attr == "select_on_aggregate"
                                                 for tg in s.targets):
                return True
            if isinstance(s, ast.FunctionDef) and s.name == "evaluate_aggregate":
                return True
            if isinstance(s, ast.If) and self._mentions(s.test, "select_on_aggregate"):
                return True
        return False

    def _filter(self, body):
        out = []
        for s in body:
            if self._drop(s):
                continue
            if ("val_split_weighting" in self.options and isinstance(s, ast.If)
                    and self._mentions(s.test, "val_weight_mask is self.val_select_mask")):
                out.extend(s.body)
                continue
            out.append(s)
        return out

    def visit(self, tree):
        # remove the added statements first, then map the selection mask back
        for node in ast.walk(tree):
            for field in ("body", "orelse"):
                if isinstance(getattr(node, field, None), list):
                    setattr(node, field, self._filter(getattr(node, field)))
        return _MaskBack().visit(tree) if "val_split_weighting" in self.options else tree


class _MaskBack(ast.NodeTransformer):
    def visit_Attribute(self, node):
        self.generic_visit(node)
        if node.attr == "val_select_mask" and isinstance(node.value, ast.Name) and node.value.id == "self":
            return ast.copy_location(ast.Attribute(value=ast.Attribute(value=ast.Name(id="self", ctx=ast.Load()),
                                                                       attr="data", ctx=ast.Load()),
                                                   attr="val_mask", ctx=node.ctx), node)
        return node


class Normalize(ast.NodeTransformer):
    """Remove what does not affect the computation: docstrings and message text.
    The virtual-node protection hook of the client is removed as well (check 3)."""

    def _body(self, node):
        b = node.body
        if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) and isinstance(b[0].value.value, str):
            node.body = b[1:] or [ast.Pass()]

    def visit_Module(self, node):
        self._body(node); self.generic_visit(node); return node

    def visit_ClassDef(self, node):
        self._body(node); self.generic_visit(node); return node

    def visit_FunctionDef(self, node):
        self._body(node)
        if node.name == "local_train":
            node.body = [s for s in node.body if not _protection_stmt(s)]
            for s in ast.walk(node):
                if isinstance(s, ast.If) and isinstance(getattr(s, "body", None), list):
                    s.body = [x for x in s.body if not _protection_stmt(x)] or [ast.Pass()]
        self.generic_visit(node)
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
        if name in LOG_CALLS:
            node.args = []
            node.keywords = [k for k in node.keywords if k.arg not in ("msg",)]
        node.keywords = [k for k in node.keywords if k.arg not in HELP_KEYWORDS]
        return node

    def visit_Assert(self, node):
        self.generic_visit(node); node.msg = None; return node

    def visit_Raise(self, node):
        self.generic_visit(node)
        if isinstance(node.exc, ast.Call):
            node.exc.args = []
        return node


def _protection_stmt(s):
    """Statements of the virtual-node payload-protection block (old and new form)."""
    if isinstance(s, ast.Assign):
        names = {getattr(t, "id", None) for t in s.targets} | {getattr(t, "attr", None) for t in s.targets}
        return bool(names & {"_vc", "_prot", "protection", "_payload_protection_applied"})
    if isinstance(s, ast.If):
        src = ast.unparse(s.test)
        return "_prot not in" in src or "protection not in" in src or (
            "round_idx" in src and "protection_cfg" in ast.unparse(s))
    return False


def archive_files(path):
    with zipfile.ZipFile(path) as z:
        return {n.split("/", 1)[1]: z.read(n) for n in z.namelist()
                if n.endswith(".py") and "/" in n and "__pycache__" not in n}


def fingerprint(files):
    h = hashlib.sha256()
    keys = sorted(k for k in files if k == "main.py" or k.startswith("src/"))
    for k in keys:
        h.update(k.encode("utf-8")); h.update(files[k])
    return h.hexdigest()[:12]


def same(a, b, path):
    if type(a) is not type(b):
        raise AssertionError(f"{path}: node type {type(a).__name__} != {type(b).__name__}")
    if isinstance(a, ast.Constant):
        if isinstance(a.value, str) and isinstance(b.value, str):
            if REWRITTEN.search(a.value):
                return  # output text that was rewritten
            if SYMBOLS.sub("", a.value) == SYMBOLS.sub("", b.value):
                return  # output text with a symbol removed
        if a.value != b.value or type(a.value) is not type(b.value):
            raise AssertionError(f"{path}: constant {a.value!r} != {b.value!r}")
        return
    if isinstance(a, ast.JoinedStr):
        # an f-string part that consisted only of a symbol disappears with it
        keep = lambda v: not (isinstance(v, ast.Constant) and isinstance(v.value, str) and not SYMBOLS.sub("", v.value))
        same([v for v in a.values if keep(v)], [v for v in b.values if keep(v)], f"{path}.values")
        return
    if isinstance(a, ast.AST):
        for field in a._fields:
            same(getattr(a, field, None), getattr(b, field, None), f"{path}.{field}")
        return
    if isinstance(a, list):
        if len(a) != len(b):
            raise AssertionError(f"{path}: list length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, f"{path}[{i}]")
        return
    if a != b:
        raise AssertionError(f"{path}: {a!r} != {b!r}")


def check_protection():
    import yaml
    bad = []
    for y in sorted((ROOT / "configs").glob("v7_*.yaml")):
        v = (yaml.safe_load(y.read_text(encoding="utf-8")) or {}).get("virtual", {}) or {}
        if str(v.get("protection", "none")).lower() not in ("none", "off", ""):
            bad.append(y.name)
    n = len(list((ROOT / "configs").glob("v7_*.yaml")))
    print(f"3. virtual.protection is 'none' in {n - len(bad)} of {n} configurations", "OK" if not bad else f"NOT IN {bad}")
    return not bad


def check_config_hashes():
    import json
    sys.path.insert(0, str(ROOT / "tools"))
    from jobs import JOBS
    recorded = {}
    for job, spec in JOBS.items():
        f = ROOT / "results" / "v7" / job / "artifacts_slim.json"
        if f.is_file():
            recorded.setdefault(Path(spec["config"]).name, set()).add(
                json.loads(f.read_text(encoding="utf-8"))["config_hash"])
    try:
        sys.path.insert(0, str(ROOT))
        import logging
        logging.disable(logging.CRITICAL)
        from src.compat import install_config_compat
        import src.config as cfg_module
        install_config_compat(cfg_module)
    except ImportError as e:
        print(f"4. config hashes: skipped (missing dependency: {e.name})")
        return True
    print("4. config hashes, released code vs recorded results:")
    ok = True
    for y in sorted((ROOT / "configs").glob("v7_*.yaml")):
        h = cfg_module.load_config(str(y)).config_hash()
        if y.name not in recorded:
            print(f"   {y.name}: {h} (no recorded runs in results/v7)")
            continue
        good = recorded[y.name] == {h}
        ok = ok and good
        print(f"   {y.name}: {h} recorded {sorted(recorded[y.name])}", "OK" if good else "MISMATCH")
    # earlier GPU runs (results/v6a): legacy config with the documented overrides.
    # Options added in v7 did not exist in v6a; they are left at their defaults
    # (which reproduce v6a behaviour) and removed before hashing.
    import hashlib
    v7_only = {"experiment": ["deterministic", "eval_local", "eval_tuned_threshold", "save_predictions"],
               "partition": ["fixed_partition_seed"], "fed": ["centralized_early_stop"]}
    import tempfile
    import yaml
    legacy = yaml.safe_load((ROOT / "configs" / "legacy" / "run_v6a_20seeds.yaml").read_text(encoding="utf-8"))
    legacy["experiment"]["save_checkpoints"] = False
    overrides = {"A": {}, "B": {"strategy": "random_node"}, "C": {"repartition_per_seed": False}}
    for name, over in overrides.items():
        d = json.loads(json.dumps(legacy))
        d["partition"].update(over)
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
            yaml.safe_dump(d, tmp, allow_unicode=True)
        cfg = cfg_module.load_config(tmp.name)
        Path(tmp.name).unlink()
        cd = cfg.canonical_dict()
        for sect, keys in v7_only.items():
            for k in keys:
                assert getattr(getattr(cfg, sect), k) in (False, 0), f"{sect}.{k} is not at its default"
                cd[sect].pop(k, None)
        h = hashlib.sha256(json.dumps(cd, sort_keys=True, ensure_ascii=False, default=str)
                           .encode("utf-8")).hexdigest()[:12]
        runs = json.loads((ROOT / "results" / "v6a" / f"{name}.json").read_text(encoding="utf-8"))["runs"]
        rec = {r["config_hash"] for r in runs.values()}
        good = rec == {h}
        ok = ok and good
        print(f"   legacy run {name}: {h} recorded {sorted(rec)}", "OK" if good else "MISMATCH")
    return ok



def main():
    ok = True
    rel_files = {str(p.relative_to(ROOT)).replace("\\", "/"): p.read_bytes()
                 for p in list((ROOT / "src").rglob("*.py")) + [ROOT / "main.py"] if "__pycache__" not in p.parts}
    print(f"   code hash of the released code: {fingerprint(rel_files)}")
    print("1. code hashes of the executed archives:")
    for path, expected, _ in ARCHIVES:
        fp = fingerprint(archive_files(path))
        print(f"   {path.name}: {fp} (expected {expected})", "OK" if fp == expected else "MISMATCH")
        ok = ok and fp == expected
    for k, (path, expected, options) in enumerate(ARCHIVES, 1):
        removed = f", without {' and '.join(options)}" if options else ""
        print(f"2.{k} syntax trees, released code{removed} vs {expected}:")
        files = archive_files(path); bad = 0
        for rel, raw in sorted(files.items()):
            new = ROOT / rel
            if not new.is_file():
                print(f"   {rel}: missing in release"); ok = False; bad += 1; continue
            try:
                same(Normalize().visit(ast.parse(raw.decode("utf-8"))),
                     Normalize().visit(StripOptions(options).visit(ast.parse(new.read_text(encoding="utf-8")))), rel)
            except AssertionError as e:
                print(f"   {rel}: DIFFERENT -> {e}"); ok = False; bad += 1
        print(f"   {len(files) - bad} of {len(files)} files: identical code")
    ok = check_protection() and ok
    ok = check_config_hashes() and ok
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
