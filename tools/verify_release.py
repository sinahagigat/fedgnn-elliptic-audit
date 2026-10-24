"""
Verify that the released code is functionally identical to the code that
produced the paper's results.

The runs reported in the paper were produced by the code archived in
executed_code/fedgnn_v7_executed_de756535ee20.zip, whose code hash (SHA-256
over main.py and src/**/*.py, see src/utils/common.py:code_fingerprint) is
de756535ee20. Before release, comments, docstrings and log/help messages that
were written in Persian were translated to English. Translation changes the
code hash, so this script checks two things:

1. the archived code reproduces the hash de756535ee20, and
2. for every Python file, the abstract syntax tree of the released file equals
   the tree of the executed file, except for string constants that contained
   non-ASCII Persian text in the executed version (docstrings and messages).
   Comments are not part of the syntax tree. Any change to executable code,
   identifiers, numbers or non-translated strings makes the check fail.
3. no Persian text remains in the released Python files, and
4. (if the dependencies are installed) loading every configs/v7_*.yaml with the
   released code gives the config hash recorded in results/v7 for that
   experiment, and the legacy config reproduces the hashes of the earlier
   GPU runs in results/v6a (see docs/CHANGES_V7.md).

    python tools/verify_release.py
"""
import ast
import hashlib
import io
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "executed_code" / "fedgnn_v7_executed_de756535ee20.zip"
EXPECTED = "de756535ee20"
PERSIAN = re.compile("[" + chr(0x0600) + "-" + chr(0x06FF) + "]")  # Arabic-script block


def archive_files():
    with zipfile.ZipFile(ARCHIVE) as z:
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
        if isinstance(a.value, str) and isinstance(b.value, str) and PERSIAN.search(a.value):
            return  # translated docstring or message
        if a.value != b.value or type(a.value) is not type(b.value):
            raise AssertionError(f"{path}: constant {a.value!r} != {b.value!r}")
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


def check_config_hashes():
    import json
    recorded = {}
    for f in sorted((ROOT / "results" / "v7").glob("*/artifacts_slim.json")):
        exp = f.parent.name.split("_")[0] if f.parent.name.startswith("C") else f.parent.name[0]
        recorded.setdefault(exp, set()).add(json.loads(f.read_text(encoding="utf-8"))["config_hash"])
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
        exp = y.stem.split("_")[1]
        h = cfg_module.load_config(str(y)).config_hash()
        good = recorded.get(exp) == {h}
        ok = ok and good
        print(f"   {y.name}: {h} recorded {sorted(recorded.get(exp, []))}", "OK" if good else "MISMATCH")
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
    files = archive_files()
    fp = fingerprint(files)
    print(f"1. code hash of the executed archive: {fp} (expected {EXPECTED})",
          "OK" if fp == EXPECTED else "MISMATCH")
    ok = fp == EXPECTED
    print("2. syntax-tree comparison, released vs executed:")
    for rel, raw in sorted(files.items()):
        new = ROOT / rel
        if not new.is_file():
            print(f"   {rel}: missing in release"); ok = False; continue
        try:
            same(ast.parse(raw.decode("utf-8")), ast.parse(new.read_text(encoding="utf-8")), rel)
            print(f"   {rel}: identical code")
        except AssertionError as e:
            print(f"   {rel}: DIFFERENT -> {e}"); ok = False
    left = [str(p.relative_to(ROOT)) for p in list(ROOT.rglob("*.py"))
            if "executed_code" not in p.parts and PERSIAN.search(p.read_text(encoding="utf-8"))]
    print("3. Persian text remaining in released Python files:", left or "none")
    ok = check_config_hashes() and ok
    print("\nRESULT:", "PASS" if ok and not left else "FAIL")
    return 0 if ok and not left else 1


if __name__ == "__main__":
    sys.exit(main())
