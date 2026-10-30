"""
Check that a copy of the repository (for example a fresh clone or the ZIP
downloaded from GitHub) contains exactly the files of the released version,
byte for byte. MANIFEST.sha256 lists the SHA-256 of every file.

    python tools/check_manifest.py            # check this folder
    python tools/check_manifest.py --write    # regenerate MANIFEST.sha256 (maintainers only)
"""
import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.sha256"
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv"}
SKIP_FILES = {"MANIFEST.sha256"}


def files():
    for p in sorted(ROOT.rglob("*")):
        rel = p.relative_to(ROOT)
        if p.is_file() and not (set(rel.parts) & SKIP_DIRS) and rel.as_posix() not in SKIP_FILES \
                and not rel.as_posix().startswith("analysis/output/figures/") or (p.is_file() and rel.suffix == ".png"
                and rel.as_posix().startswith("analysis/output/figures/")):
            yield rel.as_posix(), p


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    current = {rel: sha(p) for rel, p in files()}
    if args.write:
        MANIFEST.write_text("".join(f"{h}  {rel}\n" for rel, h in sorted(current.items())), encoding="utf-8")
        print(f"wrote {MANIFEST} ({len(current)} files)")
        return 0
    expected = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        h, rel = line.split("  ", 1)
        expected[rel] = h
    missing = sorted(set(expected) - set(current))
    changed = sorted(r for r in expected if r in current and current[r] != expected[r])
    extra = sorted(set(current) - set(expected))
    print(f"files in manifest: {len(expected)}, found: {len(expected) - len(missing)}")
    for name, lst in [("MISSING", missing), ("DIFFERENT", changed), ("EXTRA (not in the release)", extra)]:
        if lst:
            print(f"{name}: {len(lst)}")
            for r in lst[:30]:
                print("   ", r)
            if len(lst) > 30:
                print(f"    ... and {len(lst) - 30} more")
    ok = not missing and not changed
    print("\nRESULT:", "PASS (identical to the released version)" if ok else "FAIL")
    if ok and extra:
        print("Note: extra files are present; they are not part of the release and can be deleted.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
