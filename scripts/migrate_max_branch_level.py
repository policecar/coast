#!/usr/bin/env python3
"""Migrate configs and checkpoints to the tree-depth name max_branch_level.

Since #206 every model names the dendrite tree depth L `max_branch_level`
(SPEC.md M.1.5). The old names are refused, not aliased:

  - a config with `dendrite_branch_depth` (ngm3) or `max_branch_depth` (ngm25)
    fails to load: "<path> is now max_branch_level";
  - an ngm3 checkpoint recorded its depth as the scalar dataset
    `dendrite_branch_depth`, and loading it fails: "max_branch_level missing".

This script renames them in place:

  - *.json: every object key `dendrite_branch_depth` / `max_branch_depth`
    becomes `max_branch_level`. The edit is textual (only the key token
    changes, formatting and order stay), then checked by parsing: the result
    must equal the original with the keys renamed, and no object may end up
    with the key twice. An object that already has both names is reported and
    the file is left alone.
  - *.h5 / *.hdf5: every dataset named `dendrite_branch_depth` (ngm3
    checkpoints; ngm25 never stored the depth) is renamed (a link rename, the
    data are not rewritten); a group that already has the new name is
    reported, dry run included, and the file is left alone. The `config`
    attribute some files embed (the
    config a run used) is left as written: it is a record, and coast only
    displays it.

Nothing else is touched. Paths may be files or directories (searched
recursively; symlinks are not followed, so a run directory's `data` link into
the datasets is skipped).

Without --apply nothing is written: the script lists what it would change.
With --apply each changed file is first copied to <file>.pre-max_branch_level
(unless --no-backup); an existing backup is never overwritten.

Exit status: 0 = done (or nothing to do), 1 = a file could not be migrated
(conflict, parse error, unreadable HDF5), 2 = usage error.

Examples:
  scripts/migrate_max_branch_level.py /mnt/NAS/coast-experiments/basic/ngm25_gpu/LRI_sweep
  scripts/migrate_max_branch_level.py --apply my_run/config.json my_run/snapshots
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

NEW = "max_branch_level"
OLD = ("dendrite_branch_depth", "max_branch_depth")
BACKUP_SUFFIX = ".pre-max_branch_level"

# An object key: the quoted name followed (after optional whitespace) by ':'.
# A string value or a key that merely contains the name does not match.
KEY_RE = re.compile(r'"(' + "|".join(OLD) + r')"(\s*):')


class MigrationError(Exception):
    pass


# ---------------------------------------------------------------- JSON ----

def _load_pairs(text: str, where: str):
    """Parse keeping every key/value pair (duplicates included) in order."""
    try:
        return json.loads(text, object_pairs_hook=lambda pairs: ("obj", pairs))
    except json.JSONDecodeError as e:
        raise MigrationError(f"{where}: not valid JSON ({e})") from e


def _renamed(tree):
    """The parsed tree with the old keys renamed (the expected result)."""
    if isinstance(tree, tuple) and tree[:1] == ("obj",):
        return ("obj", [(NEW if k in OLD else k, _renamed(v)) for k, v in tree[1]])
    if isinstance(tree, list):
        return [_renamed(v) for v in tree]
    return tree


def _check_objects(tree, path: str, where: str):
    """Refuse an object holding max_branch_level twice (old and new name)."""
    if isinstance(tree, tuple) and tree[:1] == ("obj",):
        keys = [k for k, _ in tree[1]]
        if keys.count(NEW) > 1:
            raise MigrationError(
                f"{where}: {path or '/'} has both an old name and {NEW}; "
                f"left unchanged, resolve by hand")
        for k, v in tree[1]:
            _check_objects(v, f"{path}/{k}", where)
    elif isinstance(tree, list):
        for i, v in enumerate(tree):
            _check_objects(v, f"{path}/{i}", where)


def migrate_json(path: Path):
    """Returns (new_text, n_renamed); n_renamed == 0 means nothing to do."""
    text = path.read_text(encoding="utf-8")
    new_text, n = KEY_RE.subn(lambda m: f'"{NEW}"{m.group(2)}:', text)
    if n == 0:
        return text, 0
    where = str(path)
    before = _load_pairs(text, where)
    after = _load_pairs(new_text, where)
    _check_objects(after, "", where)
    if after != _renamed(before):
        raise MigrationError(f"{where}: the textual rename changed more than the keys; "
                             f"left unchanged")
    return new_text, n


def write_text_atomically(path: Path, text: str):
    st = path.stat()
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.chmod(tmp, st.st_mode & 0o7777)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


# ---------------------------------------------------------------- HDF5 ----

def find_h5_fields(path: Path):
    """Paths of datasets with an old name; [] if none."""
    import h5py
    hits = []
    try:
        with h5py.File(path, "r") as f:
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset) and name.rsplit("/", 1)[-1] in OLD:
                    parent = name.rpartition("/")[0]
                    if NEW in (f[parent] if parent else f):
                        raise MigrationError(f"{path}: /{parent} has both {name} and {NEW}; "
                                             f"left unchanged, resolve by hand")
                    hits.append(name)
            f.visititems(visit)
    except OSError as e:
        raise MigrationError(f"{path}: cannot read as HDF5 ({e})") from e
    return hits


def migrate_h5(path: Path, fields):
    import h5py
    with h5py.File(path, "r+") as f:
        for name in fields:
            parent, _, leaf = name.rpartition("/")
            (f[parent] if parent else f).move(leaf, NEW)


# ---------------------------------------------------------------- main ----

def candidates(paths):
    for p in paths:
        if p.is_symlink():
            continue
        if p.is_file():
            yield p
        elif p.is_dir():
            for root, dirs, files in os.walk(p):   # followlinks=False
                dirs.sort()
                for name in sorted(files):
                    f = Path(root) / name
                    if not f.is_symlink() and not name.endswith(BACKUP_SUFFIX):
                        yield f
        else:
            raise MigrationError(f"{p}: no such file or directory")


def backup(path: Path):
    b = path.with_name(path.name + BACKUP_SUFFIX)
    if not b.exists():
        shutil.copy2(path, b)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="See the module docstring (head of this file) for details.")
    ap.add_argument("paths", nargs="+", type=Path, help="files or directories")
    ap.add_argument("--apply", action="store_true",
                    help="write the changes (default: only report them)")
    ap.add_argument("--no-backup", action="store_true",
                    help=f"do not keep <file>{BACKUP_SUFFIX} copies")
    args = ap.parse_args(argv)

    changed = failed = 0
    verb = "migrated" if args.apply else "would migrate"
    try:
        files = list(candidates(args.paths))
    except MigrationError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    for f in files:
        suffix = f.suffix.lower()
        try:
            if suffix == ".json":
                new_text, n = migrate_json(f)
                what = f"{n} key{'s' if n != 1 else ''}"
                write = lambda: write_text_atomically(f, new_text)
            elif suffix in (".h5", ".hdf5"):
                fields = find_h5_fields(f)
                n = len(fields)
                what = ", ".join("/" + x for x in fields)
                write = lambda: migrate_h5(f, fields)
            else:
                continue
            if n == 0:
                continue
            if args.apply:
                if not args.no_backup:
                    backup(f)
                write()
            print(f"{verb}: {f} ({what})")
            changed += 1
        except MigrationError as e:
            print(f"error: {e}", file=sys.stderr)
            failed += 1

    print(f"{len(files)} file(s) scanned, {changed} {verb.split()[-1] if args.apply else 'to migrate'}"
          f", {failed} failed" + ("" if args.apply or changed == 0 else "; rerun with --apply"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
