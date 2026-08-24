"""Canonical serialization, hashing, and exact-roster verification."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Iterable

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    require_regular_file(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_text(value: Any) -> str:
    return json.dumps(
        value,
        indent=2,
        sort_keys=True,
        allow_nan=False,
        ensure_ascii=True,
    ) + "\n"


def write_json(path: Path, value: Any) -> None:
    text = _canonical_json_text(value)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="ascii")
    temporary.replace(path)


def read_canonical_json(path: Path) -> Any:
    require_regular_file(path)
    text = path.read_text(encoding="ascii")
    value = json.loads(
        text,
        parse_constant=lambda token: _reject_constant(token),
        object_pairs_hook=_unique_object,
    )
    if text != _canonical_json_text(value):
        raise RuntimeError(f"JSON is not canonical finite JSON: {path}")
    return value


def _reject_constant(token: str) -> None:
    raise ValueError(f"Non-finite JSON constant is prohibited: {token}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key is prohibited: {key}")
        result[key] = value
    return result


def write_csv(path: Path, frame: pd.DataFrame, columns: Iterable[str] | None = None) -> None:
    output = frame if columns is None else frame.loc[:, list(columns)]
    numeric = output.select_dtypes(include=[np.number])
    if numeric.size and np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError(f"CSV contains an infinite numeric value: {path.name}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    output.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def require_regular_file(path: Path) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise RuntimeError(f"Expected a regular non-symlink file: {path}") from error
    if not stat.S_ISREG(mode):
        raise RuntimeError(f"Expected a regular non-symlink file: {path}")


def regular_file_roster(root: Path) -> set[str]:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"Evaluation root is not a regular directory: {root}")
    files: set[str] = set()
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in directory_names:
            child = base / name
            if child.is_symlink():
                raise RuntimeError(f"Directory symlink is prohibited: {child}")
        for name in file_names:
            path = base / name
            require_regular_file(path)
            files.add(path.relative_to(root).as_posix())
    return files


def verify_exact_roster(root: Path, expected: set[str]) -> None:
    actual = regular_file_roster(root)
    if actual != expected:
        raise RuntimeError(
            "Evaluation file roster mismatch: "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )


def source_hashes(source_root: Path) -> dict[str, str]:
    if source_root.is_symlink() or not source_root.is_dir():
        raise RuntimeError("Static-evaluation source root must be a non-symlink directory")
    result: dict[str, str] = {}
    for directory, directory_names, file_names in os.walk(
        source_root, followlinks=False
    ):
        base = Path(directory)
        directory_names[:] = sorted(
            name for name in directory_names if name != "__pycache__"
        )
        for name in directory_names:
            if (base / name).is_symlink():
                raise RuntimeError(f"Source directory symlink is prohibited: {base / name}")
        for name in sorted(file_names):
            if not name.endswith(".py"):
                continue
            path = base / name
            require_regular_file(path)
            result[path.relative_to(source_root).as_posix()] = sha256_file(path)
    if not result:
        raise RuntimeError("No static-evaluation Python source files were found")
    return dict(sorted(result.items()))


def hash_roster(root: Path, names: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in sorted(names):
        path = root / name
        require_regular_file(path)
        result[name] = sha256_file(path)
    return result
