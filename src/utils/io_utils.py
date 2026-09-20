"""原子写、目录创建、哈希与 json/jsonl/npz 读写封装。

所有落盘操作统一走这里，保证：写入要么完整可见、要么完全不存在（先写临时文件再 rename）。
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """创建目录（幂等）并返回 Path。"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_text(path: str | os.PathLike[str], text: str) -> Path:
    """原子写文本：同目录临时文件 + os.replace。"""
    p = Path(path)
    ensure_dir(p.parent)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=f".{p.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return p


def atomic_write_json(path: str | os.PathLike[str], obj: Any, indent: int = 2) -> Path:
    return atomic_write_text(path, json.dumps(obj, indent=indent, ensure_ascii=False, default=_json_default))


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"不可序列化类型: {type(obj)!r}")


def read_json(path: str | os.PathLike[str]) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def append_jsonl(path: str | os.PathLike[str], records: Iterable[dict]) -> Path:
    """追加 JSONL（每条一行，flush 后 fsync），禁止手工编辑的日志文件也走这里。"""
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False, default=_json_default) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return p


def write_jsonl(path: str | os.PathLike[str], records: Iterable[dict]) -> Path:
    """覆盖写 JSONL（原子）。"""
    p = Path(path)
    ensure_dir(p.parent)
    lines = [json.dumps(rec, ensure_ascii=False, default=_json_default) for rec in records]
    return atomic_write_text(p, "\n".join(lines) + ("\n" if lines else ""))


def read_jsonl(path: str | os.PathLike[str]) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out: list[dict] = []
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def sha256_file(path: str | os.PathLike[str], chunk: int = 1 << 20) -> str:
    """文件内容哈希（用于 checkpoint / stats / manifest 一致性校验）。"""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    """稳定哈希：键排序 + 紧凑分隔符，跨进程一致。"""
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=_json_default)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_npz(path: str | os.PathLike[str], **arrays: np.ndarray) -> Path:
    """原子写 npz（先写临时文件再 rename）。"""
    p = Path(path)
    ensure_dir(p.parent)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=f".{p.name}.", suffix=".tmp.npz")
    os.close(fd)
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return p


def save_npy(path: str | os.PathLike[str], array: np.ndarray) -> Path:
    """原子写单个 .npy（训练侧 processed 数据用 .npy 而非 npz：
    只有 .npy 支持真正的 mmap 懒加载，npz 每次访问成员都会解压整个数组）。"""
    p = Path(path)
    ensure_dir(p.parent)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=f".{p.name}.", suffix=".tmp.npy")
    os.close(fd)
    try:
        with open(tmp, "wb") as fh:
            np.save(fh, array, allow_pickle=False)
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return p


def relpath_or_abs(path: str | os.PathLike[str], root: str | os.PathLike[str]) -> str:
    """相对 root 的展示路径（不在 root 内则返回绝对路径）。"""
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(Path(path).resolve())
