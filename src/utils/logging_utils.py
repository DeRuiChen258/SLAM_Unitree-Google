"""统一日志：控制台 + JSONL 双写。

字段约定：timestamp / level / module / stage / step / context。
阶段编号前缀（如 20_train）与 scripts/ 编号一致，便于 evidence 索引配对。
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io_utils import append_jsonl, ensure_dir


class StageLogger:
    """带阶段编号的结构化日志器。"""

    def __init__(self, module: str, log_dir: str | os.PathLike | None = None, stage: str = "",
                 console: bool = True, jsonl_name: str | None = None) -> None:
        self.module = module
        self.stage = stage
        self.console = console
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            ensure_dir(self.log_dir)
        self.jsonl_path = (self.log_dir / (jsonl_name or f"{stage or module}.jsonl")) if self.log_dir else None
        self._buffer: list[dict] = []

    # -- 文本日志 ---------------------------------------------------------------
    def _emit(self, level: str, message: str, **context: Any) -> dict:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "t_mono": time.monotonic(),
            "level": level,
            "module": self.module,
            "stage": self.stage,
            "message": message,
            "context": context,
        }
        if self.console:
            prefix = f"[{level:<5}] {self.stage + ' ' if self.stage else ''}{self.module}: "
            sys.stderr.write(prefix + message + "\n")
            sys.stderr.flush()
        return record

    def info(self, message: str, **context: Any) -> None:
        self.metric(self._emit("INFO", message, **context))

    def warning(self, message: str, **context: Any) -> None:
        self.metric(self._emit("WARN", message, **context))

    def error(self, message: str, **context: Any) -> None:
        self.metric(self._emit("ERROR", message, **context))

    # -- JSONL 指标 -------------------------------------------------------------
    def metric(self, record: dict) -> None:
        if self.jsonl_path is None:
            return
        self._buffer.append(record)
        if len(self._buffer) >= 32:
            self.flush()

    def metric_now(self, **fields: Any) -> None:
        rec = self._emit("METRIC", fields.pop("message", ""), **fields)
        rec.update(fields)
        self.metric(rec)

    def flush(self) -> None:
        if self.jsonl_path is not None and self._buffer:
            append_jsonl(self.jsonl_path, self._buffer)
            self._buffer.clear()

    def __enter__(self) -> "StageLogger":
        return self

    def __exit__(self, *exc_info) -> None:
        self.flush()


def banner(title: str, width: int = 78, char: str = "=") -> str:
    """阶段标题横幅（终端可读）。"""
    line = char * width
    return f"\n{line}\n{title}\n{line}"
