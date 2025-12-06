from __future__ import annotations

import sys
from datetime import datetime
from typing import Literal, TextIO


LogLevel = Literal["DEBUG", "INFO", "WARN", "ERROR"]


class Logger:
    """
    A simple, typed logger used across the project.
    Replaces the broken `Log.write(level, msg)` interface.
    """

    def __init__(self, stream: TextIO = sys.stdout) -> None:
        self.stream = stream

    def _log(self, level: LogLevel, msg: str) -> None:
        timestamp = datetime.utcnow().isoformat()
        self.stream.write(f"[{timestamp}] [{level}] {msg}\n")

    def debug(self, msg: str) -> None:
        self._log("DEBUG", msg)

    def info(self, msg: str) -> None:
        self._log("INFO", msg)

    def warn(self, msg: str) -> None:
        self._log("WARN", msg)

    def error(self, msg: str) -> None:
        self._log("ERROR", msg)
