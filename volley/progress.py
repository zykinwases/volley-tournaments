"""Однострочный индикатор прогресса для длительных расчётов."""

from __future__ import annotations

import sys
import time
from typing import TextIO


class ProgressBar:
    """Индикатор прогресса, который перерисовывает одну строку.

    По умолчанию пишет в ``stderr``, чтобы не смешиваться с основным
    результатом в ``stdout``. Обновления ограничены по времени, поэтому
    частые вызовы не замедляют расчёт и не мерцают в терминале.
    """

    def __init__(
        self,
        total: int,
        *,
        label: str = "Симуляция",
        stream: TextIO | None = None,
        width: int = 24,
        min_interval: float = 0.05,
        enabled: bool = True,
    ) -> None:
        self.total = max(int(total), 1)
        self.label = label
        self.stream = stream if stream is not None else sys.stderr
        self.width = max(int(width), 1)
        self.min_interval = max(float(min_interval), 0.0)
        self.enabled = enabled
        self._last_time = 0.0
        self._last_done = -1

    def start(self) -> None:
        self.update(0, force=True)

    def update(self, done: int, *, force: bool = False) -> None:
        if not self.enabled:
            return
        finished = done >= self.total
        now = time.monotonic()
        if not force and not finished:
            if now - self._last_time < self.min_interval:
                return
        if done == self._last_done:
            return
        self._last_time = now
        self._last_done = done

        fraction = min(max(done / self.total, 0.0), 1.0)
        filled = round(fraction * self.width)
        bar = "█" * filled + "░" * (self.width - filled)
        done_text = f"{done:,}".replace(",", " ")
        total_text = f"{self.total:,}".replace(",", " ")
        self.stream.write(
            f"\r{self.label}: [{bar}] {fraction * 100:5.1f}% "
            f"({done_text}/{total_text})"
        )
        self.stream.flush()

    def close(self) -> None:
        """Дорисовывает 100% и завершает строку."""
        if not self.enabled:
            return
        self.update(self.total)
        self.stream.write("\n")
        self.stream.flush()
