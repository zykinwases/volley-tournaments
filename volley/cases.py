"""Дополнительные разборы места команды при заданных её результатах.

Считаются два сценария:

* худший — команда проигрывает все оставшиеся матчи 0:3;
* лучший — команда выигрывает все оставшиеся матчи 3:0.

Остальные матчи турнира в обоих разборах разыгрываются по модели.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .model import StrengthModel
from .simulate import simulate
from .tournament import OUTCOMES, Tournament

# Разборы считаются на меньшем числе симуляций, чем основной прогноз:
# точности в 0.5% достаточно, а время растёт линейно.
CASE_SIMS_LIMIT = 20_000


@dataclass
class CaseAnalysis:
    """Распределение мест команды при зафиксированных исходах её матчей."""

    wins: int
    total: int
    win_score: str
    loss_score: str
    place_probabilities: list[float]
    expected_place: float

    @property
    def losses(self) -> int:
        return self.total - self.wins


@dataclass
class TeamCases:
    """Результаты двух дополнительных разборов."""

    simulations: int
    worst: CaseAnalysis
    best: CaseAnalysis


def _seed(seed: int | None, offset: int) -> int | None:
    return None if seed is None else seed + offset


def _forced_outcomes(
    remaining: list[tuple[str, str]],
    team: str,
    wins: int,
    win_score: tuple[int, int],
    loss_score: tuple[int, int],
) -> dict[int, int]:
    """Индексы исходов: первые ``wins`` матчей команды — победы, остальные — поражения."""
    forced: dict[int, int] = {}
    position = 0
    for index, (a, b) in enumerate(remaining):
        if a != team and b != team:
            continue
        score = win_score if position < wins else loss_score
        position += 1
        if b == team:
            score = (score[1], score[0])
        forced[index] = OUTCOMES.index(score)
    return forced


def _run_case(
    tournament: Tournament,
    model: StrengthModel,
    remaining: list[tuple[str, str]],
    team: str,
    *,
    wins: int,
    win_score: tuple[int, int],
    loss_score: tuple[int, int],
    sims: int,
    seed: int | None,
    uncertainty: bool,
    label: str,
    progress: Callable[[str, int, int], None] | None,
) -> CaseAnalysis:
    forced = _forced_outcomes(remaining, team, wins, win_score, loss_score)
    callback = None
    if progress is not None:
        progress(label, 0, sims)
        callback = lambda done: progress(label, done, sims)  # noqa: E731
    result = simulate(
        tournament,
        model,
        sims=sims,
        seed=seed,
        uncertainty=uncertainty,
        forced=forced,
        progress=callback,
    )
    forecast = result.forecasts[team]
    return CaseAnalysis(
        wins=wins,
        total=sum(1 for a, b in remaining if team in (a, b)),
        win_score=f"{win_score[0]}:{win_score[1]}",
        loss_score=f"{loss_score[0]}:{loss_score[1]}",
        place_probabilities=list(forecast.place_probs),
        expected_place=forecast.expected_place,
    )


def build_team_cases(
    tournament: Tournament,
    model: StrengthModel,
    *,
    team: str,
    sims: int,
    seed: int | None = None,
    uncertainty: bool = True,
    progress: Callable[[str, int, int], None] | None = None,
) -> TeamCases | None:
    """Считает лучший и худший разборы для ``team``.

    Возвращает ``None``, если у команды не осталось матчей.
    ``progress`` получает ``(название разбора, готово, всего)``.
    """
    remaining = tournament.remaining_pairs()
    total = sum(1 for a, b in remaining if team in (a, b))
    if total == 0:
        return None

    case_sims = max(1, min(sims, CASE_SIMS_LIMIT))

    worst = _run_case(
        tournament,
        model,
        remaining,
        team,
        wins=0,
        win_score=(3, 0),
        loss_score=(0, 3),
        sims=case_sims,
        seed=_seed(seed, 1),
        uncertainty=uncertainty,
        label="Худший случай",
        progress=progress,
    )
    best = _run_case(
        tournament,
        model,
        remaining,
        team,
        wins=total,
        win_score=(3, 0),
        loss_score=(0, 3),
        sims=case_sims,
        seed=_seed(seed, 2),
        uncertainty=uncertainty,
        label="Лучший случай",
        progress=progress,
    )

    return TeamCases(
        simulations=case_sims,
        worst=worst,
        best=best,
    )
