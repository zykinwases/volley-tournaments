"""Монте-Карло симуляция оставшихся матчей турнира."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

from .model import StrengthModel
from .tournament import Tournament

# (побеждает a?, очки победителя, очки проигравшего, партии победителя, партии проигравшего)
_OUTCOME_INFO: tuple[tuple[bool, int, int, int, int], ...] = (
    (True, 3, 0, 3, 0),
    (True, 3, 0, 3, 1),
    (True, 2, 1, 3, 2),
    (False, 2, 1, 3, 2),
    (False, 3, 0, 3, 1),
    (False, 3, 0, 3, 0),
)


@dataclass
class TeamForecast:
    team: str
    place_probs: list[float]
    expected_place: float
    expected_wins: float
    expected_points: float
    expected_sets_won: float
    expected_sets_lost: float

    @property
    def expected_set_ratio(self) -> float:
        if self.expected_sets_lost == 0.0:
            return float("inf") if self.expected_sets_won else 1.0
        return self.expected_sets_won / self.expected_sets_lost

    def prob_places(self, start: int, end: int) -> float:
        """Суммарная вероятность мест с start по end включительно (1-based)."""
        return sum(self.place_probs[start - 1 : end])


@dataclass
class Scenario:
    """Один точный набор исходов оставшихся матчей."""

    outcomes: tuple[int, ...]
    count: int
    probability: float
    probability_given_place: float


@dataclass
class SimulationResult:
    sims: int
    teams: list[str]
    remaining: list[tuple[str, str]]
    forecasts: dict[str, TeamForecast]
    ranking_counts: Counter
    uncertainty: bool = True
    team: str | None = None
    place_counts: list[int] | None = None
    # [место][матч][исход] — условные вероятности
    conditional_outcomes: list[list[list[float]]] | None = None
    # [место] -> отсортированные сценарии
    scenarios: list[list[Scenario]] | None = None

    def place_probability(self, team: str, place: int) -> float:
        return self.forecasts[team].place_probs[place - 1]


def _build_tables(
    model: StrengthModel,
    teams: list[str],
    index: dict[str, int],
    remaining: list[tuple[str, str]],
) -> list[tuple[list[float], list[tuple[int, int, int, int, int, int]]]]:
    """Для каждого матча: накопленные вероятности и дельты таблицы."""
    tables = []
    for a, b in remaining:
        ia, ib = index[a], index[b]
        probabilities = model.outcome_probs(a, b)
        cumulative: list[float] = []
        total = 0.0
        for probability in probabilities:
            total += probability
            cumulative.append(total)
        cumulative[-1] = 1.0  # страховка от погрешности округления

        deltas = []
        for a_wins, win_points, lose_points, win_sets, lose_sets in _OUTCOME_INFO:
            if a_wins:
                deltas.append(
                    (ia, ib, win_points, lose_points, win_sets, lose_sets)
                )
            else:
                deltas.append(
                    (ib, ia, win_points, lose_points, win_sets, lose_sets)
                )
        tables.append((cumulative, deltas))
    return tables


def simulate(
    tournament: Tournament,
    model: StrengthModel,
    *,
    sims: int = 100_000,
    seed: int | None = None,
    team: str | None = None,
    uncertainty: bool = True,
    rating_draws: int = 64,
    scenarios_per_place: int = 25,
) -> SimulationResult:
    """Разыгрывает оставшиеся матчи ``sims`` раз.

    При ``uncertainty=True`` рейтинги на каждую серию испытаний берутся
    случайно из апостериорного распределения, что учитывает неполноту
    турнирных данных. При ``uncertainty=False`` используются точечные оценки.
    """
    if sims < 1:
        raise ValueError("Число симуляций должно быть положительным")

    teams = tournament.teams
    count = len(teams)
    index = {name: i for i, name in enumerate(teams)}
    if team is not None and team not in index:
        raise ValueError(f"Неизвестная команда: {team!r}")
    team_index = index[team] if team is not None else None

    remaining = tournament.remaining_pairs()
    standings = tournament.standings()
    base_wins = [standings[t].wins for t in teams]
    base_points = [standings[t].points for t in teams]
    base_sets_won = [standings[t].sets_won for t in teams]
    base_sets_lost = [standings[t].sets_lost for t in teams]

    place_counts = [[0] * count for _ in range(count)]
    sum_wins = [0] * count
    sum_points = [0] * count
    sum_sets_won = [0] * count
    sum_sets_lost = [0] * count
    ranking_counts: Counter = Counter()

    if team_index is not None:
        place_totals = [0] * count
        conditional_counts = [
            [[0] * 6 for _ in remaining] for _ in range(count)
        ]
        scenario_counts: list[Counter] = [Counter() for _ in range(count)]
    else:
        place_totals = None
        conditional_counts = None
        scenario_counts = None

    rng = random.Random(seed)
    draws = 1 if not uncertainty else max(1, min(rating_draws, sims))
    base = sims // draws
    extra = sims % draws

    for draw_index in range(draws):
        batch = base + (1 if draw_index < extra else 0)
        if batch <= 0:
            continue
        current_model = model.draw(rng) if uncertainty else model
        tables = _build_tables(current_model, teams, index, remaining)

        for _ in range(batch):
            wins = base_wins[:]
            points = base_points[:]
            sets_won = base_sets_won[:]
            sets_lost = base_sets_lost[:]

            outcomes: list[int] = []
            for cumulative, deltas in tables:
                value = rng.random()
                outcome = 0
                while value > cumulative[outcome]:
                    outcome += 1
                outcomes.append(outcome)

                winner, loser, win_points, lose_points, win_sets, lose_sets = (
                    deltas[outcome]
                )
                wins[winner] += 1
                points[winner] += win_points
                points[loser] += lose_points
                sets_won[winner] += win_sets
                sets_lost[winner] += lose_sets
                sets_won[loser] += lose_sets
                sets_lost[loser] += win_sets

            order = sorted(
                range(count),
                key=lambda i: (
                    -wins[i],
                    -points[i],
                    -(sets_won[i] / sets_lost[i] if sets_lost[i] else 1e9),
                    -sets_won[i],
                    teams[i],
                ),
            )

            for position, team_id in enumerate(order):
                place_counts[team_id][position] += 1
                sum_wins[team_id] += wins[team_id]
                sum_points[team_id] += points[team_id]
                sum_sets_won[team_id] += sets_won[team_id]
                sum_sets_lost[team_id] += sets_lost[team_id]

            ranking_counts[tuple(order)] += 1

            if team_index is not None:
                position = 0
                for pos, team_id in enumerate(order):
                    if team_id == team_index:
                        position = pos
                        break
                place_totals[position] += 1
                for match_index, outcome in enumerate(outcomes):
                    conditional_counts[position][match_index][outcome] += 1
                scenario_counts[position][tuple(outcomes)] += 1

    forecasts: dict[str, TeamForecast] = {}
    for team_id, name in enumerate(teams):
        forecasts[name] = TeamForecast(
            team=name,
            place_probs=[c / sims for c in place_counts[team_id]],
            expected_place=sum(
                (position + 1) * c
                for position, c in enumerate(place_counts[team_id])
            )
            / sims,
            expected_wins=sum_wins[team_id] / sims,
            expected_points=sum_points[team_id] / sims,
            expected_sets_won=sum_sets_won[team_id] / sims,
            expected_sets_lost=sum_sets_lost[team_id] / sims,
        )

    conditional_outcomes = None
    scenarios = None
    if team_index is not None:
        conditional_outcomes = [
            [
                [
                    (c / place_totals[position] if place_totals[position] else 0.0)
                    for c in conditional_counts[position][match_index]
                ]
                for match_index in range(len(remaining))
            ]
            for position in range(count)
        ]
        scenarios = []
        for position in range(count):
            total = place_totals[position]
            scenarios.append(
                [
                    Scenario(
                        outcomes=key,
                        count=value,
                        probability=value / sims,
                        probability_given_place=(value / total if total else 0.0),
                    )
                    for key, value in scenario_counts[position].most_common(
                        scenarios_per_place
                    )
                ]
            )

    return SimulationResult(
        sims=sims,
        teams=list(teams),
        remaining=remaining,
        forecasts=forecasts,
        ranking_counts=ranking_counts,
        uncertainty=uncertainty,
        team=team,
        place_counts=place_totals,
        conditional_outcomes=conditional_outcomes,
        scenarios=scenarios,
    )
