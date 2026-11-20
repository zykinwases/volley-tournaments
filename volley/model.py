"""Вероятностная модель силы команд.

Сила команды — это логит вероятности выиграть одну партию у «средней»
команды. Для пары команд i и j:

    P(i выигрывает партию) = sigmoid(r_i - r_j)

Рейтинги подбираются методом максимального правдоподобия по всем
сыгранным партиям (модель Брэдли–Терри) с L2-регуляризацией. Партии
внутри матча считаются независимыми испытаниями Бернулли, поэтому
распределение счёта матча выводится аналитически:

    P(3:0) = p^3
    P(3:1) = 3 p^3 (1-p)
    P(3:2) = 6 p^3 (1-p)^2
    ...

Ковариация оценок берётся как обратная матрица Гессе (приближение
Лапласа), что позволяет учитывать неопределённость рейтингов.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .tournament import Match

# Насколько сильно рейтинги притягиваются к нулю (априорная точность).
DEFAULT_PRIOR = 1.0


def sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _log1pexp(x: float) -> float:
    """log(1 + exp(x)) без переполнения."""
    if x > 0.0:
        return x + math.log1p(math.exp(-x))
    return math.log1p(math.exp(x))


# --------------------------------------------------------------- Линейная алгебра


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float] | None:
    """Решение СЛАУ методом Гаусса с выбором ведущего элемента."""
    n = len(matrix)
    work = [matrix[i][:] + [rhs[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(work[row][col]))
        if abs(work[pivot][col]) < 1e-14:
            return None
        if pivot != col:
            work[col], work[pivot] = work[pivot], work[col]
        pivot_value = work[col][col]
        for row in range(col + 1, n):
            factor = work[row][col] / pivot_value
            if factor:
                for c in range(col, n + 1):
                    work[row][c] -= factor * work[col][c]
    result = [0.0] * n
    for row in range(n - 1, -1, -1):
        total = work[row][n] - sum(
            work[row][col] * result[col] for col in range(row + 1, n)
        )
        if abs(work[row][row]) < 1e-14:
            return None
        result[row] = total / work[row][row]
    return result


def _inverse(matrix: list[list[float]]) -> list[list[float]]:
    """Обратная матрица методом Гаусса–Жордана."""
    n = len(matrix)
    work = [
        matrix[i][:] + [1.0 if i == j else 0.0 for j in range(n)]
        for i in range(n)
    ]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(work[row][col]))
        if abs(work[pivot][col]) < 1e-12:
            work[pivot][col] = 1e-12
        if pivot != col:
            work[col], work[pivot] = work[pivot], work[col]
        pivot_value = work[col][col]
        for c in range(2 * n):
            work[col][c] /= pivot_value
        for row in range(n):
            if row == col:
                continue
            factor = work[row][col]
            if factor:
                for c in range(2 * n):
                    work[row][c] -= factor * work[col][c]
    return [row[n:] for row in work]


def _cholesky(matrix: list[list[float]]) -> list[list[float]]:
    """Разложение Холецкого; некорректные элементы подправляются."""
    n = len(matrix)
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            total = matrix[i][j] - sum(
                lower[i][k] * lower[j][k] for k in range(j)
            )
            if i == j:
                if total <= 0.0:
                    total = 1e-12
                lower[i][j] = math.sqrt(total)
            else:
                lower[i][j] = total / lower[j][j]
    return lower


# ------------------------------------------------------------------ Оценка силы


def _aggregate(
    teams: list[str], matches: list[Match]
) -> tuple[list[list[float]], list[list[float]]]:
    """Возвращает (n, s): число партий и число выигранных партий для i<j."""
    k = len(teams)
    index = {team: i for i, team in enumerate(teams)}
    n = [[0.0] * k for _ in range(k)]
    s = [[0.0] * k for _ in range(k)]
    for match in matches:
        i, j = index[match.a], index[match.b]
        if i > j:
            i, j = j, i
            a_sets, b_sets = match.b_sets, match.a_sets
        else:
            a_sets, b_sets = match.a_sets, match.b_sets
        n[i][j] += a_sets + b_sets
        s[i][j] += a_sets
        s[j][i] += b_sets
    return n, s


def _loglik(
    theta: list[float],
    n: list[list[float]],
    s: list[list[float]],
    prior: float,
) -> float:
    k = len(theta)
    value = -0.5 * prior * sum(t * t for t in theta)
    for i in range(k):
        for j in range(i + 1, k):
            if not n[i][j]:
                continue
            diff = theta[i] - theta[j]
            value -= s[i][j] * _log1pexp(-diff) + s[j][i] * _log1pexp(diff)
    return value


def _grad_hess(
    theta: list[float],
    n: list[list[float]],
    s: list[list[float]],
    prior: float,
) -> tuple[list[float], list[list[float]]]:
    k = len(theta)
    grad = [-prior * t for t in theta]
    hess = [[0.0] * k for _ in range(k)]
    for i in range(k):
        hess[i][i] = prior
    for i in range(k):
        for j in range(i + 1, k):
            if not n[i][j]:
                continue
            p = sigmoid(theta[i] - theta[j])
            weight = n[i][j] * p * (1.0 - p)
            diff = s[i][j] - n[i][j] * p
            grad[i] += diff
            grad[j] -= diff
            hess[i][i] += weight
            hess[j][j] += weight
            hess[i][j] -= weight
            hess[j][i] -= weight
    return grad, hess


@dataclass
class StrengthModel:
    """Рейтинги команд и (опционально) их ковариационная матрица."""

    teams: list[str]
    values: list[float]
    covariance: list[list[float]] | None = None
    prior: float = DEFAULT_PRIOR
    _index: dict[str, int] = field(init=False, repr=False, default_factory=dict)
    _chol: list[list[float]] | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        self._index = {team: i for i, team in enumerate(self.teams)}

    def p_set(self, a: str, b: str) -> float:
        return sigmoid(self.values[self._index[a]] - self.values[self._index[b]])

    def rating(self, team: str) -> float:
        return self.values[self._index[team]]

    def outcome_probs(self, a: str, b: str) -> list[float]:
        """Вероятности исходов (3:0, 3:1, 3:2, 2:3, 1:3, 0:3) для пары a—b."""
        p = self.p_set(a, b)
        q = 1.0 - p
        p3 = p * p * p
        q3 = q * q * q
        return [
            p3,
            3.0 * p3 * q,
            6.0 * p3 * q * q,
            6.0 * p * p * q3,
            3.0 * q3 * p,
            q3,
        ]

    def mean_rating(self) -> float:
        return sum(self.values) / len(self.values) if self.values else 0.0

    def p_set_vs_average(self, team: str) -> float:
        return sigmoid(self.values[self._index[team]] - self.mean_rating())

    def draw(self, rng: random.Random) -> "StrengthModel":
        """Случайная реализация рейтингов из апостериорного распределения."""
        if self.covariance is None:
            return self
        if self._chol is None:
            self._chol = _cholesky(self.covariance)
        count = len(self.values)
        z = [rng.gauss(0.0, 1.0) for _ in range(count)]
        values = [
            self.values[i]
            + sum(self._chol[i][j] * z[j] for j in range(i + 1))
            for i in range(count)
        ]
        return StrengthModel(self.teams, values, prior=self.prior)


def fit_model(
    teams: list[str],
    matches: list[Match],
    prior: float = DEFAULT_PRIOR,
) -> StrengthModel:
    """Оценивает рейтинги по сыгранным матчам методом Ньютона."""
    if prior <= 0.0:
        raise ValueError("prior должен быть больше нуля")
    k = len(teams)
    n, s = _aggregate(teams, matches)
    theta = [0.0] * k
    current = _loglik(theta, n, s, prior)

    for _ in range(100):
        grad, hess = _grad_hess(theta, n, s, prior)
        if max(abs(g) for g in grad) < 1e-10:
            break
        step = _solve(hess, grad)
        if step is None:
            for i in range(k):
                hess[i][i] += 1e-8
            step = _solve(hess, grad)
            if step is None:
                break
        scale = 1.0
        improved = False
        while scale > 1e-6:
            candidate = [theta[i] + scale * step[i] for i in range(k)]
            value = _loglik(candidate, n, s, prior)
            if value > current:
                theta = candidate
                current = value
                improved = True
                break
            scale *= 0.5
        if not improved:
            break

    _, hess = _grad_hess(theta, n, s, prior)
    covariance = _inverse(hess)
    return StrengthModel(teams, theta, covariance=covariance, prior=prior)
