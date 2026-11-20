"""Вспомогательные функции для русскоязычного текста."""

from __future__ import annotations


def plural(number: int, one: str, few: str, many: str) -> str:
    """Выбирает форму слова по числу: 1 матч, 2 матча, 5 матчей."""
    if number % 10 == 1 and number % 100 != 11:
        return one
    if number % 10 in (2, 3, 4) and number % 100 not in (12, 13, 14):
        return few
    return many


_ROUND_WORDS = {
    1: "один круг",
    2: "два круга",
    3: "три круга",
    4: "четыре круга",
    5: "пять кругов",
}


def rounds_label(rounds: int) -> str:
    """Человекочитаемое название формата: «два круга», «5 кругов»."""
    return _ROUND_WORDS.get(
        rounds, f"{rounds} {plural(rounds, 'круг', 'круга', 'кругов')}"
    )
