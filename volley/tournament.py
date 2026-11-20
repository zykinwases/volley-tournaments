"""Данные турнира: команды, матчи, таблица и календарь.

Формат файла (JSON):

    {
      "name": "Летний кубок",
      "teams": ["Команда A", "Команда B", ...],
      "handicap": {
        "Команда A": {"wins": 2, "points": 5}
      },
      "matches": [
        {"a": "Команда A", "b": "Команда B", "score": "3:1"},
        ...
      ]
    }

В файле хранятся только сыгранные матчи. Несыгранные пары вычисляются
из списка команд (круг в один круг). Поле ``handicap`` необязательно:
стартовые победы и очки прибавляются к набранным в турнире, но не
учитываются как сыгранные матчи и не влияют на соотношение партий.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# Возможные исходы матча в порядке (3:0, 3:1, 3:2, 2:3, 1:3, 0:3)
# для пары (a, b): первые три — победы a, последние три — победы b.
OUTCOMES: tuple[tuple[int, int], ...] = (
    (3, 0),
    (3, 1),
    (3, 2),
    (2, 3),
    (1, 3),
    (0, 3),
)

VALID_SCORES = frozenset(OUTCOMES)


class TournamentError(ValueError):
    """Ошибка в данных турнира."""


def points_for(a_sets: int, b_sets: int) -> tuple[int, int]:
    """Очки по партиям: 3:0/3:1 — 3:0, 3:2 — 2:1."""
    if a_sets > b_sets:
        return (3 if b_sets <= 1 else 2, 1 if b_sets == 2 else 0)
    return (1 if a_sets == 2 else 0, 3 if a_sets <= 1 else 2)


def parse_score(raw: object) -> tuple[int, int]:
    """Разбирает счёт вида ``"3:1"`` или ``[3, 1]``."""
    if isinstance(raw, str):
        text = raw.strip().replace("–", "-").replace("—", "-").replace(" ", "")
        if ":" not in text:
            raise TournamentError(
                f"Не удалось разобрать счёт {raw!r}: ожидается формат «3:1»"
            )
        left, right = text.split(":", 1)
        try:
            a_sets, b_sets = int(left), int(right)
        except ValueError:
            raise TournamentError(
                f"Не удалось разобрать счёт {raw!r}: ожидается формат «3:1»"
            ) from None
    elif isinstance(raw, (list, tuple)) and len(raw) == 2:
        try:
            a_sets, b_sets = int(raw[0]), int(raw[1])
        except (TypeError, ValueError):
            raise TournamentError(f"Не удалось разобрать счёт {raw!r}") from None
    else:
        raise TournamentError(f"Не удалось разобрать счёт {raw!r}")

    if (a_sets, b_sets) not in VALID_SCORES:
        raise TournamentError(
            f"Некорректный счёт {a_sets}:{b_sets}: "
            "допустимы 3:0, 3:1, 3:2, 2:3, 1:3 и 0:3"
        )
    return a_sets, b_sets


@dataclass(frozen=True)
class Match:
    """Сыгранный матч; счёт указан в партиях."""

    a: str
    b: str
    a_sets: int
    b_sets: int

    def __post_init__(self) -> None:
        if self.a == self.b:
            raise TournamentError(f"Команда не может играть сама с собой: {self.a!r}")
        if (self.a_sets, self.b_sets) not in VALID_SCORES:
            raise TournamentError(
                f"Некорректный счёт {self.a_sets}:{self.b_sets} "
                f"в матче {self.a!r} — {self.b!r}"
            )

    @property
    def total_sets(self) -> int:
        return self.a_sets + self.b_sets

    @property
    def winner(self) -> str:
        return self.a if self.a_sets > self.b_sets else self.b

    @property
    def loser(self) -> str:
        return self.b if self.a_sets > self.b_sets else self.a

    def points(self) -> tuple[int, int]:
        return points_for(self.a_sets, self.b_sets)

    def score_text(self) -> str:
        return f"{self.a_sets}:{self.b_sets}"


@dataclass
class Standing:
    """Строка турнирной таблицы."""

    team: str
    played: int = 0
    wins: int = 0
    points: int = 0
    sets_won: int = 0
    sets_lost: int = 0

    @property
    def set_ratio(self) -> float:
        if self.sets_lost == 0:
            return float("inf") if self.sets_won else 1.0
        return self.sets_won / self.sets_lost

    @property
    def set_ratio_text(self) -> str:
        if self.sets_lost == 0:
            return "∞" if self.sets_won else "—"
        return f"{self.sets_won / self.sets_lost:.2f}"


def standing_key(standing: Standing) -> tuple:
    """Ключ сортировки: победы, очки, соотношение партий, затем имя.

    Имя используется только как детерминированный финальный тай-брейк,
    потому что регламент не задаёт критерий дальше соотношения партий.
    """
    return (
        -standing.wins,
        -standing.points,
        -standing.set_ratio,
        -standing.sets_won,
        standing.team,
    )


def _normalize_handicap(
    raw: object, team_set: set[str]
) -> dict[str, tuple[int, int]]:
    """Проверяет и нормализует стартовый гандикап.

    Допустимы значения ``{"wins": 2, "points": 5}`` и ``[2, 5]``.
    Нулевые гандикапы отбрасываются.
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TournamentError(
            "Поле «handicap» должно быть объектом вида "
            '{"Команда": {"wins": 2, "points": 5}}'
        )
    result: dict[str, tuple[int, int]] = {}
    for team, value in raw.items():
        if team not in team_set:
            raise TournamentError(f"Гандикап задан для неизвестной команды: {team!r}")
        if isinstance(value, dict):
            unknown = set(value) - {"wins", "points"}
            if unknown:
                raise TournamentError(
                    f"Гандикап для {team!r}: неизвестные поля {sorted(unknown)}"
                )
            wins = value.get("wins", 0)
            points = value.get("points", 0)
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            wins, points = value
        else:
            raise TournamentError(
                f"Гандикап для {team!r}: ожидается "
                '{"wins": ..., "points": ...} или [победы, очки]'
            )
        for label, number in (("побед", wins), ("очков", points)):
            if isinstance(number, bool) or not isinstance(number, int):
                raise TournamentError(
                    f"Гандикап для {team!r}: количество {label} "
                    "должно быть целым числом"
                )
            if number < 0:
                raise TournamentError(
                    f"Гандикап для {team!r}: количество {label} "
                    "не может быть отрицательным"
                )
        if wins or points:
            result[team] = (wins, points)
    return result


@dataclass
class Tournament:
    """Турнир: список команд и сыгранные матчи."""

    teams: list[str]
    matches: list[Match] = field(default_factory=list)
    name: str = ""
    # Стартовый гандикап: команда -> (победы, очки). Прибавляется к итоговым
    # победам и очкам, но не считается сыгранными матчами и не влияет на партии.
    handicap: dict[str, tuple[int, int]] = field(default_factory=dict)
    _team_set: set[str] = field(init=False, repr=False, default_factory=set)

    def __post_init__(self) -> None:
        self.teams = list(self.teams)
        if len(self.teams) < 2:
            raise TournamentError("Нужно хотя бы две команды")
        if len(set(self.teams)) != len(self.teams):
            raise TournamentError("Названия команд должны быть уникальными")
        if any(not isinstance(t, str) or not t.strip() for t in self.teams):
            raise TournamentError("Название команды не может быть пустым")

        self._team_set = set(self.teams)
        self.handicap = _normalize_handicap(self.handicap, self._team_set)
        max_matches = len(self.teams) * (len(self.teams) - 1) // 2
        if len(self.matches) > max_matches:
            raise TournamentError(
                f"Матчей больше, чем возможно в один круг: "
                f"{len(self.matches)} > {max_matches}"
            )

        seen: set[frozenset[str]] = set()
        for match in self.matches:
            if match.a not in self._team_set:
                raise TournamentError(f"Неизвестная команда: {match.a!r}")
            if match.b not in self._team_set:
                raise TournamentError(f"Неизвестная команда: {match.b!r}")
            pair = frozenset((match.a, match.b))
            if pair in seen:
                raise TournamentError(
                    f"Пара {match.a!r} — {match.b!r} уже встречалась: "
                    "в один круг допускается только один матч"
                )
            seen.add(pair)

    # ------------------------------------------------------------------ Table

    @property
    def total_matches(self) -> int:
        return len(self.teams) * (len(self.teams) - 1) // 2

    def initial_wins(self, team: str) -> int:
        return self.handicap.get(team, (0, 0))[0]

    def initial_points(self, team: str) -> int:
        return self.handicap.get(team, (0, 0))[1]

    def standings(self) -> dict[str, Standing]:
        table = {
            team: Standing(
                team,
                wins=self.initial_wins(team),
                points=self.initial_points(team),
            )
            for team in self.teams
        }
        for match in self.matches:
            a, b = table[match.a], table[match.b]
            a.played += 1
            b.played += 1
            a.sets_won += match.a_sets
            a.sets_lost += match.b_sets
            b.sets_won += match.b_sets
            b.sets_lost += match.a_sets
            a_points, b_points = match.points()
            a.points += a_points
            b.points += b_points
            if match.a_sets > match.b_sets:
                a.wins += 1
            else:
                b.wins += 1
        return table

    def ranking(self) -> list[Standing]:
        return sorted(self.standings().values(), key=standing_key)

    def remaining_pairs(self) -> list[tuple[str, str]]:
        played = {frozenset((m.a, m.b)) for m in self.matches}
        pairs: list[tuple[str, str]] = []
        for i, a in enumerate(self.teams):
            for b in self.teams[i + 1 :]:
                if frozenset((a, b)) not in played:
                    pairs.append((a, b))
        return pairs

    # ----------------------------------------------------------------- Editing

    def add_result(self, a: str, b: str, score: object) -> Match:
        if a not in self._team_set:
            raise TournamentError(f"Неизвестная команда: {a!r}")
        if b not in self._team_set:
            raise TournamentError(f"Неизвестная команда: {b!r}")
        if a == b:
            raise TournamentError("Команда не может играть сама с собой")
        pair = frozenset((a, b))
        for match in self.matches:
            if frozenset((match.a, match.b)) == pair:
                raise TournamentError(
                    f"Матч {a!r} — {b!r} уже сыгран ({match.score_text()})"
                )
        a_sets, b_sets = parse_score(score)
        match = Match(a, b, a_sets, b_sets)
        self.matches.append(match)
        return match


# --------------------------------------------------------------------- IO


def load_tournament(path: str | Path) -> Tournament:
    file_path = Path(path)
    try:
        text = file_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise TournamentError(f"Файл не найден: {file_path}") from None
    except OSError as exc:
        raise TournamentError(f"Не удалось прочитать {file_path}: {exc}") from exc

    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TournamentError(f"Некорректный JSON в {file_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise TournamentError("Корень JSON-файла должен быть объектом")

    teams = raw.get("teams")
    if not isinstance(teams, list):
        raise TournamentError("Поле «teams» должно быть списком названий команд")

    raw_matches = raw.get("matches", [])
    if not isinstance(raw_matches, list):
        raise TournamentError("Поле «matches» должно быть списком")

    matches: list[Match] = []
    for i, item in enumerate(raw_matches, 1):
        if not isinstance(item, dict):
            raise TournamentError(f"Матч №{i}: ожидается объект")
        for key in ("a", "b", "score"):
            if key not in item:
                raise TournamentError(f"Матч №{i}: отсутствует поле «{key}»")
        try:
            matches.append(Match(item["a"], item["b"], *parse_score(item["score"])))
        except TournamentError as exc:
            raise TournamentError(f"Матч №{i}: {exc}") from None

    name = raw.get("name", "")
    if not isinstance(name, str):
        raise TournamentError("Поле «name» должно быть строкой")

    handicap = raw.get("handicap", {})
    if handicap is None:
        handicap = {}
    if not isinstance(handicap, dict):
        raise TournamentError(
            "Поле «handicap» должно быть объектом вида "
            '{"Команда": {"wins": 2, "points": 5}}'
        )
    return Tournament(
        teams=teams, matches=matches, name=name, handicap=handicap
    )


def save_tournament(tournament: Tournament, path: str | Path) -> None:
    file_path = Path(path)
    data = {
        "name": tournament.name,
        "teams": tournament.teams,
    }
    if tournament.handicap:
        data["handicap"] = {
            team: {"wins": wins, "points": points}
            for team, (wins, points) in tournament.handicap.items()
        }
    data["matches"] = [
        {"a": m.a, "b": m.b, "score": m.score_text()}
        for m in tournament.matches
    ]
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
