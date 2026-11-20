#!/usr/bin/env python3
"""CLI прогноза волейбольного турнира.

Примеры:

    python main.py predict --data data/example.json --team "Зенит"
    python main.py predict --data data/example.json --team "Зенит" --place 1 --sims 200000
    python main.py predict --data data/example.json --team "Зенит" --markdown report.md
    python main.py standings --data data/example.json
    python main.py add-result --data data/example.json "Зенит" "Динамо" 3:1
    python main.py new --teams "A,B,C" --out data/new.json
    python main.py new --teams "A,B,C" --rounds 2 --out data/new.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from volley.model import DEFAULT_PRIOR, fit_model
from volley.report import format_markdown, format_report, report_to_dict
from volley.simulate import simulate
from volley.text import plural as _plural
from volley.text import rounds_label
from volley.tournament import (
    Tournament,
    TournamentError,
    load_tournament,
    repeated_pair_suffixes,
    save_tournament,
)


def _load(path: str) -> Tournament:
    return load_tournament(path)


def _parse_handicap(
    specs: list[str], teams: list[str]
) -> dict[str, tuple[int, int]]:
    """Разбирает повторяемый флаг --handicap «Команда:победы:очки»."""
    result: dict[str, tuple[int, int]] = {}
    for spec in specs:
        parts = spec.split(":")
        if len(parts) != 3:
            raise TournamentError(
                f"Некорректный гандикап {spec!r}: "
                "ожидается «Команда:победы:очки»"
            )
        team, wins_raw, points_raw = (part.strip() for part in parts)
        if team not in teams:
            raise TournamentError(
                f"Гандикап задан для неизвестной команды: {team!r}"
            )
        try:
            wins = int(wins_raw)
            points = int(points_raw)
        except ValueError:
            raise TournamentError(
                f"Некорректный гандикап {spec!r}: победы и очки "
                "должны быть целыми числами"
            ) from None
        if wins < 0 or points < 0:
            raise TournamentError(
                f"Некорректный гандикап {spec!r}: значения не могут быть "
                "отрицательными"
            )
        if team in result:
            raise TournamentError(
                f"Гандикап для команды {team!r} задан дважды"
            )
        result[team] = (wins, points)
    return result


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("ожидается целое число") from None
    if number < 1:
        raise argparse.ArgumentTypeError("значение должно быть положительным")
    return number


def _positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("ожидается число") from None
    if number <= 0:
        raise argparse.ArgumentTypeError("значение должно быть больше нуля")
    return number


def _place(value: str) -> int:
    number = _positive_int(value)
    if number > 100:
        raise argparse.ArgumentTypeError("слишком большое место")
    return number


def cmd_predict(args: argparse.Namespace) -> int:
    tournament = _load(args.data)
    if args.team is not None and args.team not in tournament.teams:
        available = ", ".join(tournament.teams)
        print(
            f"Ошибка: команда {args.team!r} не найдена.\nДоступны: {available}",
            file=sys.stderr,
        )
        return 2
    if args.place is not None and args.team is None:
        print("Ошибка: --place имеет смысл только вместе с --team", file=sys.stderr)
        return 2
    if args.place is not None and args.place > len(tournament.teams):
        print(
            f"Ошибка: в турнире {len(tournament.teams)} команд, "
            f"места {args.place} не существует",
            file=sys.stderr,
        )
        return 2

    model = fit_model(tournament.teams, tournament.matches, prior=args.prior)
    started = time.perf_counter()
    result = simulate(
        tournament,
        model,
        sims=args.sims,
        seed=args.seed,
        team=args.team,
        uncertainty=not args.no_uncertainty,
    )
    elapsed = time.perf_counter() - started

    report = format_report(
        tournament,
        model,
        result,
        team=args.team,
        target_place=args.place,
        top=args.top,
        uncertainty=not args.no_uncertainty,
        elapsed=elapsed,
    )
    print(report)

    if args.json:
        payload = report_to_dict(
            tournament,
            model,
            result,
            team=args.team,
            target_place=args.place,
            top=args.top,
            uncertainty=not args.no_uncertainty,
        )
        output = Path(args.json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nJSON сохранён: {output}")

    if args.markdown:
        markdown = format_markdown(
            tournament,
            model,
            result,
            team=args.team,
            target_place=args.place,
            top=args.top,
            uncertainty=not args.no_uncertainty,
            elapsed=elapsed,
        )
        output = Path(args.markdown)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown + "\n", encoding="utf-8")
        print(f"\nMarkdown сохранён: {output}")
    return 0


def cmd_standings(args: argparse.Namespace) -> int:
    tournament = _load(args.data)
    print(f"Турнир: {tournament.name}" if tournament.name else "Турнир")
    print(
        f"Сыграно {len(tournament.matches)} из {tournament.total_matches} "
        f"{_plural(tournament.total_matches, 'матч', 'матча', 'матчей')}"
    )
    if tournament.rounds != 1:
        print(f"Формат: {rounds_label(tournament.rounds)}")
    print()
    print(
        f"  {'#':>2}  {'Команда':<22}{'И':>3}{'В':>4}{'О':>4}"
        f"{'Партии':>10}{'Ratio':>8}"
    )
    for position, row in enumerate(tournament.ranking(), 1):
        sets = f"{row.sets_won}:{row.sets_lost}"
        print(
            f"  {position:>2}  {row.team:<22}{row.played:>3}{row.wins:>4}"
            f"{row.points:>4}{sets:>10}{row.set_ratio_text:>8}"
        )

    if tournament.handicap:
        summary = ", ".join(
            f"{team} +{wins}/+{points}"
            for team, (wins, points) in tournament.handicap.items()
        )
        print(f"\nСтартовый гандикап (победы/очки): {summary}")

    remaining = tournament.remaining_pairs()
    if remaining and args.remaining:
        print("\nОставшиеся матчи:")
        suffixes = repeated_pair_suffixes(remaining)
        for (a, b), suffix in zip(remaining, suffixes):
            print(f"  {a} — {b}{suffix}")
    return 0


def cmd_add_result(args: argparse.Namespace) -> int:
    tournament = _load(args.data)
    match = tournament.add_result(args.a, args.b, args.score)
    save_tournament(tournament, args.data)
    print(
        f"Добавлен матч: {match.a} — {match.b} {match.score_text()}. "
        f"Всего матчей: {len(tournament.matches)} из "
        f"{tournament.total_matches}."
    )
    return 0


def cmd_new(args: argparse.Namespace) -> int:
    teams = [part.strip() for part in args.teams.split(",") if part.strip()]
    if len(teams) < 2:
        print("Ошибка: нужно хотя бы две команды через запятую", file=sys.stderr)
        return 2
    handicap = _parse_handicap(args.handicap, teams)
    tournament = Tournament(
        teams=teams, name=args.name, handicap=handicap, rounds=args.rounds
    )
    save_tournament(tournament, args.out)
    print(
        f"Создан турнир «{tournament.name or 'без названия'}»: "
        f"{len(teams)} {_plural(len(teams), 'команда', 'команды', 'команд')}, "
        f"{tournament.total_matches} "
        f"{_plural(tournament.total_matches, 'матч', 'матча', 'матчей')}"
        + (
            f" ({rounds_label(tournament.rounds)})"
            if tournament.rounds != 1
            else ""
        )
        + f". Файл: {args.out}"
    )
    if handicap:
        summary = ", ".join(
            f"{team} +{wins}/+{points}"
            for team, (wins, points) in handicap.items()
        )
        print(f"Стартовый гандикап (победы/очки): {summary}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Прогноз волейбольного турнира по текущим результатам."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser(
        "predict", help="рассчитать вероятности мест и сценарии"
    )
    predict.add_argument(
        "--data", default="data/example.json", help="файл турнира (JSON)"
    )
    predict.add_argument("--team", help="выбранная команда")
    predict.add_argument(
        "--place", type=_place, help="целевое место для сценариев (1-based)"
    )
    predict.add_argument(
        "--sims", type=_positive_int, default=100_000, help="число симуляций"
    )
    predict.add_argument("--seed", type=int, default=2026, help="зерно генератора")
    predict.add_argument(
        "--prior",
        type=_positive_float,
        default=DEFAULT_PRIOR,
        help="сила регуляризации рейтингов (по умолчанию 1.0)",
    )
    predict.add_argument(
        "--no-uncertainty",
        action="store_true",
        help="использовать точечные оценки силы без апостериорной неопределённости",
    )
    predict.add_argument(
        "--top", type=_positive_int, default=5, help="сколько сценариев показывать"
    )
    predict.add_argument("--json", help="сохранить отчёт в JSON")
    predict.add_argument(
        "--markdown",
        "--md",
        dest="markdown",
        metavar="PATH",
        help="сохранить отчёт в Markdown",
    )
    predict.set_defaults(func=cmd_predict)

    standings = subparsers.add_parser(
        "standings", help="показать таблицу и оставшиеся матчи"
    )
    standings.add_argument("--data", default="data/example.json")
    standings.add_argument(
        "--no-remaining", dest="remaining", action="store_false", help="без календаря"
    )
    standings.set_defaults(func=cmd_standings, remaining=True)

    add_result = subparsers.add_parser(
        "add-result", help="добавить результат матча в файл"
    )
    add_result.add_argument("--data", default="data/example.json")
    add_result.add_argument("a", help="первая команда")
    add_result.add_argument("b", help="вторая команда")
    add_result.add_argument("score", help="счёт в партиях, например 3:1")
    add_result.set_defaults(func=cmd_add_result)

    new = subparsers.add_parser("new", help="создать пустой турнир")
    new.add_argument("--teams", required=True, help="команды через запятую")
    new.add_argument("--name", default="", help="название турнира")
    new.add_argument(
        "--rounds",
        type=_positive_int,
        default=1,
        help="число кругов: 1 — каждая пара играет один раз, "
        "2 — дважды (по умолчанию 1)",
    )
    new.add_argument(
        "--handicap",
        action="append",
        default=[],
        metavar="КОМАНДА:ПОБЕДЫ:ОЧКИ",
        help="стартовый гандикап команды, например «Зенит:2:5» "
        "(флаг можно повторять)",
    )
    new.add_argument("--out", default="data/tournament.json", help="файл турнира")
    new.set_defaults(func=cmd_new)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except TournamentError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nПрервано.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
