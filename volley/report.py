"""Текстовый отчёт и экспорт прогноза в JSON."""

from __future__ import annotations

import math
import textwrap

from .model import StrengthModel
from .simulate import SimulationResult
from .tournament import OUTCOMES, Tournament

LINE_WIDTH = 100


def _pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}"


def _plural(number: int, one: str, few: str, many: str) -> str:
    if number % 10 == 1 and number % 100 != 11:
        return one
    if number % 10 in (2, 3, 4) and number % 100 not in (12, 13, 14):
        return few
    return many


def _ratio(sets_won: float, sets_lost: float) -> str:
    if sets_lost == 0:
        return "∞" if sets_won else "—"
    return f"{sets_won / sets_lost:.2f}"


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _format_handicap(tournament: Tournament) -> list[str]:
    parts = []
    for team in tournament.teams:
        wins, points = tournament.handicap.get(team, (0, 0))
        if wins or points:
            parts.append(
                f"{team}: +{wins} "
                f"{_plural(wins, 'победа', 'победы', 'побед')}, +{points} "
                f"{_plural(points, 'очко', 'очка', 'очков')}"
            )
    if not parts:
        return []
    lines = ["Стартовый гандикап (уже учтён в таблице)"]
    lines.extend(
        textwrap.wrap(
            "; ".join(parts),
            width=LINE_WIDTH,
            initial_indent="  ",
            subsequent_indent="  ",
        )
    )
    return lines


def _format_standings(tournament: Tournament) -> list[str]:
    lines = ["Текущая таблица"]
    lines.append(
        f"  {'#':>2}  {'Команда':<22}{'И':>3}{'В':>4}{'О':>4}"
        f"{'Партии':>10}{'Ratio':>8}"
    )
    for position, row in enumerate(tournament.ranking(), 1):
        sets = f"{row.sets_won}:{row.sets_lost}"
        lines.append(
            f"  {position:>2}  {row.team:<22}{row.played:>3}{row.wins:>4}"
            f"{row.points:>4}{sets:>10}{row.set_ratio_text:>8}"
        )
    return lines


def _format_ratings(model: StrengthModel) -> list[str]:
    lines = ["Оценка силы команд по сыгранным партиям"]
    lines.append(
        f"  {'Команда':<22}{'Рейтинг':>9}{'Партия vs ср.':>16}"
    )
    for team in sorted(
        model.teams, key=model.rating, reverse=True
    ):
        rating = model.rating(team)
        probability = model.p_set_vs_average(team)
        lines.append(f"  {team:<22}{rating:>+9.2f}{_pct(probability):>15}%")
    lines.extend(
        textwrap.wrap(
            "Рейтинг — логит вероятности выиграть партию у средней команды; "
            "0 — средний уровень.",
            width=LINE_WIDTH,
            initial_indent="  ",
            subsequent_indent="  ",
        )
    )
    return lines


def _format_forecast_table(result: SimulationResult) -> list[str]:
    count = len(result.teams)
    lines = ["Вероятности итоговых мест, %"]
    header = f"  {'Команда':<22}" + "".join(f"{place:>6}" for place in range(1, count + 1))
    lines.append(header)
    ordered = sorted(
        result.teams, key=lambda name: result.forecasts[name].expected_place
    )
    for team in ordered:
        probabilities = result.forecasts[team].place_probs
        lines.append(
            f"  {team:<22}" + "".join(f"{_pct(p):>6}" for p in probabilities)
        )
    return lines


def _format_expected(result: SimulationResult) -> list[str]:
    lines = ["Ожидаемые итоговые показатели"]
    lines.append(
        f"  {'Команда':<22}{'Место':>8}{'Победы':>8}{'Очки':>7}"
        f"{'Партии':>13}{'Ratio':>8}"
    )
    ordered = sorted(
        result.teams, key=lambda name: result.forecasts[name].expected_place
    )
    for team in ordered:
        forecast = result.forecasts[team]
        sets = f"{forecast.expected_sets_won:.1f}:{forecast.expected_sets_lost:.1f}"
        lines.append(
            f"  {team:<22}{forecast.expected_place:>8.2f}"
            f"{forecast.expected_wins:>8.1f}{forecast.expected_points:>7.1f}"
            f"{sets:>13}{_ratio(forecast.expected_sets_won, forecast.expected_sets_lost):>8}"
        )
    return lines


def _format_rankings(
    result: SimulationResult, top: int
) -> list[str]:
    lines = ["Наиболее вероятные итоговые расстановки"]
    for number, (order, count) in enumerate(
        result.ranking_counts.most_common(top), 1
    ):
        probability = count / result.sims
        lines.append(f"  {number}) {_pct(probability):>5}%")
        text = ", ".join(result.teams[i] for i in order)
        lines.extend(
            textwrap.wrap(
                text,
                width=LINE_WIDTH,
                initial_indent="     ",
                subsequent_indent="     ",
            )
        )
    if len(result.teams) >= 5:
        lines.extend(
            textwrap.wrap(
                "Конкретных полных расстановок очень много, поэтому вероятность "
                "каждой из них невелика — надёжнее смотреть на вероятности мест выше.",
                width=LINE_WIDTH,
                initial_indent="  ",
                subsequent_indent="  ",
            )
        )
    return lines


def _oriented_match(
    result: SimulationResult,
    match_index: int,
    probabilities: list[float],
    team: str | None,
) -> tuple[str, str, list[float]]:
    """Поворачивает матч так, чтобы выбранная команда была первой."""
    a, b = result.remaining[match_index]
    if team is not None and b == team:
        return b, a, list(reversed(probabilities))
    return a, b, probabilities


def _format_team_section(
    tournament: Tournament,
    result: SimulationResult,
    team: str,
    target_place: int | None,
    top: int,
) -> list[str]:
    forecast = result.forecasts[team]
    standings = tournament.standings()[team]
    ranking = tournament.ranking()
    current_place = next(
        position for position, row in enumerate(ranking, 1) if row.team == team
    )

    lines = ["", f"Команда «{team}»"]
    lines.append(
        f"  Сейчас: {current_place}-е место, {standings.wins} "
        f"{_plural(standings.wins, 'победа', 'победы', 'побед')}, "
        f"{standings.points} "
        f"{_plural(standings.points, 'очко', 'очка', 'очков')}, "
        f"партии {standings.sets_won}:{standings.sets_lost}"
    )
    team_remaining = sum(
        1 for a, b in result.remaining if team in (a, b)
    )
    lines.append(
        f"  Осталось матчей у команды: {team_remaining}; "
        f"ожидание: {forecast.expected_wins:.2f} побед, "
        f"{forecast.expected_points:.2f} очков, "
        f"партии {forecast.expected_sets_won:.1f}:{forecast.expected_sets_lost:.1f}"
    )

    lines.append("")
    lines.append("  Вероятности мест:")
    best = max(forecast.place_probs) if forecast.place_probs else 0.0
    for place, probability in enumerate(forecast.place_probs, 1):
        bar_length = round(probability / best * 24) if best else 0
        lines.append(
            f"    {place:>2}-е  {_pct(probability):>6}%  {'█' * bar_length}"
        )
    place_count = len(forecast.place_probs)
    top3 = min(3, place_count)
    top5 = min(5, place_count)
    summary = f"  P(1–{top3}) = {_pct(forecast.prob_places(1, top3))}%"
    if top5 != top3:
        summary += f", P(1–{top5}) = {_pct(forecast.prob_places(1, top5))}%"
    lines.append(summary)

    if target_place is None:
        target_place = max(
            range(1, len(forecast.place_probs) + 1),
            key=lambda place: forecast.place_probs[place - 1],
        )
    place_probability = forecast.place_probs[target_place - 1]
    observations = (
        result.place_counts[target_place - 1] if result.place_counts else 0
    )

    lines.append("")
    lines.append(
        f"  Сценарии для {target_place}-го места: вероятность "
        f"{_pct(place_probability)}% ({observations} из {result.sims} симуляций)"
    )

    if not result.remaining:
        lines.append("  Все матчи уже сыграны — оставшихся игр нет.")
        return lines
    if observations == 0:
        lines.append(
            "  Это место не встретилось ни в одной симуляции. "
            "Увеличьте --sims или проверьте данные."
        )
        return lines

    lines.append("")
    lines.append(
        "  Условные вероятности исходов оставшихся матчей, % "
        "(счёт — для первой команды пары):"
    )
    lines.append(
        f"  {'Матч':<30}{'3:0':>7}{'3:1':>7}{'3:2':>7}"
        f"{'2:3':>7}{'1:3':>7}{'0:3':>7}{'Мода':>9}"
    )
    assert result.conditional_outcomes is not None
    for match_index in range(len(result.remaining)):
        probabilities = result.conditional_outcomes[target_place - 1][match_index]
        a, b, oriented = _oriented_match(
            result, match_index, probabilities, team
        )
        label = f"{a} — {b}"
        if len(label) > 29:
            label = label[:28] + "…"
        best = max(oriented)
        modes = [
            f"{OUTCOMES[index][0]}:{OUTCOMES[index][1]}"
            for index, probability in enumerate(oriented)
            if best - probability < 1e-9
        ]
        lines.append(
            f"  {label:<30}"
            + "".join(f"{_pct(p):>7}" for p in oriented)
            + f"{'/'.join(modes):>9}"
        )
    lines.extend(
        textwrap.wrap(
            "Мода — самый вероятный счёт отдельного матча при условии места; "
            "у совместного сценария всех матчей вероятность всегда ниже.",
            width=LINE_WIDTH,
            initial_indent="  ",
            subsequent_indent="  ",
        )
    )

    scenarios = (
        result.scenarios[target_place - 1][:top] if result.scenarios else []
    )
    if scenarios and len(result.remaining) <= 6:
        lines.append("")
        lines.append("  Наиболее вероятные точные наборы исходов:")
        for number, scenario in enumerate(scenarios, 1):
            parts = []
            for match_index, outcome in enumerate(scenario.outcomes):
                a, b = result.remaining[match_index]
                a_sets, b_sets = OUTCOMES[outcome]
                if team == b:
                    a, b = b, a
                    a_sets, b_sets = b_sets, a_sets
                parts.append(f"{a}—{b} {a_sets}:{b_sets}")
            lines.append(
                f"  {number}) {_pct(scenario.probability_given_place, 2)}% "
                f"при условии места "
                f"({_pct(scenario.probability, 3)}% безусловно)"
            )
            lines.extend(
                textwrap.wrap(
                    "; ".join(parts),
                    width=LINE_WIDTH,
                    initial_indent="     ",
                    subsequent_indent="     ",
                )
            )

    return lines


def format_report(
    tournament: Tournament,
    model: StrengthModel,
    result: SimulationResult,
    *,
    team: str | None = None,
    target_place: int | None = None,
    top: int = 5,
    uncertainty: bool | None = None,
    elapsed: float | None = None,
) -> str:
    played = len(tournament.matches)
    total = tournament.total_matches
    if uncertainty is None:
        uncertainty = result.uncertainty

    lines = ["ПРОГНОЗ ТУРНИРА", "=" * 15]
    if tournament.name:
        lines.append(f"Турнир: {tournament.name}")
    lines.append(
        f"Сыграно матчей: {played} из {total} (осталось {total - played})"
    )
    mode = (
        "с учётом неопределённости оценок"
        if uncertainty
        else "по точечным оценкам силы"
    )
    info = (
        f"Симуляций: {result.sims:,}".replace(",", " ")
        + f"; модель Брэдли–Терри по партиям, prior={model.prior:.2f}, {mode}."
    )
    lines.extend(textwrap.wrap(info, width=LINE_WIDTH))
    if elapsed is not None:
        lines.append(f"Время расчёта: {elapsed:.2f} с")
    lines.append("")

    lines.extend(_format_standings(tournament))
    handicap_lines = _format_handicap(tournament)
    if handicap_lines:
        lines.append("")
        lines.extend(handicap_lines)
    lines.append("")
    lines.extend(_format_ratings(model))
    lines.append("")
    lines.extend(_format_forecast_table(result))
    lines.append("")
    lines.extend(_format_expected(result))
    lines.append("")
    lines.extend(_format_rankings(result, top))

    if team is not None:
        lines.extend(
            _format_team_section(tournament, result, team, target_place, top)
        )
    else:
        lines.append("")
        lines.append(
            "Чтобы получить вероятности мест и сценарии для команды, "
            "запустите с параметром --team «Название»."
        )

    return "\n".join(lines)


_MD_ORDER = [f"{a}:{b}" for a, b in OUTCOMES]


def _md_escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _md_table(headers: list[str], rows: list[list[object]], align: list[str]) -> list[str]:
    marks = {"left": ":---", "right": "---:", "center": ":---:"}
    lines = [
        "| " + " | ".join(_md_escape(h) for h in headers) + " |",
        "| " + " | ".join(marks[a] for a in align) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_md_escape(c) for c in row) + " |")
    return lines


def format_markdown(
    tournament: Tournament,
    model: StrengthModel,
    result: SimulationResult,
    *,
    team: str | None = None,
    target_place: int | None = None,
    top: int = 5,
    uncertainty: bool | None = None,
    elapsed: float | None = None,
) -> str:
    """Отчёт в Markdown: заголовки, таблицы и списки."""
    data = report_to_dict(
        tournament,
        model,
        result,
        team=team,
        target_place=target_place,
        top=top,
        uncertainty=uncertainty,
    )

    lines: list[str] = ["# Прогноз турнира", ""]
    if data["tournament"]:
        lines.append(f"- **Турнир:** {_md_escape(data['tournament'])}")
    lines.append(
        f"- **Сыграно матчей:** {data['played_matches']} из "
        f"{data['total_matches']} "
        f"(осталось {data['total_matches'] - data['played_matches']})"
    )
    mode = (
        "с учётом неопределённости оценок"
        if data["model"]["rating_uncertainty"]
        else "по точечным оценкам силы"
    )
    simulations = f"{data['simulations']:,}".replace(",", " ")
    lines.append(
        f"- **Симуляций:** {simulations}; модель Брэдли–Терри по партиям, "
        f"prior={data['model']['prior']:.2f}, {mode}"
    )
    if elapsed is not None:
        lines.append(f"- **Время расчёта:** {elapsed:.2f} с")

    # ---------------------------------------------------------- Таблица
    lines.extend(["", "## Текущая таблица", ""])
    rows = []
    for row in data["standings"]:
        if row["sets_lost"] == 0:
            ratio = "∞" if row["sets_won"] else "—"
        else:
            ratio = f"{row['set_ratio']:.2f}"
        rows.append(
            [
                row["place"],
                row["team"],
                row["played"],
                row["wins"],
                row["points"],
                f"{row['sets_won']}:{row['sets_lost']}",
                ratio,
            ]
        )
    lines.extend(
        _md_table(
            ["", "Команда", "И", "В", "О", "Партии", "Ratio"],
            rows,
            ["right", "left", "right", "right", "right", "right", "right"],
        )
    )

    # ---------------------------------------------------------- Гандикап
    if data["handicap"]:
        lines.extend(
            [
                "",
                "## Стартовый гандикап",
                "",
                "Уже учтён в таблице; на число сыгранных матчей и партии "
                "не влияет.",
                "",
            ]
        )
        rows = [
            [team, f"+{values['wins']}", f"+{values['points']}"]
            for team, values in data["handicap"].items()
        ]
        lines.extend(
            _md_table(
                ["Команда", "Победы", "Очки"],
                rows,
                ["left", "right", "right"],
            )
        )

    # ---------------------------------------------------------- Рейтинги
    lines.extend(["", "## Оценка силы команд", ""])
    ratings = sorted(
        data["ratings"], key=lambda item: item["rating"], reverse=True
    )
    rows = [
        [
            item["team"],
            f"{item['rating']:+.2f}",
            f"{item['set_probability_vs_average'] * 100:.1f}%",
        ]
        for item in ratings
    ]
    lines.extend(
        _md_table(
            ["Команда", "Рейтинг", "Партия vs ср."],
            rows,
            ["left", "right", "right"],
        )
    )

    # ---------------------------------------------------------- Прогноз
    forecasts = sorted(
        data["forecasts"], key=lambda item: item["expected_place"]
    )
    count = len(forecasts)
    lines.extend(["", "## Вероятности итоговых мест, %", ""])
    rows = [
        [item["team"]]
        + [f"{probability * 100:.1f}" for probability in item["place_probabilities"]]
        for item in forecasts
    ]
    lines.extend(
        _md_table(
            ["Команда"] + [str(place) for place in range(1, count + 1)],
            rows,
            ["left"] + ["right"] * count,
        )
    )

    lines.extend(["", "## Ожидаемые итоговые показатели", ""])
    rows = [
        [
            item["team"],
            f"{item['expected_place']:.2f}",
            f"{item['expected_wins']:.1f}",
            f"{item['expected_points']:.1f}",
            f"{item['expected_sets_won']:.1f}:{item['expected_sets_lost']:.1f}",
            _ratio(item["expected_sets_won"], item["expected_sets_lost"]),
        ]
        for item in forecasts
    ]
    lines.extend(
        _md_table(
            ["Команда", "Место", "Победы", "Очки", "Партии", "Ratio"],
            rows,
            ["left", "right", "right", "right", "right", "right"],
        )
    )

    # ---------------------------------------------------------- Расстановки
    lines.extend(["", "## Наиболее вероятные итоговые расстановки", ""])
    for number, item in enumerate(data["most_likely_rankings"], 1):
        order = ", ".join(_md_escape(name) for name in item["order"])
        lines.append(
            f"{number}. **{item['probability'] * 100:.1f}%** — {order}"
        )
    if count >= 5:
        lines.extend(
            [
                "",
                "*Конкретных полных расстановок очень много, поэтому "
                "вероятность каждой из них невелика — надёжнее смотреть на "
                "вероятности мест выше.*",
            ]
        )

    # ---------------------------------------------------------- Команда
    analysis = data.get("team_analysis")
    if analysis is None:
        lines.extend(
            [
                "",
                "Чтобы получить вероятности мест и сценарии для команды, "
                "запустите с параметром `--team`.",
            ]
        )
        return "\n".join(lines)

    team_name = analysis["team"]
    current = analysis["current"]
    forecast = next(
        item for item in data["forecasts"] if item["team"] == team_name
    )
    team_remaining = sum(
        1
        for match in data["remaining_matches"]
        if team_name in (match["a"], match["b"])
    )

    lines.extend(["", f"## Команда «{_md_escape(team_name)}»", ""])
    current_line = (
        f"- **Сейчас:** {analysis['current_place']}-е место, "
        f"{current['wins']} "
        f"{_plural(current['wins'], 'победа', 'победы', 'побед')}, "
        f"{current['points']} "
        f"{_plural(current['points'], 'очко', 'очка', 'очков')}, "
        f"партии {current['sets_won']}:{current['sets_lost']}"
    )
    if current["initial_wins"] or current["initial_points"]:
        current_line += (
            f" (включая гандикап +{current['initial_wins']}/"
            f"+{current['initial_points']})"
        )
    lines.append(current_line)
    lines.append(f"- **Осталось матчей у команды:** {team_remaining}")
    lines.append(
        f"- **Ожидание:** {forecast['expected_wins']:.2f} побед, "
        f"{forecast['expected_points']:.2f} очков, партии "
        f"{forecast['expected_sets_won']:.1f}:"
        f"{forecast['expected_sets_lost']:.1f}"
    )

    place_count = len(analysis["place_probabilities"])
    top3 = min(3, place_count)
    top5 = min(5, place_count)
    lines.extend(["", "### Вероятности мест", ""])
    rows = [
        [f"{place}-е", f"{probability * 100:.1f}%"]
        for place, probability in enumerate(
            analysis["place_probabilities"], 1
        )
    ]
    lines.extend(
        _md_table(["Место", "Вероятность"], rows, ["right", "right"])
    )
    summary = (
        f"**P(1–{top3}):** "
        f"{sum(analysis['place_probabilities'][:top3]) * 100:.1f}%"
    )
    if top5 != top3:
        summary += (
            f"; **P(1–{top5}):** "
            f"{sum(analysis['place_probabilities'][:top5]) * 100:.1f}%"
        )
    lines.extend(["", summary])

    # ---------------------------------------------------------- Сценарии
    target = analysis["target_place"]
    lines.extend(["", f"### Сценарии для {target}-го места", ""])
    lines.append(
        f"**Вероятность:** {analysis['target_place_probability'] * 100:.1f}% "
        f"({analysis['target_place_observations']} из "
        f"{data['simulations']} симуляций)."
    )

    if not data["remaining_matches"]:
        lines.extend(["", "Все матчи уже сыграны — оставшихся игр нет."])
        return "\n".join(lines)
    if analysis["target_place_observations"] == 0:
        lines.extend(
            [
                "",
                "Это место не встретилось ни в одной симуляции. "
                "Увеличьте `--sims` или проверьте данные.",
            ]
        )
        return "\n".join(lines)

    lines.extend(
        [
            "",
            "Условные вероятности исходов оставшихся матчей, % "
            "(счёт — для первой команды пары):",
            "",
        ]
    )
    rows = []
    for match in analysis["conditional_matches"]:
        outcomes = match["outcomes"]
        best = max(outcomes.values())
        modes = [
            key
            for key in _MD_ORDER
            if best - outcomes.get(key, 0.0) < 1e-9
        ]
        rows.append(
            [f"{match['a']} — {match['b']}"]
            + [f"{outcomes.get(key, 0.0) * 100:.1f}" for key in _MD_ORDER]
            + ["/".join(modes)]
        )
    lines.extend(
        _md_table(
            ["Матч", *_MD_ORDER, "Мода"],
            rows,
            ["left"] + ["right"] * 6 + ["right"],
        )
    )
    lines.extend(
        [
            "",
            "*Мода — самый вероятный счёт отдельного матча при условии "
            "места; у совместного сценария всех матчей вероятность всегда "
            "ниже.*",
        ]
    )

    if analysis["scenarios"] and len(data["remaining_matches"]) <= 6:
        lines.extend(
            ["", "#### Наиболее вероятные точные наборы исходов", ""]
        )
        for number, scenario in enumerate(analysis["scenarios"][:top], 1):
            lines.append(
                f"{number}. "
                f"**{scenario['probability_given_place'] * 100:.2f}%** "
                f"при условии места "
                f"({scenario['probability'] * 100:.3f}% безусловно)"
            )
            for item in scenario["results"]:
                lines.append(
                    f"   - {_md_escape(item['a'])}—"
                    f"{_md_escape(item['b'])} {item['score']}"
                )

    return "\n".join(lines)


def report_to_dict(
    tournament: Tournament,
    model: StrengthModel,
    result: SimulationResult,
    *,
    team: str | None = None,
    target_place: int | None = None,
    top: int = 5,
    uncertainty: bool | None = None,
) -> dict:
    standings = tournament.standings()
    ranking = tournament.ranking()
    if uncertainty is None:
        uncertainty = result.uncertainty

    data: dict = {
        "tournament": tournament.name,
        "played_matches": len(tournament.matches),
        "total_matches": tournament.total_matches,
        "simulations": result.sims,
        "model": {
            "type": "bradley-terry-sets",
            "prior": model.prior,
            "rating_uncertainty": uncertainty,
        },
        "ratings": [
            {
                "team": name,
                "rating": model.rating(name),
                "set_probability_vs_average": model.p_set_vs_average(name),
            }
            for name in model.teams
        ],
        "handicap": {
            team: {"wins": wins, "points": points}
            for team, (wins, points) in tournament.handicap.items()
        },
        "standings": [
            {
                "place": position,
                "team": row.team,
                "played": row.played,
                "wins": row.wins,
                "points": row.points,
                "initial_wins": tournament.initial_wins(row.team),
                "initial_points": tournament.initial_points(row.team),
                "sets_won": row.sets_won,
                "sets_lost": row.sets_lost,
                "set_ratio": _finite(row.set_ratio),
            }
            for position, row in enumerate(ranking, 1)
        ],
        "forecasts": [
            {
                "team": name,
                "expected_place": result.forecasts[name].expected_place,
                "expected_wins": result.forecasts[name].expected_wins,
                "expected_points": result.forecasts[name].expected_points,
                "expected_sets_won": result.forecasts[name].expected_sets_won,
                "expected_sets_lost": result.forecasts[name].expected_sets_lost,
                "place_probabilities": result.forecasts[name].place_probs,
            }
            for name in result.teams
        ],
        "most_likely_rankings": [
            {
                "probability": count / result.sims,
                "order": [result.teams[i] for i in order],
            }
            for order, count in result.ranking_counts.most_common(top)
        ],
        "remaining_matches": [
            {"a": a, "b": b} for a, b in result.remaining
        ],
    }

    if team is not None:
        forecast = result.forecasts[team]
        if target_place is None:
            target_place = max(
                range(1, len(forecast.place_probs) + 1),
                key=lambda place: forecast.place_probs[place - 1],
            )
        team_data: dict = {
            "team": team,
            "current_place": next(
                position
                for position, row in enumerate(ranking, 1)
                if row.team == team
            ),
            "current": {
                "wins": standings[team].wins,
                "points": standings[team].points,
                "initial_wins": tournament.initial_wins(team),
                "initial_points": tournament.initial_points(team),
                "sets_won": standings[team].sets_won,
                "sets_lost": standings[team].sets_lost,
            },
            "place_probabilities": forecast.place_probs,
            "target_place": target_place,
            "target_place_probability": forecast.place_probs[target_place - 1],
            "target_place_observations": (
                result.place_counts[target_place - 1]
                if result.place_counts
                else 0
            ),
            "conditional_matches": [],
            "scenarios": [],
        }

        if result.conditional_outcomes and result.remaining:
            for match_index, (a, b) in enumerate(result.remaining):
                probabilities = result.conditional_outcomes[target_place - 1][
                    match_index
                ]
                if b == team:
                    a, b = b, a
                    probabilities = list(reversed(probabilities))
                team_data["conditional_matches"].append(
                    {
                        "a": a,
                        "b": b,
                        "outcomes": {
                            f"{outcome[0]}:{outcome[1]}": probability
                            for outcome, probability in zip(
                                OUTCOMES, probabilities
                            )
                        },
                    }
                )

        if result.scenarios and result.remaining:
            for scenario in result.scenarios[target_place - 1][:top]:
                scenario_data = []
                for match_index, outcome in enumerate(scenario.outcomes):
                    a, b = result.remaining[match_index]
                    a_sets, b_sets = OUTCOMES[outcome]
                    if team == b:
                        a, b = b, a
                        a_sets, b_sets = b_sets, a_sets
                    scenario_data.append(
                        {"a": a, "b": b, "score": f"{a_sets}:{b_sets}"}
                    )
                team_data["scenarios"].append(
                    {
                        "probability_given_place": scenario.probability_given_place,
                        "probability": scenario.probability,
                        "results": scenario_data,
                    }
                )

        data["team_analysis"] = team_data

    return data
