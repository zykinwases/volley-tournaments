import math
import tempfile
import unittest
from pathlib import Path

from volley.model import fit_model
from volley.report import (
    _md_escape,
    format_markdown,
    format_report,
    report_to_dict,
)
from volley.simulate import simulate
from volley.tournament import (
    Match,
    Tournament,
    TournamentError,
    load_tournament,
    points_for,
    repeated_pair_suffixes,
    save_tournament,
)


class ScoringTests(unittest.TestCase):
    def test_points(self):
        self.assertEqual(points_for(3, 0), (3, 0))
        self.assertEqual(points_for(3, 1), (3, 0))
        self.assertEqual(points_for(3, 2), (2, 1))
        self.assertEqual(points_for(2, 3), (1, 2))
        self.assertEqual(points_for(1, 3), (0, 3))
        self.assertEqual(points_for(0, 3), (0, 3))

    def test_ranking_by_points_then_ratio(self):
        teams = ["A", "B", "C", "D"]
        matches = [
            Match("A", "C", 3, 0),
            Match("D", "A", 3, 1),
            Match("B", "C", 3, 2),
            Match("D", "B", 3, 2),
        ]
        tournament = Tournament(teams, matches)
        ranking = [row.team for row in tournament.ranking()]
        self.assertEqual(ranking, ["D", "A", "B", "C"])

        standings = tournament.standings()
        self.assertEqual(standings["A"].points, 3)
        self.assertEqual(standings["B"].points, 3)
        self.assertGreater(standings["A"].set_ratio, standings["B"].set_ratio)

    def test_set_ratio_tiebreak(self):
        teams = ["A", "B", "C"]
        matches = [Match("A", "C", 3, 1), Match("B", "C", 2, 3)]
        tournament = Tournament(teams, matches)
        ranking = [row.team for row in tournament.ranking()]
        self.assertEqual(ranking, ["A", "C", "B"])
        standings = tournament.standings()
        self.assertEqual(standings["A"].points, 3)
        self.assertEqual(standings["C"].points, 2)
        self.assertEqual(standings["B"].points, 1)


class HandicapTests(unittest.TestCase):
    def test_standings_and_ranking(self):
        tournament = Tournament(
            ["A", "B", "C"],
            [],
            handicap={"B": {"wins": 1, "points": 2}, "C": [0, 3]},
        )
        standings = tournament.standings()
        self.assertEqual(
            (standings["B"].wins, standings["B"].points), (1, 2)
        )
        self.assertEqual(
            (standings["C"].wins, standings["C"].points), (0, 3)
        )
        self.assertEqual(standings["B"].played, 0)
        # Победы важнее очков: B с одной победой выше C с тремя очками.
        self.assertEqual(
            [row.team for row in tournament.ranking()], ["B", "C", "A"]
        )

    def test_roundtrip(self):
        tournament = Tournament(
            ["A", "B"],
            [Match("A", "B", 3, 2)],
            handicap={"B": (2, 4)},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tournament.json"
            save_tournament(tournament, path)
            loaded = load_tournament(path)
        self.assertEqual(loaded.handicap, {"B": (2, 4)})
        self.assertEqual(loaded.initial_wins("B"), 2)
        self.assertEqual(loaded.initial_points("B"), 4)

    def test_validation(self):
        with self.assertRaises(TournamentError):
            Tournament(["A", "B"], [], handicap={"C": (1, 2)})
        with self.assertRaises(TournamentError):
            Tournament(["A", "B"], [], handicap={"A": (-1, 2)})
        with self.assertRaises(TournamentError):
            Tournament(["A", "B"], [], handicap={"A": (True, 2)})
        with self.assertRaises(TournamentError):
            Tournament(["A", "B"], [], handicap={"A": {"wins": 1, "x": 2}})

    def test_simulation_uses_handicap(self):
        tournament = Tournament(
            ["A", "B"], [], handicap={"A": (2, 6)}
        )
        model = fit_model(tournament.teams, tournament.matches)
        result = simulate(
            tournament, model, sims=200, seed=1, team="A", uncertainty=False
        )
        self.assertEqual(result.forecasts["A"].place_probs, [1.0, 0.0])
        self.assertGreaterEqual(result.forecasts["A"].expected_wins, 2.0)


class RoundsTests(unittest.TestCase):
    def test_double_round_schedule(self):
        tournament = Tournament(["A", "B", "C"], rounds=2)
        self.assertEqual(tournament.total_matches, 6)
        remaining = tournament.remaining_pairs()
        self.assertEqual(remaining.count(("A", "B")), 2)
        self.assertEqual(remaining.count(("A", "C")), 2)
        self.assertEqual(remaining.count(("B", "C")), 2)
        self.assertEqual(len(remaining), 6)

    def test_two_meetings_allowed(self):
        tournament = Tournament(["A", "B"], rounds=2)
        tournament.add_result("A", "B", "3:0")
        tournament.add_result("B", "A", "3:2")
        self.assertEqual(tournament.remaining_pairs(), [])
        standings = tournament.standings()
        self.assertEqual((standings["A"].played, standings["B"].played), (2, 2))
        self.assertEqual((standings["A"].wins, standings["B"].wins), (1, 1))
        with self.assertRaises(TournamentError):
            tournament.add_result("A", "B", "3:1")

    def test_third_meeting_rejected_on_load(self):
        with self.assertRaises(TournamentError):
            Tournament(
                ["A", "B"],
                [
                    Match("A", "B", 3, 0),
                    Match("B", "A", 3, 1),
                    Match("A", "B", 3, 2),
                ],
                rounds=2,
            )

    def test_rounds_roundtrip(self):
        tournament = Tournament(["A", "B"], [Match("A", "B", 3, 0)], rounds=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tournament.json"
            save_tournament(tournament, path)
            saved = path.read_text(encoding="utf-8")
            loaded = load_tournament(path)
        self.assertIn('"rounds": 2', saved)
        self.assertEqual(loaded.rounds, 2)
        self.assertEqual(loaded.total_matches, 2)

    def test_single_round_rounds_not_saved(self):
        tournament = Tournament(["A", "B"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tournament.json"
            save_tournament(tournament, path)
            saved = path.read_text(encoding="utf-8")
            loaded = load_tournament(path)
        self.assertNotIn('"rounds"', saved)
        self.assertEqual(loaded.rounds, 1)

    def test_invalid_rounds(self):
        for rounds in (0, -1, True, 1.5):
            with self.assertRaises(TournamentError):
                Tournament(["A", "B"], rounds=rounds)

    def test_pair_suffixes(self):
        suffixes = repeated_pair_suffixes([("A", "B"), ("A", "B"), ("A", "C")])
        self.assertEqual(suffixes, [" (1/2)", " (2/2)", ""])

    def test_simulation_double_round(self):
        tournament = Tournament(["A", "B"], rounds=2)
        model = fit_model(tournament.teams, tournament.matches)
        result = simulate(
            tournament, model, sims=300, seed=11, team="A", uncertainty=False
        )
        self.assertEqual(len(result.remaining), 2)
        self.assertEqual(len(result.forecasts["A"].place_probs), 2)
        self.assertAlmostEqual(
            sum(forecast.expected_wins for forecast in result.forecasts.values()),
            2.0,
        )
        self.assertAlmostEqual(
            sum(result.forecasts["A"].place_probs), 1.0
        )

    def test_reports_show_rounds_and_legs(self):
        tournament = Tournament(["A", "B"], rounds=2)
        model = fit_model(tournament.teams, tournament.matches)
        result = simulate(
            tournament, model, sims=400, seed=5, team="A", uncertainty=False
        )
        text = format_report(
            tournament, model, result, team="A", target_place=1, top=3
        )
        self.assertIn("Формат: два круга", text)
        self.assertIn("A — B (1/2)", text)
        self.assertIn("A — B (2/2)", text)
        markdown = format_markdown(
            tournament, model, result, team="A", target_place=1, top=3
        )
        self.assertIn("**Формат:** два круга", markdown)
        self.assertIn("A — B (1/2)", markdown)
        payload = report_to_dict(
            tournament, model, result, team="A", target_place=1, top=3
        )
        self.assertEqual(payload["rounds"], 2)
        self.assertEqual(len(payload["remaining_matches"]), 2)


class ValidationTests(unittest.TestCase):
    def test_invalid_scores(self):
        for a, b in [(3, 3), (2, 2), (4, 0), (1, 1)]:
            with self.assertRaises(TournamentError):
                Match("A", "B", a, b)

    def test_duplicate_pair(self):
        with self.assertRaises(TournamentError):
            Tournament(
                ["A", "B"],
                [Match("A", "B", 3, 0), Match("B", "A", 3, 1)],
            )

    def test_self_match(self):
        with self.assertRaises(TournamentError):
            Match("A", "A", 3, 0)


class ModelTests(unittest.TestCase):
    def test_outcome_probabilities(self):
        tournament = Tournament(["A", "B"], [Match("A", "B", 3, 0)])
        model = fit_model(tournament.teams, tournament.matches)
        for a, b in [("A", "B"), ("B", "A")]:
            probabilities = model.outcome_probs(a, b)
            self.assertAlmostEqual(sum(probabilities), 1.0, places=12)
            self.assertTrue(all(p >= 0 for p in probabilities))
            reversed_probabilities = list(reversed(model.outcome_probs(b, a)))
            for left, right in zip(probabilities, reversed_probabilities):
                self.assertAlmostEqual(left, right, places=12)

    def test_fit_recovers_order(self):
        teams = ["Strong", "Good", "Middle", "Weak"]
        matches = []
        for strong in ["Good", "Middle", "Weak"]:
            matches.append(Match("Strong", strong, 3, 0))
        matches.append(Match("Good", "Middle", 3, 1))
        matches.append(Match("Good", "Weak", 3, 0))
        matches.append(Match("Middle", "Weak", 3, 1))
        model = fit_model(teams, matches)
        ratings = dict(zip(model.teams, model.values))
        self.assertGreater(ratings["Strong"], ratings["Good"])
        self.assertGreater(ratings["Good"], ratings["Middle"])
        self.assertGreater(ratings["Middle"], ratings["Weak"])

    def test_no_data(self):
        model = fit_model(["A", "B"], [])
        self.assertEqual(model.values, [0.0, 0.0])
        self.assertAlmostEqual(model.p_set("A", "B"), 0.5)


class SimulationTests(unittest.TestCase):
    def setUp(self):
        self.tournament = Tournament(
            ["A", "B", "C", "D"],
            [
                Match("A", "B", 3, 0),
                Match("C", "D", 3, 2),
                Match("A", "C", 3, 1),
                Match("B", "D", 2, 3),
            ],
        )
        self.model = fit_model(self.tournament.teams, self.tournament.matches)

    def test_probabilities_sum_to_one(self):
        result = simulate(
            self.tournament,
            self.model,
            sims=4000,
            seed=1,
            team="A",
            uncertainty=False,
        )
        for team in result.teams:
            self.assertAlmostEqual(
                sum(result.forecasts[team].place_probs), 1.0, places=9
            )
        self.assertEqual(sum(result.place_counts), result.sims)

    def test_reproducible(self):
        first = simulate(
            self.tournament, self.model, sims=1000, seed=42, uncertainty=False
        )
        second = simulate(
            self.tournament, self.model, sims=1000, seed=42, uncertainty=False
        )
        self.assertEqual(first.ranking_counts, second.ranking_counts)

    def test_conditional_outcomes_are_distribution(self):
        # В турнире из двух команд единственный матч решает место.
        # При условии первого места команда A обязана выиграть,
        # поэтому условные вероятности — это P(счёт | победа A).
        tournament = Tournament(["A", "B"], [])
        model = fit_model(tournament.teams, tournament.matches)
        result = simulate(
            tournament, model, sims=6000, seed=3, team="A", uncertainty=False
        )
        conditional = result.conditional_outcomes[0][0]
        self.assertAlmostEqual(sum(conditional), 1.0, places=12)
        self.assertAlmostEqual(conditional[0], 0.25, places=1)
        self.assertAlmostEqual(conditional[1], 0.375, places=1)
        self.assertAlmostEqual(conditional[2], 0.375, places=1)
        for probability in conditional[3:]:
            self.assertAlmostEqual(probability, 0.0, places=12)

    def test_all_matches_played(self):
        tournament = Tournament(
            ["A", "B"], [Match("A", "B", 3, 0)]
        )
        model = fit_model(tournament.teams, tournament.matches)
        result = simulate(
            tournament, model, sims=50, seed=1, team="A", uncertainty=False
        )
        self.assertEqual(result.remaining, [])
        self.assertEqual(result.forecasts["A"].place_probs, [1.0, 0.0])

    def test_team_without_matches(self):
        tournament = Tournament(
            ["A", "B", "C"], [Match("A", "B", 3, 0)]
        )
        model = fit_model(tournament.teams, tournament.matches)
        result = simulate(
            tournament, model, sims=3000, seed=2, team="C", uncertainty=False
        )
        self.assertAlmostEqual(
            sum(result.forecasts["C"].place_probs), 1.0, places=9
        )
        self.assertEqual(sum(result.place_counts), result.sims)

    def test_uncertainty_sums_to_one(self):
        result = simulate(
            self.tournament,
            self.model,
            sims=2000,
            seed=5,
            team="B",
            uncertainty=True,
        )
        for team in result.teams:
            self.assertAlmostEqual(
                sum(result.forecasts[team].place_probs), 1.0, places=9
            )


class IOAndReportTests(unittest.TestCase):
    def test_roundtrip(self):
        tournament = Tournament(
            ["A", "B", "C"],
            [Match("A", "B", 3, 2), Match("B", "C", 3, 0)],
            name="Test Cup",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tournament.json"
            save_tournament(tournament, path)
            loaded = load_tournament(path)
        self.assertEqual(loaded.name, "Test Cup")
        self.assertEqual(loaded.teams, tournament.teams)
        self.assertEqual(len(loaded.matches), 2)

    def test_report_and_json(self):
        tournament = Tournament(
            ["A", "B", "C"],
            [Match("A", "B", 3, 1), Match("B", "C", 3, 2)],
        )
        model = fit_model(tournament.teams, tournament.matches)
        result = simulate(
            tournament, model, sims=500, seed=9, team="A", uncertainty=False
        )
        text = format_report(
            tournament, model, result, team="A", target_place=1, top=3
        )
        self.assertIn("ПРОГНОЗ ТУРНИРА", text)
        self.assertIn("Вероятности мест", text)
        self.assertIn("Симуляций", text)
        payload = report_to_dict(
            tournament, model, result, team="A", target_place=1, top=3
        )
        self.assertEqual(payload["team_analysis"]["target_place"], 1)
        observations = payload["team_analysis"]["place_observations"]
        self.assertEqual(len(observations), 3)
        self.assertEqual(sum(observations), 500)
        self.assertAlmostEqual(
            sum(payload["team_analysis"]["place_probabilities"]), 1.0
        )

    def test_markdown_report(self):
        tournament = Tournament(
            ["A", "B", "C"],
            [Match("A", "B", 3, 1), Match("B", "C", 3, 2)],
            handicap={"B": (1, 2)},
        )
        model = fit_model(tournament.teams, tournament.matches)
        result = simulate(
            tournament, model, sims=500, seed=9, team="A", uncertainty=False
        )
        markdown = format_markdown(
            tournament, model, result, team="A", target_place=1, top=3
        )
        self.assertTrue(markdown.startswith("# Прогноз турнира"))
        self.assertIn("## Текущая таблица", markdown)
        self.assertIn("## Стартовый гандикап", markdown)
        self.assertIn("## Команда «A»", markdown)
        self.assertIn("### Вероятности мест", markdown)
        self.assertIn("| Место | Вероятность | Симуляций |", markdown)
        self.assertIn("### Сценарии для 1-го места", markdown)
        self.assertIn("| Матч | 3:0 | 3:1 | 3:2 | 2:3 | 1:3 | 0:3 | Мода |", markdown)
        # Все строки таблиц должны начинаться с «|».
        for line in markdown.splitlines():
            if line.startswith("|"):
                self.assertTrue(line.rstrip().endswith("|"), line)
        # Символ «|» в названии команды не ломает таблицу.
        escaped = _md_escape("A|B")
        self.assertEqual(escaped, "A\\|B")

    def test_markdown_without_team(self):
        tournament = Tournament(["A", "B"], [Match("A", "B", 3, 0)])
        model = fit_model(tournament.teams, tournament.matches)
        result = simulate(tournament, model, sims=200, seed=3, uncertainty=False)
        markdown = format_markdown(tournament, model, result)
        self.assertIn("## Наиболее вероятные итоговые расстановки", markdown)
        self.assertIn("`--team`", markdown)
        self.assertNotIn("## Команда", markdown)


if __name__ == "__main__":
    unittest.main()
