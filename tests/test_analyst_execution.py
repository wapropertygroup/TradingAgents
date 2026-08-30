import unittest

from tradingagents.graph.analyst_execution import (
    AnalystWallTimeTracker,
    build_analyst_execution_plan,
    get_initial_analyst_node,
    sync_analyst_tracker_from_chunk,
)


class AnalystExecutionPlanTests(unittest.TestCase):
    def test_build_plan_preserves_selected_order(self):
        plan = build_analyst_execution_plan(["news", "market"])

        self.assertEqual([spec.key for spec in plan.specs], ["news", "market"])
        self.assertEqual(plan.specs[0].agent_node, "News Analyst")
        self.assertEqual(plan.specs[0].tool_node, "tools_news")
        self.assertEqual(plan.specs[0].clear_node, "Msg Clear News")

    def test_rejects_unknown_analyst_keys(self):
        with self.assertRaises(ValueError):
            build_analyst_execution_plan(["market", "macro"])

    def test_get_initial_analyst_node_uses_plan_metadata(self):
        plan = build_analyst_execution_plan(["fundamentals", "news"])

        self.assertEqual(
            get_initial_analyst_node(plan),
            "Fundamentals Analyst",
        )

    def test_social_key_displays_as_sentiment_analyst(self):
        # The wire key stays "social" for saved-config back-compat, but the
        # user-visible agent_node label must match the v0.2.5 rename so the
        # wall-time summary and any future consumer of agent_node says
        # "Sentiment Analyst" rather than the legacy "Social Analyst".
        plan = build_analyst_execution_plan(["social"])
        spec = plan.specs[0]
        self.assertEqual(spec.key, "social")
        self.assertEqual(spec.agent_node, "Sentiment Analyst")
        self.assertEqual(spec.report_key, "sentiment_report")

    def test_astock_plan_has_all_seven_analysts_in_order(self):
        keys = ["market", "social", "news", "fundamentals", "policy", "hot_money", "lockup"]
        plan = build_analyst_execution_plan(keys)

        self.assertEqual([spec.key for spec in plan.specs], keys)
        self.assertEqual(plan.specs[4].report_key, "policy_report")
        self.assertEqual(plan.specs[5].agent_node, "Hot Money Tracker")
        self.assertEqual(plan.specs[6].agent_node, "Lock-up Monitor")

    def test_earnings_spec_strings_are_stable(self):
        """These strings are graph node names and a state key.

        The tool and clear node names are matched exactly by
        ``ConditionalLogic.should_continue_earnings``, and ``report_key`` is what
        every downstream consumer reads, so a rename here silently detaches the
        analyst from its own routing.
        """
        spec = build_analyst_execution_plan(["earnings"]).specs[0]
        self.assertEqual(spec.key, "earnings")
        self.assertEqual(spec.agent_node, "Earnings Analyst")
        self.assertEqual(spec.clear_node, "Msg Clear Earnings")
        self.assertEqual(spec.tool_node, "tools_earnings")
        self.assertEqual(spec.report_key, "earnings_report")

    def test_earnings_can_be_selected_in_any_position(self):
        for keys in (
            ["earnings"],
            ["earnings", "market"],
            ["market", "earnings", "news"],
            ["market", "social", "news", "fundamentals", "earnings"],
        ):
            with self.subTest(keys=tuple(keys)):
                plan = build_analyst_execution_plan(keys)
                self.assertEqual([spec.key for spec in plan.specs], keys)

    def test_every_report_key_is_unique_across_all_analysts(self):
        """Two analysts sharing a report key would silently overwrite each other."""
        from tradingagents.graph.analyst_execution import ANALYST_NODE_SPECS

        keys = [spec.report_key for spec in ANALYST_NODE_SPECS.values()]
        self.assertEqual(len(keys), len(set(keys)), keys)
        node_names = [spec.agent_node for spec in ANALYST_NODE_SPECS.values()]
        self.assertEqual(len(node_names), len(set(node_names)), node_names)


class AnalystWallTimeTrackerTests(unittest.TestCase):
    def test_records_wall_time_when_analyst_completes(self):
        plan = build_analyst_execution_plan(["market", "news"])
        tracker = AnalystWallTimeTracker(plan)

        tracker.mark_started("market", started_at=10.0)
        tracker.mark_completed("market", completed_at=13.5)

        self.assertEqual(tracker.get_wall_times(), {"market": 3.5})

    def test_formats_summary_in_plan_order(self):
        plan = build_analyst_execution_plan(["news", "market"])
        tracker = AnalystWallTimeTracker(plan)

        tracker.mark_started("market", started_at=20.0)
        tracker.mark_completed("market", completed_at=22.25)
        tracker.mark_started("news", started_at=10.0)
        tracker.mark_completed("news", completed_at=14.0)

        self.assertEqual(
            tracker.format_summary(),
            "Analyst wall time: News 4.00s | Market 2.25s",
        )

    def test_syncs_wall_time_from_sequential_chunks(self):
        plan = build_analyst_execution_plan(["market", "news"])
        tracker = AnalystWallTimeTracker(plan)

        sync_analyst_tracker_from_chunk(tracker, {}, now=10.0)
        self.assertEqual(tracker.get_wall_times(), {})

        sync_analyst_tracker_from_chunk(
            tracker,
            {"market_report": "done"},
            now=13.0,
        )
        self.assertEqual(tracker.get_wall_times(), {"market": 3.0})

        sync_analyst_tracker_from_chunk(
            tracker,
            {"market_report": "done", "news_report": "done"},
            now=18.0,
        )
        self.assertEqual(
            tracker.get_wall_times(),
            {"market": 3.0, "news": 5.0},
        )
