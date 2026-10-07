import unittest

from tradingagents.graph.analyst_execution import (
    build_analyst_execution_plan,
)


class AnalystExecutionPlanTests(unittest.TestCase):
    def test_build_plan_preserves_selected_order(self):
        plan = build_analyst_execution_plan(["news", "market"])

        self.assertEqual([spec.key for spec in plan.specs], ["news", "market"])
        self.assertEqual(plan.specs[0].agent_node, "News Analyst")
        self.assertEqual(plan.specs[0].report_key, "news_report")
        self.assertFalse(hasattr(plan.specs[0], "clear_node"))

    def test_rejects_unknown_analyst_keys(self):
        with self.assertRaises(ValueError):
            build_analyst_execution_plan(["market", "macro"])

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

        ``report_key`` is what every downstream consumer reads, and the node names
        are what a checkpointed run resumes into, so a rename here silently
        detaches the analyst from its readers or from a saved run.
        """
        spec = build_analyst_execution_plan(["earnings"]).specs[0]
        self.assertEqual(spec.key, "earnings")
        self.assertEqual(spec.agent_node, "Earnings Analyst")
        self.assertEqual(spec.report_key, "earnings_report")
        # Each analyst is a subgraph with its own tools and messages since the
        # analysts run side by side, so no clear or tool node is named here.
        self.assertFalse(hasattr(spec, "clear_node"))
        self.assertFalse(hasattr(spec, "tool_node"))

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

    def test_code_driven_analysts_have_a_tool_node_for_every_call_they_emit(self):
        """The earnings, quality and valuation analysts emit their tool calls by
        name, from code, without binding tools to a model. Their tool node is
        built from the same ``TOOLS`` tuple, so a name the node does not hold
        would come back as a tool error and be reported as missing evidence."""
        from tradingagents.agents.analysts import (
            earnings_analyst,
            quality_analyst,
            valuation_analyst,
        )

        emitted = {
            "earnings": {earnings_analyst.EVIDENCE_TOOL, earnings_analyst.COMMENTARY_TOOL},
            "quality": {quality_analyst.EVIDENCE_TOOL},
            "valuation": {valuation_analyst.EVIDENCE_TOOL},
        }
        for key, names in emitted.items():
            with self.subTest(analyst=key):
                spec = build_analyst_execution_plan([key]).specs[0]
                self.assertEqual({tool.name for tool in spec.tools}, names)
