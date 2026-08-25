"""A-share policy analyst."""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    get_global_news,
    get_instrument_context_from_state,
    get_language_instruction,
    get_news,
)

POLICY_TOOLS = (get_news, get_global_news)


def create_policy_analyst(llm):
    """Create an analyst for regulatory, macro, and industrial policy."""

    def policy_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)
        tools = list(POLICY_TOOLS)
        system_message = (
            "You are the Policy Analyst for a mainland China A-share. Analyze "
            "monetary and fiscal policy, CSRC and other regulatory actions, "
            "industrial policy, local-government support, window guidance, and "
            "international restrictions that affect the company or its sector. "
            "For each verified policy, identify the issuer and date, distinguish "
            "announced policy from speculation, trace policy -> sector -> company "
            "-> financial impact, and estimate direction, strength, and time "
            "horizon. Use get_news for company/sector policy evidence and "
            "get_global_news for macro policy. End with an overall policy rating "
            "(major positive/positive/neutral/negative/major negative) and a "
            "Markdown table. Mark unavailable evidence explicitly; never invent "
            "a policy or publication date."
            + get_language_instruction()
        )
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You collaborate with other investment-research agents. Use the "
                "provided tools before reaching conclusions. Available tools: "
                "{tool_names}. The analysis date is {current_date}. "
                "{instrument_context}\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]).partial(
            tool_names=", ".join(tool.name for tool in tools),
            current_date=current_date,
            instrument_context=instrument_context,
            system_message=system_message,
        )
        result = (prompt | llm.bind_tools(tools)).invoke(state["messages"])
        return {
            "messages": [result],
            "policy_report": result.content if not result.tool_calls else "",
        }

    return policy_analyst_node
