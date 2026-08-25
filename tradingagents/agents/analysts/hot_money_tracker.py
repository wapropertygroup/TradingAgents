"""A-share hot-money and capital-flow analyst."""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    get_concept_blocks,
    get_dragon_tiger_board,
    get_fund_flow,
    get_hot_stocks,
    get_industry_comparison,
    get_insider_transactions,
    get_instrument_context_from_state,
    get_language_instruction,
    get_news,
    get_northbound_flow,
    get_stock_data,
)

HOT_MONEY_TOOLS = (
    get_stock_data, get_news, get_insider_transactions, get_hot_stocks,
    get_northbound_flow, get_concept_blocks, get_fund_flow,
    get_dragon_tiger_board, get_industry_comparison,
)


def create_hot_money_tracker(llm):
    """Create an analyst for Dragon-Tiger, volume, and main-capital signals."""

    def hot_money_tracker_node(state):
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)
        # The first three tools are the user-approved minimum contract.
        tools = list(HOT_MONEY_TOOLS)
        system_message = (
            "You are the Hot Money Tracker for a mainland China A-share. Track "
            "Dragon-Tiger List seats, large-order and main-capital flow, abnormal "
            "turnover and volume, limit-up behavior, theme rotation, northbound "
            "flow, and relevant insider activity. Separate verified seat/flow "
            "data from inference. At minimum use price/volume, news, and insider "
            "tools; use dedicated signal tools when data exists. Conclude whether "
            "the evidence indicates accumulation, distribution, speculative relay, "
            "mixed positioning, or no verified signal. End with a Markdown table "
            "and label missing evidence instead of guessing."
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
            "hot_money_report": result.content if not result.tool_calls else "",
        }

    return hot_money_tracker_node
