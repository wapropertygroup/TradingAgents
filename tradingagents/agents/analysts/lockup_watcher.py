"""A-share lock-up, reduction, and pledge analyst."""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    get_fundamentals,
    get_insider_transactions,
    get_instrument_context_from_state,
    get_language_instruction,
    get_lockup_expiry,
    get_news,
)

LOCKUP_TOOLS = (get_insider_transactions, get_news, get_fundamentals, get_lockup_expiry)


def create_lockup_watcher(llm):
    """Create an analyst for restricted-share supply shocks."""

    def lockup_watcher_node(state):
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)
        # The first three tools are the user-approved minimum contract.
        tools = list(LOCKUP_TOOLS)
        system_message = (
            "You are the Lock-up Monitor for a mainland China A-share. Assess "
            "restricted-share unlock schedules, unlock size versus float, holder "
            "type and cost basis where verified, announced major-shareholder or "
            "director reductions, equity pledges, and applicable reduction limits. "
            "Distinguish an unlock from an announced sale: unlocked shares are "
            "potential supply, not proof of selling. Cover the next 90 days and "
            "recent holder behavior. Conclude with major/moderate/minor/no evident "
            "pressure and a Markdown event table. Mark missing data explicitly and "
            "never infer a sale solely from an unlock date."
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
            "lockup_report": result.content if not result.tool_calls else "",
        }

    return lockup_watcher_node
