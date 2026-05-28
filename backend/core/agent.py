import traceback
from typing import TypedDict, Annotated, Any

import pandas as pd
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from config import settings


# ── Lazy LLM init (avoids crash if key is missing at import time) ──
def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o", temperature=0, api_key=settings.openai_api_key)


# ── Agent state ────────────────────────────────────────────────────
class AgentState(TypedDict):
    df_summary:  str
    schema:      dict
    eda_results: dict
    tools_run:   list[str]
    insights:    list[str]
    narrative:   str
    next_tool:   str


AVAILABLE_TOOLS = [
    "descriptive_stats",
    "missing_analysis",
    "outlier_detection",
    "correlation_analysis",
    "anomaly_flags",
    "generate_narrative",
]


# ── Tool nodes ─────────────────────────────────────────────────────
def tool_descriptive_stats(state: AgentState) -> AgentState:
    desc = state["eda_results"].get("descriptive", [])
    if desc:
        df = pd.DataFrame(desc)
        state["insights"].append(f"DESCRIPTIVE STATS:\n{df.to_string(index=False)}")
    state["tools_run"].append("descriptive_stats")
    return state


def tool_missing_analysis(state: AgentState) -> AgentState:
    missing = state["eda_results"].get("missing", [])
    if missing:
        df = pd.DataFrame(missing)
        state["insights"].append(f"MISSING VALUES:\n{df.to_string(index=False)}")
    else:
        state["insights"].append("MISSING VALUES: None detected.")
    state["tools_run"].append("missing_analysis")
    return state


def tool_outlier_detection(state: AgentState) -> AgentState:
    outliers = state["eda_results"].get("outliers", {})
    flagged  = {k: v for k, v in outliers.items() if v["iqr_count"] > 0}
    if flagged:
        lines = [
            f"  {c}: {v['iqr_count']} outliers ({v['iqr_pct']}%), severity={v['severity']}"
            for c, v in flagged.items()
        ]
        state["insights"].append("OUTLIERS:\n" + "\n".join(lines))
    else:
        state["insights"].append("OUTLIERS: None detected.")
    state["tools_run"].append("outlier_detection")
    return state


def tool_correlation_analysis(state: AgentState) -> AgentState:
    pairs = state["eda_results"].get("correlation", {}).get("strong_pairs", [])
    if pairs:
        lines = [
            f"  {p['col_a']} ↔ {p['col_b']}: r={p['pearson_r']} ({p['strength']} {p['direction']})"
            for p in pairs[:10]
        ]
        state["insights"].append("CORRELATIONS:\n" + "\n".join(lines))
    else:
        state["insights"].append("CORRELATIONS: No strong correlations found.")
    state["tools_run"].append("correlation_analysis")
    return state


def tool_anomaly_flags(state: AgentState) -> AgentState:
    flags = state["eda_results"].get("flags", [])
    if flags:
        lines = [
            f"  [{f['severity'].upper()}] {f['column']}: {f['issue']} → {f['recommendation']}"
            for f in flags
        ]
        state["insights"].append("ANOMALY FLAGS:\n" + "\n".join(lines))
    state["tools_run"].append("anomaly_flags")
    return state


def tool_generate_narrative(state: AgentState) -> AgentState:
    llm     = _get_llm()
    context = "\n\n".join(state["insights"])

    messages = [
        SystemMessage(content="""You are a senior data scientist.
Given EDA results, write a clear 4-5 paragraph narrative covering:
1. What the dataset likely represents
2. Key patterns and distributions
3. Data quality issues (missing, outliers, skew)
4. Notable correlations
5. Recommended next steps for analysis or modeling
Be specific — cite column names and numbers. Prose only, no bullets."""),
        HumanMessage(
            content=f"Dataset summary:\n{state['df_summary']}\n\nEDA findings:\n{context}"
        ),
    ]

    try:
        response = llm.invoke(messages)
        state["narrative"] = response.content
    except Exception as e:
        state["narrative"] = f"Narrative generation failed: {e}"

    state["tools_run"].append("generate_narrative")
    state["next_tool"] = "END"
    return state


# ── Router ─────────────────────────────────────────────────────────
def agent_router(state: AgentState) -> str:
    if state.get("next_tool") == "END":
        return END

    done      = state["tools_run"]
    remaining = [t for t in AVAILABLE_TOOLS if t not in done]

    if not remaining:
        return "generate_narrative"

    llm = _get_llm()
    messages = [
        SystemMessage(content="""You are an EDA agent controller.
Choose the SINGLE best next tool to run from the list.
Respond with ONLY the tool name, nothing else."""),
        HumanMessage(
            content=(
                f"Dataset: {state['df_summary']}\n"
                f"Tools already run: {done}\n"
                f"Available tools: {remaining}\n"
                f"What should run next?"
            )
        ),
    ]

    try:
        response = llm.invoke(messages)
        choice   = response.content.strip().lower().replace(" ", "_")
        if choice not in remaining:
            choice = remaining[0]
    except Exception:
        choice = remaining[0]

    return choice


# ── Build graph ────────────────────────────────────────────────────
def _build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("descriptive_stats",    tool_descriptive_stats)
    graph.add_node("missing_analysis",     tool_missing_analysis)
    graph.add_node("outlier_detection",    tool_outlier_detection)
    graph.add_node("correlation_analysis", tool_correlation_analysis)
    graph.add_node("anomaly_flags",        tool_anomaly_flags)
    graph.add_node("generate_narrative",   tool_generate_narrative)

    edges = {t: t for t in AVAILABLE_TOOLS} | {END: END}
    for tool in AVAILABLE_TOOLS:
        graph.add_conditional_edges(tool, agent_router, edges)

    graph.set_entry_point("descriptive_stats")
    return graph.compile()


# ── Public entry point ─────────────────────────────────────────────
def run_agent(df: pd.DataFrame, schema: dict, eda_results: dict) -> tuple[str, list[str]]:
    """
    Run the full LangGraph EDA agent.
    Returns (narrative, tools_run).
    """
    df_summary = (
        f"Rows: {len(df)} | Cols: {len(df.columns)} | "
        f"Numeric cols: {sum(1 for s in schema.values() if s['role'] in ('numeric','numeric_categorical'))} | "
        f"Categorical cols: {sum(1 for s in schema.values() if s['role'] in ('categorical','binary'))}"
    )

    initial_state: AgentState = {
        "df_summary":  df_summary,
        "schema":      schema,
        "eda_results": eda_results,
        "tools_run":   [],
        "insights":    [],
        "narrative":   "",
        "next_tool":   "",
    }

    try:
        compiled = _build_agent()
        final    = compiled.invoke(initial_state)
        return final["narrative"], final.get("tools_run", [])
    except Exception:
        return f"Agent error:\n\n{traceback.format_exc()}", []