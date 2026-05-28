#!/usr/bin/env python
# coding: utf-8

# # Automated Exploratory Data Analysis (EDA) Agent
# 
# ## Project Overview
# 
# The objective of this project is to build an automated Exploratory Data Analysis (EDA) agent capable of analyzing datasets with minimal human intervention. The system automatically inspects data quality, detects patterns, generates statistical summaries, visualizes relationships, and produces actionable insights.
# 
# This agent is designed to streamline the initial stages of the data analysis workflow by reducing manual effort and improving analytical efficiency.
# 
# ## Build order:
# 
# Setup & installs
# Data loader + schema detection
# EDA engine (stats, outliers, correlations)
# Visualizations (matplotlib + plotly)
# LangGraph agent (tool-choosing loop)
# NL Q&A chat interface
# Report generator

# # 1. Installs

# In[1]:


get_ipython().system('pip install -q pandas numpy scipy scikit-learn matplotlib plotly               langgraph langchain langchain-openai openai               ipywidgets ydata-profiling kaleido')


# ## 2.Imports & config

# In[16]:


import os
import io
import json
import warnings
import traceback
from typing import TypedDict, Annotated, Any
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.preprocessing import LabelEncoder

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from IPython.display import display, Markdown, HTML, clear_output
import ipywidgets as widgets

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", "{:.4f}".format)

# --- Config ---
# Colab: from google.colab import userdata; OPENAI_API_KEY = userdata.get("OPENAI_API_KEY")
# Jupyter: set directly or load from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-proj-")

llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)

print("✅ Setup complete")


# ## 3..Data loader + schema detection

# In[17]:


import io
import pandas as pd
import numpy as np
from pathlib import Path

class DataLoader:
    SUPPORTED = [".csv", ".tsv", ".xlsx", ".xls", ".json", ".parquet"]

    def __init__(self):
        self.df = None
        self.load_errors = []

    def load(self, source, filename="dataset"):
        suffix = Path(filename).suffix.lower()

        if suffix not in self.SUPPORTED:
            raise ValueError(f"Unsupported file type: {suffix}")

        try:
            source = self._normalize(source)
            df = self._read(source, suffix)
        except Exception as e:
            raise RuntimeError(f"Failed to read file: {e}") from e

        if df.empty:
            raise ValueError("Dataset is empty")

        return self._clean(df)

    # 🔥 FIX: normalize EVERYTHING here
    def _normalize(self, source):
        if isinstance(source, memoryview):
            return io.BytesIO(source.tobytes())

        if isinstance(source, (bytes, bytearray)):
            return io.BytesIO(source)

        if isinstance(source, str):
            return source  # file path

        if isinstance(source, Path):
            return str(source)

        if hasattr(source, "read"):
            return source  # file-like

        raise TypeError(f"Unsupported input type: {type(source)}")

    def _read(self, source, suffix):
        readers = {
            ".csv": lambda s: pd.read_csv(s, encoding_errors="replace"),
            ".tsv": lambda s: pd.read_csv(s, sep="\t", encoding_errors="replace"),
            ".xlsx": lambda s: pd.read_excel(s, engine="openpyxl"),
            ".xls": lambda s: pd.read_excel(s, engine="xlrd"),
            ".json": lambda s: pd.read_json(s),
            ".parquet": lambda s: pd.read_parquet(s),
        }
        return readers[suffix](source)

    def _clean(self, df):
        df.columns = (
            df.columns.astype(str)
            .str.lower()
            .str.strip()
            .str.replace(r"[^\w]", "_", regex=True)
        )

        df = df.dropna(how="all").dropna(axis=1, how="all")

        return df.reset_index(drop=True)


class SchemaDetector:
    def detect(self, df: pd.DataFrame) -> dict:
        schema = {}
        for col in df.columns:
            s = df[col]
            dtype = str(s.dtype)
            role = self._infer_role(s, col)
            schema[col] = {
                "dtype": dtype,
                "role": role,
                "n_unique": int(s.nunique()),
                "null_count": int(s.isna().sum()),
                "null_pct": round(s.isna().mean(), 4),
                "sample": s.dropna().head(3).tolist(),
            }
        return schema

    def _infer_role(self, s: pd.Series, col: str) -> str:
        col_lower = col.lower()
        n_unique = s.nunique()
        n_total = len(s)

        # ID detection
        id_keywords = ["id", "uuid", "key", "index", "code"]
        if any(k in col_lower for k in id_keywords) and n_unique / max(n_total, 1) > 0.9:
            return "id"

        # Datetime detection
        if pd.api.types.is_datetime64_any_dtype(s):
            return "datetime"
        if s.dtype == object:
            sample = s.dropna().head(50)
            try:
                pd.to_datetime(sample, infer_datetime_format=True)
                return "datetime"
            except Exception:
                pass

        if pd.api.types.is_numeric_dtype(s):
            if n_unique <= 2:
                return "binary"
            if n_unique <= 20 or n_unique / max(n_total, 1) < 0.05:
                return "numeric_categorical"
            return "numeric"

        if n_unique / max(n_total, 1) > 0.9:
            return "text"
        return "categorical"

    def summary(self, schema: dict) -> pd.DataFrame:
        rows = []
        for col, info in schema.items():
            rows.append({
                "column": col,
                "dtype": info["dtype"],
                "role": info["role"],
                "unique": info["n_unique"],
                "nulls": info["null_count"],
                "null_%": f"{info['null_pct']*100:.1f}%",
                "sample": str(info["sample"])[:60],
            })
        return pd.DataFrame(rows)


loader = DataLoader()
detector = SchemaDetector()
print("✅ Data loader and schema detector ready")


# ## 4.EDA engine

# In[18]:


class EDAEngine:
    def __init__(self, df: pd.DataFrame, schema: dict):
        self.df = df
        self.schema = schema
        self.n = len(df)
        self.numeric_cols = [c for c, s in schema.items() if s["role"] in ("numeric", "numeric_categorical")]
        self.cat_cols = [c for c, s in schema.items() if s["role"] in ("categorical", "binary")]

    # ── Descriptive stats ──────────────────────────────────────────
    def descriptive_stats(self) -> pd.DataFrame:
        if not self.numeric_cols:
            return pd.DataFrame()
        stats_df = self.df[self.numeric_cols].describe(percentiles=[.05, .25, .5, .75, .95]).T
        stats_df["skew"] = self.df[self.numeric_cols].skew()
        stats_df["kurtosis"] = self.df[self.numeric_cols].kurtosis()
        stats_df["cv"] = stats_df["std"] / stats_df["mean"].replace(0, np.nan)
        return stats_df.round(4)

    # ── Missing value analysis ─────────────────────────────────────
    def missing_analysis(self) -> pd.DataFrame:
        missing = self.df.isna().sum()
        missing = missing[missing > 0]
        if missing.empty:
            return pd.DataFrame(columns=["column", "missing", "pct", "recommendation"])
        result = []
        for col, count in missing.items():
            pct = count / self.n
            if pct > 0.5:
                rec = "Drop column"
            elif pct > 0.2:
                rec = "Impute (median/mode) or flag"
            else:
                rec = "Impute (mean/median/mode)"
            result.append({"column": col, "missing": count, "pct": f"{pct*100:.1f}%", "recommendation": rec})
        return pd.DataFrame(result).sort_values("missing", ascending=False)

    # ── Outlier detection ─────────────────────────────────────────
    def detect_outliers(self) -> dict:
        results = {}
        for col in self.numeric_cols:
            s = self.df[col].dropna()
            if len(s) < 4:
                continue

            # IQR method
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            iqr_mask = (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)

            # Z-score method
            z_scores = np.abs(stats.zscore(s))
            z_mask = z_scores > 3

            results[col] = {
                "iqr_count": int(iqr_mask.sum()),
                "iqr_pct": round(iqr_mask.mean() * 100, 2),
                "zscore_count": int(z_mask.sum()),
                "zscore_pct": round(z_mask.mean() * 100, 2),
                "lower_fence": round(q1 - 1.5 * iqr, 4),
                "upper_fence": round(q3 + 1.5 * iqr, 4),
                "severity": "high" if iqr_mask.mean() > 0.1 else
                            "medium" if iqr_mask.mean() > 0.02 else "low",
            }
        return results

    # ── Correlation analysis ───────────────────────────────────────
    def correlation_analysis(self) -> dict:
        if len(self.numeric_cols) < 2:
            return {"matrix": pd.DataFrame(), "strong_pairs": []}

        corr = self.df[self.numeric_cols].corr(method="pearson")
        strong_pairs = []

        for i, col_a in enumerate(self.numeric_cols):
            for col_b in self.numeric_cols[i+1:]:
                val = corr.loc[col_a, col_b]
                if abs(val) >= 0.5:
                    strong_pairs.append({
                        "col_a": col_a,
                        "col_b": col_b,
                        "pearson_r": round(val, 4),
                        "strength": "strong" if abs(val) >= 0.7 else "moderate",
                        "direction": "positive" if val > 0 else "negative",
                    })

        strong_pairs.sort(key=lambda x: abs(x["pearson_r"]), reverse=True)
        return {"matrix": corr.round(4), "strong_pairs": strong_pairs}

    # ── Anomaly flags ─────────────────────────────────────────────
    def anomaly_flags(self, outliers: dict) -> list[dict]:
        flags = []

        # Missing values
        for _, row in self.missing_analysis().iterrows():
            pct = float(row["pct"].replace("%", ""))
            sev = "high" if pct > 50 else "medium" if pct > 20 else "low"
            flags.append({"severity": sev, "column": row["column"],
                          "issue": f"{pct:.1f}% missing values", "recommendation": row["recommendation"]})

        # Outliers
        for col, info in outliers.items():
            if info["iqr_count"] > 0:
                flags.append({"severity": info["severity"], "column": col,
                               "issue": f"{info['iqr_count']} IQR outliers ({info['iqr_pct']}%)",
                               "recommendation": "Investigate; cap, remove, or transform"})

        # Skew
        for col in self.numeric_cols:
            sk = self.df[col].skew()
            if abs(sk) > 2:
                flags.append({"severity": "medium", "column": col,
                               "issue": f"High skew ({sk:.2f})",
                               "recommendation": "Log or Box-Cox transform"})

        # Zero variance
        for col in self.numeric_cols:
            if self.df[col].std() == 0:
                flags.append({"severity": "high", "column": col,
                               "issue": "Zero variance (constant column)",
                               "recommendation": "Drop — no predictive value"})

        # High cardinality categoricals
        for col in self.cat_cols:
            n_unique = self.df[col].nunique()
            if n_unique / self.n > 0.9:
                flags.append({"severity": "low", "column": col,
                               "issue": f"Very high cardinality ({n_unique} unique)",
                               "recommendation": "Likely an ID column — exclude from modeling"})

        return sorted(flags, key=lambda x: ["high","medium","low"].index(x["severity"]))

    def run_all(self) -> dict:
        outliers = self.detect_outliers()
        corr = self.correlation_analysis()
        return {
            "descriptive": self.descriptive_stats(),
            "missing": self.missing_analysis(),
            "outliers": outliers,
            "correlation": corr,
            "flags": self.anomaly_flags(outliers),
        }


print("✅ EDA engine ready")


# ## 5.Visualizer

# In[19]:


class Visualizer:
    PALETTE = px.colors.qualitative.Set2

    def __init__(self, df: pd.DataFrame, schema: dict):
        self.df = df
        self.schema = schema
        self.numeric_cols = [c for c, s in schema.items() if s["role"] in ("numeric", "numeric_categorical")]
        self.cat_cols = [c for c, s in schema.items() if s["role"] in ("categorical", "binary")]

    def plot_distributions(self):
        if not self.numeric_cols:
            print("No numeric columns to plot.")
            return
        cols = self.numeric_cols[:8]  # cap at 8
        n = len(cols)
        fig = make_subplots(rows=2, cols=n, subplot_titles=[f"{c} — hist" for c in cols] +
                            [f"{c} — box" for c in cols])
        for i, col in enumerate(cols, 1):
            s = self.df[col].dropna()
            fig.add_trace(go.Histogram(x=s, name=col, showlegend=False,
                          marker_color=self.PALETTE[i % len(self.PALETTE)]), row=1, col=i)
            fig.add_trace(go.Box(y=s, name=col, showlegend=False,
                          marker_color=self.PALETTE[i % len(self.PALETTE)]), row=2, col=i)
        fig.update_layout(title="Numeric distributions", height=500, template="plotly_white")
        fig.show()

    def plot_correlation_heatmap(self, corr_matrix: pd.DataFrame):
        if corr_matrix.empty:
            print("Not enough numeric columns for correlation.")
            return
        fig = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1, title="Correlation matrix", aspect="auto")
        fig.update_layout(template="plotly_white", height=500)
        fig.show()

    def plot_categoricals(self):
        if not self.cat_cols:
            return
        cols = self.cat_cols[:4]
        fig = make_subplots(rows=1, cols=len(cols), subplot_titles=cols)
        for i, col in enumerate(cols, 1):
            freq = self.df[col].value_counts().head(10)
            fig.add_trace(go.Bar(x=freq.index.astype(str), y=freq.values,
                          name=col, showlegend=False,
                          marker_color=self.PALETTE[i % len(self.PALETTE)]), row=1, col=i)
        fig.update_layout(title="Categorical frequencies", height=400, template="plotly_white")
        fig.show()

    def plot_missing(self, missing_df: pd.DataFrame):
        if missing_df.empty:
            display(Markdown("✅ **No missing values found.**"))
            return
        fig = px.bar(missing_df, x="column", y="missing",
                     text="pct", title="Missing values per column",
                     color="missing", color_continuous_scale="Reds")
        fig.update_layout(template="plotly_white", height=350)
        fig.show()

    def plot_outlier_summary(self, outliers: dict):
        if not outliers:
            return
        rows = [{"column": c, "iqr_outliers": v["iqr_count"],
                 "pct": v["iqr_pct"], "severity": v["severity"]}
                for c, v in outliers.items() if v["iqr_count"] > 0]
        if not rows:
            display(Markdown("✅ **No outliers detected.**"))
            return
        df_o = pd.DataFrame(rows).sort_values("iqr_outliers", ascending=False)
        color_map = {"high": "#D85A30", "medium": "#BA7517", "low": "#5F9E5F"}
        fig = px.bar(df_o, x="column", y="iqr_outliers", color="severity",
                     text="pct", title="Outlier counts by column (IQR method)",
                     color_discrete_map=color_map)
        fig.update_layout(template="plotly_white", height=350)
        fig.show()

    def run_all(self, eda_results: dict):
        self.plot_distributions()
        self.plot_categoricals()
        self.plot_missing(eda_results["missing"])
        self.plot_outlier_summary(eda_results["outliers"])
        self.plot_correlation_heatmap(eda_results["correlation"]["matrix"])


print("✅ Visualizer ready")


# ## 5.Agent

# In[20]:


# ── Agent state ────────────────────────────────────────────────────
class AgentState(TypedDict):
    df_summary: str
    schema: dict
    eda_results: dict
    tools_run: list[str]
    insights: list[str]
    narrative: str
    next_tool: str

# ── Tool functions the agent can call ─────────────────────────────
def tool_descriptive_stats(state: AgentState) -> AgentState:
    desc = state["eda_results"].get("descriptive", pd.DataFrame())
    if not desc.empty:
        state["insights"].append(f"DESCRIPTIVE STATS:\n{desc.to_string()}")
    state["tools_run"].append("descriptive_stats")
    return state

def tool_missing_analysis(state: AgentState) -> AgentState:
    missing = state["eda_results"].get("missing", pd.DataFrame())
    if not missing.empty:
        state["insights"].append(f"MISSING VALUES:\n{missing.to_string(index=False)}")
    else:
        state["insights"].append("MISSING VALUES: None detected.")
    state["tools_run"].append("missing_analysis")
    return state

def tool_outlier_detection(state: AgentState) -> AgentState:
    outliers = state["eda_results"].get("outliers", {})
    flagged = {k: v for k, v in outliers.items() if v["iqr_count"] > 0}
    if flagged:
        lines = [f"  {c}: {v['iqr_count']} outliers ({v['iqr_pct']}%), severity={v['severity']}"
                 for c, v in flagged.items()]
        state["insights"].append("OUTLIERS:\n" + "\n".join(lines))
    else:
        state["insights"].append("OUTLIERS: None detected.")
    state["tools_run"].append("outlier_detection")
    return state

def tool_correlation_analysis(state: AgentState) -> AgentState:
    pairs = state["eda_results"].get("correlation", {}).get("strong_pairs", [])
    if pairs:
        lines = [f"  {p['col_a']} ↔ {p['col_b']}: r={p['pearson_r']} ({p['strength']} {p['direction']})"
                 for p in pairs[:10]]
        state["insights"].append("CORRELATIONS:\n" + "\n".join(lines))
    else:
        state["insights"].append("CORRELATIONS: No strong correlations found.")
    state["tools_run"].append("correlation_analysis")
    return state

def tool_anomaly_flags(state: AgentState) -> AgentState:
    flags = state["eda_results"].get("flags", [])
    if flags:
        lines = [f"  [{f['severity'].upper()}] {f['column']}: {f['issue']} → {f['recommendation']}"
                 for f in flags]
        state["insights"].append("ANOMALY FLAGS:\n" + "\n".join(lines))
    state["tools_run"].append("anomaly_flags")
    return state

def tool_generate_narrative(state: AgentState) -> AgentState:
    """Call GPT-4 to synthesize all insights into a narrative."""
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
        HumanMessage(content=f"Dataset summary:\n{state['df_summary']}\n\nEDA findings:\n{context}")
    ]
    try:
        response = llm.invoke(messages)
        state["narrative"] = response.content
    except Exception as e:
        state["narrative"] = f"Narrative generation failed: {e}"
    state["tools_run"].append("generate_narrative")
    state["next_tool"] = "END"
    return state

# ── Router: GPT-4 decides which tool to run next ──────────────────
AVAILABLE_TOOLS = [
    "descriptive_stats",
    "missing_analysis",
    "outlier_detection",
    "correlation_analysis",
    "anomaly_flags",
    "generate_narrative",
]

def agent_router(state: AgentState) -> str:
    """Ask GPT-4 which tool to run next given what's already been done."""
    if state["next_tool"] == "END":
        return END

    done = state["tools_run"]
    remaining = [t for t in AVAILABLE_TOOLS if t not in done]

    if not remaining:
        return "generate_narrative"

    messages = [
        SystemMessage(content="""You are an EDA agent controller.
Choose the SINGLE best next tool to run from the list.
Respond with ONLY the tool name, nothing else."""),
        HumanMessage(content=f"""Dataset: {state['df_summary']}
Tools already run: {done}
Available tools: {remaining}
What should run next?""")
    ]
    try:
        response = llm.invoke(messages)
        choice = response.content.strip().lower().replace(" ", "_")
        if choice not in remaining:
            choice = remaining[0]
    except Exception:
        choice = remaining[0]

    return choice

# ── Build the graph ────────────────────────────────────────────────
def build_agent() -> Any:
    graph = StateGraph(AgentState)

    graph.add_node("descriptive_stats", tool_descriptive_stats)
    graph.add_node("missing_analysis", tool_missing_analysis)
    graph.add_node("outlier_detection", tool_outlier_detection)
    graph.add_node("correlation_analysis", tool_correlation_analysis)
    graph.add_node("anomaly_flags", tool_anomaly_flags)
    graph.add_node("generate_narrative", tool_generate_narrative)

    for tool in AVAILABLE_TOOLS:
        graph.add_conditional_edges(tool, agent_router, {
            t: t for t in AVAILABLE_TOOLS
        } | {END: END})

    graph.set_entry_point("descriptive_stats")
    return graph.compile()


agent = build_agent()
print("✅ LangGraph agent ready")


# ## 6.NL&QA

# In[21]:


class EDAChat:
    """Natural language Q&A about the dataset after analysis."""

    def __init__(self, df: pd.DataFrame, schema: dict, eda_results: dict, narrative: str):
        self.df = df
        self.schema = schema
        self.eda_results = eda_results
        self.narrative = narrative
        self.history: list[dict] = []
        self._build_context()

    def _build_context(self):
        desc = self.eda_results.get("descriptive", pd.DataFrame())
        flags = self.eda_results.get("flags", [])
        pairs = self.eda_results.get("correlation", {}).get("strong_pairs", [])

        self.context = f"""
DATASET: {len(self.df)} rows × {len(self.df.columns)} columns
COLUMNS: {', '.join(self.df.columns.tolist())}

DESCRIPTIVE STATS:
{desc.to_string() if not desc.empty else 'No numeric columns.'}

ANOMALY FLAGS:
{json.dumps(flags[:20], indent=2)}

STRONG CORRELATIONS:
{json.dumps(pairs[:10], indent=2)}

NARRATIVE SUMMARY:
{self.narrative}
""".strip()

    def ask(self, question: str) -> str:
        if not question.strip():
            return "Please enter a question."

        # Build message history for multi-turn
        messages = [
            SystemMessage(content=f"""You are a data analyst assistant.
Answer questions about this dataset accurately and concisely.
Use specific numbers from the EDA results.
If asked to compute something not in the results, say so clearly.

DATASET CONTEXT:
{self.context}""")
        ]

        # Include last 6 turns for context window efficiency
        for turn in self.history[-6:]:
            messages.append(HumanMessage(content=turn["q"]))
            messages.append(SystemMessage(content=turn["a"]))

        messages.append(HumanMessage(content=question))

        try:
            response = llm.invoke(messages)
            answer = response.content
        except Exception as e:
            answer = f"Error: {e}"

        self.history.append({"q": question, "a": answer})
        return answer

    def interactive(self):
        """Render a chat widget in Jupyter/Colab."""
        display(Markdown("### 💬 Ask questions about your data"))
        display(Markdown("*Type a question and press Enter or click Ask*"))

        out = widgets.Output()
        text = widgets.Text(
            placeholder="e.g. Which column has the most outliers?",
            layout=widgets.Layout(width="70%")
        )
        btn = widgets.Button(description="Ask", button_style="primary")
        clear_btn = widgets.Button(description="Clear", button_style="warning")

        def on_ask(_):
            q = text.value.strip()
            if not q:
                return
            text.value = ""
            with out:
                display(Markdown(f"**You:** {q}"))
                display(Markdown("*Thinking…*"))
                clear_output(wait=True)
                display(Markdown(f"**You:** {q}"))
                answer = self.ask(q)
                display(Markdown(f"**Agent:** {answer}"))
                display(Markdown("---"))

        def on_clear(_):
            self.history.clear()
            with out:
                clear_output()

        btn.on_click(on_ask)
        clear_btn.on_click(on_clear)
        text.on_submit(on_ask)

        display(widgets.HBox([text, btn, clear_btn]))
        display(out)


print("✅ NL Q&A chat ready")


# ## 7.Report Generator

# In[24]:


class ReportGenerator:
    def generate(self, filename: str, df: pd.DataFrame, schema: dict,
                 eda_results: dict, narrative: str) -> str:

        flags = eda_results.get("flags", [])
        desc = eda_results.get("descriptive", pd.DataFrame())
        missing = eda_results.get("missing", pd.DataFrame())
        pairs = eda_results.get("correlation", {}).get("strong_pairs", [])

        sev_icons = {"high": "🔴", "medium": "🟡", "low": "⚪"}

        flag_md = "\n".join(
            f"- {sev_icons.get(f['severity'], '')} **{f['column']}**: {f['issue']} — *{f['recommendation']}*"
            for f in flags
        ) or "_No anomalies detected._"

        corr_md = "\n".join(
            f"- **{p['col_a']}** ↔ **{p['col_b']}**: r = {p['pearson_r']} ({p['strength']} {p['direction']})"
            for p in pairs[:10]
        ) or "_No strong correlations found._"

        # ✅ FIX APPLIED HERE (NO to_markdown dependency)
        if not desc.empty:
            try:
                desc_md = desc.to_markdown()
            except Exception:
                desc_md = desc.to_string()
        else:
            desc_md = "_No numeric columns._"

        if not missing.empty:
            try:
                missing_md = missing.to_markdown(index=False)
            except Exception:
                missing_md = missing.to_string(index=False)
        else:
            missing_md = "_No missing values._"

        report = f"""# EDA Report — {filename}

---

## Overview
| | |
|---|---|
| **Rows** | {len(df):,} |
| **Columns** | {len(df.columns)} |
| **Numeric columns** | {sum(1 for s in schema.values() if s['role'] in ('numeric','numeric_categorical'))} |
| **Categorical columns** | {sum(1 for s in schema.values() if s['role'] in ('categorical','binary'))} |
| **Total missing cells** | {df.isna().sum().sum():,} |
| **Anomalies flagged** | {len(flags)} |

---

## AI Narrative
{narrative}

---

## Anomalies & Data Quality
{flag_md}

---

## Descriptive Statistics
{desc_md}

---

## Missing Values
{missing_md}

---

## Strong Correlations
{corr_md}

---

## Schema
| Column | Type | Role | Unique | Nulls |
|--------|------|------|--------|-------|
""" + "\n".join(
    f"| {col} | {info['dtype']} | {info['role']} | {info['n_unique']} | {info['null_count']} |"
    for col, info in schema.items()
)

        return report

    def save(self, report: str, path: str = "eda_report.md"):
        with open(path, "w") as f:
            f.write(report)
        print(f"✅ Report saved to {path}")
        return path


print("✅ Report generator ready")


# ## Master Runner

# In[25]:


def run_eda_agent(source, filename: str = "dataset.csv"):
    display(Markdown(f"# 🔍 EDA Agent — `{filename}`"))

    # 1. Load
    display(Markdown("## 1. Loading data…"))
    df = loader.load(source, filename)

    if loader.load_errors:
        for e in loader.load_errors:
            display(Markdown(f"> ⚠️ {e}"))

    display(Markdown(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns"))

    # 2. Schema
    display(Markdown("## 2. Schema detection"))
    schema = detector.detect(df)
    display(detector.summary(schema))

    # 3. EDA
    display(Markdown("## 3. Running EDA engine…"))
    engine = EDAEngine(df, schema)
    eda_results = engine.run_all()

    display(Markdown("### Descriptive statistics"))
    display(eda_results["descriptive"])

    display(Markdown("### Missing values"))
    if eda_results["missing"].empty:
        display(Markdown("✅ No missing values."))
    else:
        display(eda_results["missing"])

    # 4. Visualizations
    display(Markdown("## 4. Visualizations"))
    viz = Visualizer(df, schema)
    viz.run_all(eda_results)

    # 5. LangGraph agent
    display(Markdown("## 5. Running LangGraph agent…"))

    df_summary = (
        f"Filename: {filename} | Rows: {len(df)} | Cols: {len(df.columns)} | "
        f"Numeric: {len(engine.numeric_cols)} | Categorical: {len(engine.cat_cols)}"
    )

    initial_state: AgentState = {
        "df_summary": df_summary,
        "schema": schema,
        "eda_results": eda_results,
        "tools_run": [],
        "insights": [],
        "narrative": "",
        "next_tool": "",
    }

    try:
        final_state = agent.invoke(initial_state)
        narrative = final_state["narrative"]

        display(
            Markdown(
                f"**Tools run by agent:** {' → '.join(final_state.get('tools_run', []))}"
            )
        )

    except Exception:
        narrative = f"Agent error:\n\n{traceback.format_exc()}"
        display(Markdown("⚠️ Agent failed"))

    # 6. Narrative
    display(Markdown("## 6. AI Narrative"))
    display(Markdown(narrative))

    # 7. Report
    display(Markdown("## 7. Generating report…"))

    reporter = ReportGenerator()
    report = reporter.generate(filename, df, schema, eda_results, narrative)

    out_path = f"eda_{Path(filename).stem}.md"
    reporter.save(report, out_path)

    display(Markdown(f"✅ Report saved: `{out_path}`"))

    # 8. Q&A
    display(Markdown("## 8. Ask questions about your data"))

    chat = EDAChat(df, schema, eda_results, narrative)
    chat.interactive()

    return {
        "df": df,
        "schema": schema,
        "eda_results": eda_results,
        "narrative": narrative,
        "chat": chat,
    }


# ─────────────────────────────────────────────
# File upload (FIXED VERSION)
# ─────────────────────────────────────────────

import ipywidgets as widgets
from IPython.display import display

upload = widgets.FileUpload(accept=".csv", multiple=False)
display(upload)


def on_upload(change):
    uploaded = upload.value

    if not uploaded:
        return

    file_obj = uploaded[0]

    fname = file_obj["name"]

    # 🔥 FIX: ALWAYS convert memoryview → bytes
    content = file_obj["content"]

    if isinstance(content, memoryview):
        content = content.tobytes()

    run_eda_agent(content, filename=fname)


upload.observe(on_upload, names="value")


# In[ ]:




