import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


PALETTE = px.colors.qualitative.Set2


class Visualizer:
    def __init__(self, df: pd.DataFrame, schema: dict):
        self.df  = df
        self.schema = schema
        self.numeric_cols = [
            c for c, s in schema.items()
            if s["role"] in ("numeric", "numeric_categorical")
        ]
        self.cat_cols = [
            c for c, s in schema.items()
            if s["role"] in ("categorical", "binary")
        ]

    def _fig_to_dict(self, fig: go.Figure) -> dict:
        """Convert a Plotly figure to a JSON-serialisable dict for the API."""
        return fig.to_dict()

    # ── Distributions (histograms + box plots) ─────────────────────
    def distributions(self) -> dict | None:
        if not self.numeric_cols:
            return None

        cols = self.numeric_cols[:8]
        n    = len(cols)

        fig = make_subplots(
            rows=2, cols=n,
            subplot_titles=(
                [f"{c} — dist" for c in cols] +
                [f"{c} — box"  for c in cols]
            ),
        )

        for i, col in enumerate(cols, 1):
            s     = self.df[col].dropna()
            color = PALETTE[i % len(PALETTE)]
            fig.add_trace(
                go.Histogram(x=s, name=col, showlegend=False, marker_color=color),
                row=1, col=i,
            )
            fig.add_trace(
                go.Box(y=s, name=col, showlegend=False, marker_color=color),
                row=2, col=i,
            )

        fig.update_layout(
            title="Numeric distributions",
            height=520,
            template="plotly_white",
            margin=dict(t=60, b=40),
        )
        return self._fig_to_dict(fig)

    # ── Correlation heatmap ────────────────────────────────────────
    def correlation_heatmap(self, corr_matrix: dict) -> dict | None:
        if not corr_matrix:
            return None

        df_corr = pd.DataFrame(corr_matrix)
        if df_corr.empty:
            return None

        fig = px.imshow(
            df_corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title="Correlation matrix",
            aspect="auto",
        )
        fig.update_layout(template="plotly_white", height=500)
        return self._fig_to_dict(fig)

    # ── Categorical bar charts ─────────────────────────────────────
    def categoricals(self) -> dict | None:
        if not self.cat_cols:
            return None

        cols = self.cat_cols[:4]
        fig  = make_subplots(rows=1, cols=len(cols), subplot_titles=cols)

        for i, col in enumerate(cols, 1):
            freq  = self.df[col].value_counts().head(10)
            color = PALETTE[i % len(PALETTE)]
            fig.add_trace(
                go.Bar(
                    x=freq.index.astype(str),
                    y=freq.values,
                    name=col,
                    showlegend=False,
                    marker_color=color,
                ),
                row=1, col=i,
            )

        fig.update_layout(
            title="Categorical frequencies",
            height=400,
            template="plotly_white",
        )
        return self._fig_to_dict(fig)

    # ── Missing values bar chart ───────────────────────────────────
    def missing_chart(self, missing: list[dict]) -> dict | None:
        if not missing:
            return None

        df_m = pd.DataFrame(missing)
        fig  = px.bar(
            df_m,
            x="column",
            y="missing",
            text="pct",
            title="Missing values per column",
            color="missing",
            color_continuous_scale="Reds",
        )
        fig.update_layout(template="plotly_white", height=350)
        return self._fig_to_dict(fig)

    # ── Outlier summary bar chart ──────────────────────────────────
    def outlier_chart(self, outliers: dict) -> dict | None:
        rows = [
            {
                "column":   c,
                "iqr_outliers": v["iqr_count"],
                "pct":      v["iqr_pct"],
                "severity": v["severity"],
            }
            for c, v in outliers.items() if v["iqr_count"] > 0
        ]
        if not rows:
            return None

        df_o      = pd.DataFrame(rows).sort_values("iqr_outliers", ascending=False)
        color_map = {"high": "#D85A30", "medium": "#BA7517", "low": "#5F9E5F"}

        fig = px.bar(
            df_o,
            x="column",
            y="iqr_outliers",
            color="severity",
            text="pct",
            title="Outlier counts by column (IQR method)",
            color_discrete_map=color_map,
        )
        fig.update_layout(template="plotly_white", height=350)
        return self._fig_to_dict(fig)

    # ── Run all, return dict of chart dicts ────────────────────────
    def run_all(self, eda_results: dict) -> dict:
        return {
            "distributions":     self.distributions(),
            "categoricals":      self.categoricals(),
            "missing_chart":     self.missing_chart(eda_results.get("missing", [])),
            "outlier_chart":     self.outlier_chart(eda_results.get("outliers", {})),
            "correlation_heatmap": self.correlation_heatmap(
                eda_results.get("correlation", {}).get("matrix", {})
            ),
        }