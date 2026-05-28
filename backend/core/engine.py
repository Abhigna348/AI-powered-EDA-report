import numpy as np
import pandas as pd
from scipy import stats


class EDAEngine:
    def __init__(self, df: pd.DataFrame, schema: dict):
        self.df      = df
        self.schema  = schema
        self.n       = len(df)
        self.numeric_cols = [
            c for c, s in schema.items()
            if s["role"] in ("numeric", "numeric_categorical")
        ]
        self.cat_cols = [
            c for c, s in schema.items()
            if s["role"] in ("categorical", "binary")
        ]

    # ── Descriptive stats ─────────────────────────────────────────
    def descriptive_stats(self) -> dict:
        if not self.numeric_cols:
            return {}
        df_stats = (
            self.df[self.numeric_cols]
            .describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
            .T
        )
        df_stats["skew"]     = self.df[self.numeric_cols].skew()
        df_stats["kurtosis"] = self.df[self.numeric_cols].kurtosis()
        df_stats["cv"]       = df_stats["std"] / df_stats["mean"].replace(0, np.nan)
        return df_stats.round(4).reset_index().rename(columns={"index": "column"}).to_dict(orient="records")

    # ── Missing value analysis ─────────────────────────────────────
    def missing_analysis(self) -> list[dict]:
        missing = self.df.isna().sum()
        missing = missing[missing > 0]
        if missing.empty:
            return []
        result = []
        for col, count in missing.items():
            pct = count / self.n
            if pct > 0.5:
                rec = "Drop column"
            elif pct > 0.2:
                rec = "Impute (median/mode) or flag"
            else:
                rec = "Impute (mean/median/mode)"
            result.append({
                "column":         col,
                "missing":        int(count),
                "pct":            round(float(pct) * 100, 1),
                "recommendation": rec,
            })
        return sorted(result, key=lambda x: x["missing"], reverse=True)

    # ── Outlier detection ──────────────────────────────────────────
    def detect_outliers(self) -> dict:
        results = {}
        for col in self.numeric_cols:
            s = self.df[col].dropna()
            if len(s) < 4:
                continue
            q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
            iqr     = q3 - q1
            iqr_mask = (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)
            z_mask   = np.abs(stats.zscore(s)) > 3

            results[col] = {
                "iqr_count":   int(iqr_mask.sum()),
                "iqr_pct":     round(float(iqr_mask.mean()) * 100, 2),
                "zscore_count": int(z_mask.sum()),
                "zscore_pct":  round(float(z_mask.mean()) * 100, 2),
                "lower_fence": round(q1 - 1.5 * iqr, 4),
                "upper_fence": round(q3 + 1.5 * iqr, 4),
                "severity":    (
                    "high"   if float(iqr_mask.mean()) > 0.1  else
                    "medium" if float(iqr_mask.mean()) > 0.02 else "low"
                ),
            }
        return results

    # ── Correlation analysis ───────────────────────────────────────
    def correlation_analysis(self) -> dict:
        if len(self.numeric_cols) < 2:
            return {"matrix": {}, "strong_pairs": []}

        corr = self.df[self.numeric_cols].corr(method="pearson").round(4)
        strong_pairs = []

        for i, col_a in enumerate(self.numeric_cols):
            for col_b in self.numeric_cols[i + 1:]:
                val = float(corr.loc[col_a, col_b])
                if abs(val) >= 0.5:
                    strong_pairs.append({
                        "col_a":     col_a,
                        "col_b":     col_b,
                        "pearson_r": round(val, 4),
                        "strength":  "strong" if abs(val) >= 0.7 else "moderate",
                        "direction": "positive" if val > 0 else "negative",
                    })

        strong_pairs.sort(key=lambda x: abs(x["pearson_r"]), reverse=True)
        return {
            "matrix":      corr.to_dict(),
            "strong_pairs": strong_pairs,
        }

    # ── Anomaly flags ──────────────────────────────────────────────
    def anomaly_flags(self, outliers: dict) -> list[dict]:
        flags = []

        for row in self.missing_analysis():
            pct = row["pct"]
            sev = "high" if pct > 50 else "medium" if pct > 20 else "low"
            flags.append({
                "severity":       sev,
                "column":         row["column"],
                "issue":          f"{pct:.1f}% missing values",
                "recommendation": row["recommendation"],
            })

        for col, info in outliers.items():
            if info["iqr_count"] > 0:
                flags.append({
                    "severity":       info["severity"],
                    "column":         col,
                    "issue":          f"{info['iqr_count']} IQR outliers ({info['iqr_pct']}%)",
                    "recommendation": "Investigate; cap, remove, or transform",
                })

        for col in self.numeric_cols:
            sk = float(self.df[col].skew())
            if abs(sk) > 2:
                flags.append({
                    "severity":       "medium",
                    "column":         col,
                    "issue":          f"High skew ({sk:.2f})",
                    "recommendation": "Log or Box-Cox transform",
                })

        for col in self.numeric_cols:
            if float(self.df[col].std()) == 0:
                flags.append({
                    "severity":       "high",
                    "column":         col,
                    "issue":          "Zero variance (constant column)",
                    "recommendation": "Drop — no predictive value",
                })

        for col in self.cat_cols:
            n_unique = self.df[col].nunique()
            if n_unique / self.n > 0.9:
                flags.append({
                    "severity":       "low",
                    "column":         col,
                    "issue":          f"Very high cardinality ({n_unique} unique)",
                    "recommendation": "Likely an ID column — exclude from modeling",
                })

        order = ["high", "medium", "low"]
        return sorted(flags, key=lambda x: order.index(x["severity"]))

    # ── Run everything ─────────────────────────────────────────────
    def run_all(self) -> dict:
        outliers = self.detect_outliers()
        corr     = self.correlation_analysis()
        return {
            "descriptive": self.descriptive_stats(),
            "missing":     self.missing_analysis(),
            "outliers":    outliers,
            "correlation": corr,
            "flags":       self.anomaly_flags(outliers),
        }