import pandas as pd


class ReportGenerator:
    SEV_ICONS = {"high": "🔴", "medium": "🟡", "low": "⚪"}

    def generate(
        self,
        filename:    str,
        df:          pd.DataFrame,
        schema:      dict,
        eda_results: dict,
        narrative:   str,
    ) -> str:
        flags   = eda_results.get("flags", [])
        desc    = eda_results.get("descriptive", [])
        missing = eda_results.get("missing", [])
        pairs   = eda_results.get("correlation", {}).get("strong_pairs", [])

        flag_md = "\n".join(
            f"- {self.SEV_ICONS.get(f['severity'], '')} **{f['column']}**: "
            f"{f['issue']} — *{f['recommendation']}*"
            for f in flags
        ) or "_No anomalies detected._"

        corr_md = "\n".join(
            f"- **{p['col_a']}** ↔ **{p['col_b']}**: "
            f"r = {p['pearson_r']} ({p['strength']} {p['direction']})"
            for p in pairs[:10]
        ) or "_No strong correlations found._"

        # Descriptive stats table
        if desc:
            df_desc = pd.DataFrame(desc)
            try:
                desc_md = df_desc.to_markdown(index=False)
            except Exception:
                desc_md = df_desc.to_string(index=False)
        else:
            desc_md = "_No numeric columns._"

        # Missing values table
        if missing:
            df_miss = pd.DataFrame(missing)
            try:
                missing_md = df_miss.to_markdown(index=False)
            except Exception:
                missing_md = df_miss.to_string(index=False)
        else:
            missing_md = "_No missing values._"

        total_missing = df.isna().sum().sum()
        n_numeric     = sum(1 for s in schema.values() if s["role"] in ("numeric", "numeric_categorical"))
        n_categorical = sum(1 for s in schema.values() if s["role"] in ("categorical", "binary"))

        schema_rows = "\n".join(
            f"| {col} | {info['dtype']} | {info['role']} | "
            f"{info['n_unique']} | {info['null_count']} |"
            for col, info in schema.items()
        )

        report = f"""# EDA Report — {filename}

---

## Overview
| | |
|---|---|
| **Rows** | {len(df):,} |
| **Columns** | {len(df.columns)} |
| **Numeric columns** | {n_numeric} |
| **Categorical columns** | {n_categorical} |
| **Total missing cells** | {total_missing:,} |
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
{schema_rows}
"""
        return report