import io
import pandas as pd
import numpy as np
from pathlib import Path


class DataLoader:
    SUPPORTED = [".csv", ".tsv", ".xlsx", ".xls", ".json", ".parquet"]

    def __init__(self):
        self.load_errors = []

    def load(self, source, filename: str = "dataset") -> pd.DataFrame:
        suffix = Path(filename).suffix.lower()
        if suffix not in self.SUPPORTED:
            raise ValueError(f"Unsupported file type: {suffix}. Supported: {self.SUPPORTED}")

        source = self._normalize(source)
        try:
            df = self._read(source, suffix)
        except Exception as e:
            raise RuntimeError(f"Failed to read file: {e}") from e

        if df.empty:
            raise ValueError("Dataset is empty after loading.")

        return self._clean(df)

    def _normalize(self, source):
        if isinstance(source, memoryview):
            return io.BytesIO(source.tobytes())
        if isinstance(source, (bytes, bytearray)):
            return io.BytesIO(source)
        if isinstance(source, str):
            return source
        if isinstance(source, Path):
            return str(source)
        if hasattr(source, "read"):
            return source
        raise TypeError(f"Unsupported input type: {type(source)}")

    def _read(self, source, suffix: str) -> pd.DataFrame:
        readers = {
            ".csv":     lambda s: pd.read_csv(s, encoding_errors="replace"),
            ".tsv":     lambda s: pd.read_csv(s, sep="\t", encoding_errors="replace"),
            ".xlsx":    lambda s: pd.read_excel(s, engine="openpyxl"),
            ".xls":     lambda s: pd.read_excel(s, engine="xlrd"),
            ".json":    lambda s: pd.read_json(s),
            ".parquet": lambda s: pd.read_parquet(s),
        }
        return readers[suffix](source)

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
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
            schema[col] = {
                "dtype":      str(s.dtype),
                "role":       self._infer_role(s, col),
                "n_unique":   int(s.nunique()),
                "null_count": int(s.isna().sum()),
                "null_pct":   round(float(s.isna().mean()), 4),
                "sample":     s.dropna().head(3).tolist(),
            }
        return schema

    def _infer_role(self, s: pd.Series, col: str) -> str:
        col_lower = col.lower()
        n_unique  = s.nunique()
        n_total   = len(s)

        id_keywords = ["id", "uuid", "key", "index", "code"]
        if any(k in col_lower for k in id_keywords) and n_unique / max(n_total, 1) > 0.9:
            return "id"

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

    def summary_records(self, schema: dict) -> list[dict]:
        rows = []
        for col, info in schema.items():
            rows.append({
                "column":   col,
                "dtype":    info["dtype"],
                "role":     info["role"],
                "unique":   info["n_unique"],
                "nulls":    info["null_count"],
                "null_pct": f"{info['null_pct'] * 100:.1f}%",
                "sample":   str(info["sample"])[:60],
            })
        return rows