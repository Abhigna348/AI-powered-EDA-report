import json
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from fastapi.responses import JSONResponse

from core.loader     import DataLoader, SchemaDetector
from core.engine     import EDAEngine
from core.visualizer import Visualizer
from core.agent      import run_agent
from core.report     import ReportGenerator
from routers.auth    import get_current_user

router = APIRouter(tags=["upload"])

loader   = DataLoader()
detector = SchemaDetector()
reporter = ReportGenerator()

# ── POST /api/analyze ─────────────────────────────────────────────
@router.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    user: dict       = Depends(get_current_user),
):
    # ── 1. Validate file type ──────────────────────────────────────
    allowed = {".csv", ".tsv", ".xlsx", ".xls", ".json", ".parquet"}
    suffix  = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if suffix not in allowed:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(allowed)}",
        )

    # ── 2. Load ────────────────────────────────────────────────────
    try:
        content = await file.read()
        df      = loader.load(content, filename=file.filename)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse file: {exc}") from exc

    # ── 3. Schema ──────────────────────────────────────────────────
    schema         = detector.detect(df)
    schema_summary = detector.summary_records(schema)

    # ── 4. EDA engine ──────────────────────────────────────────────
    engine      = EDAEngine(df, schema)
    eda_results = engine.run_all()

    # ── 5. Visualizations ──────────────────────────────────────────
    viz    = Visualizer(df, schema)
    charts = viz.run_all(eda_results)

    # ── 6. LangGraph agent (narrative) ────────────────────────────
    narrative, tools_run = run_agent(df, schema, eda_results)

    # ── 7. Markdown report ─────────────────────────────────────────
    report_md = reporter.generate(
        filename    = file.filename,
        df          = df,
        schema      = schema,
        eda_results = eda_results,
        narrative   = narrative,
    )

    # ── 8. Return everything ───────────────────────────────────────
    return {
        "filename":       file.filename,
        "shape":          {"rows": len(df), "cols": len(df.columns)},
        "schema_summary": schema_summary,
        "eda": {
            "descriptive": eda_results["descriptive"],
            "missing":     eda_results["missing"],
            "outliers":    eda_results["outliers"],
            "correlation": eda_results["correlation"],
            "flags":       eda_results["flags"],
        },
        "charts":     charts,
        "narrative":  narrative,
        "tools_run":  tools_run,
        "report_md":  report_md,
    }