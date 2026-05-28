import json
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from config  import settings
from routers.auth import get_current_user

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    question:    str
    # Condensed EDA context passed back from the frontend
    context:     dict
    # Previous turns: [{"role": "user"|"assistant", "content": "..."}]
    history:     list[dict] = []


class ChatResponse(BaseModel):
    answer: str


def _build_system_prompt(ctx: dict) -> str:
    shape      = ctx.get("shape", {})
    schema     = ctx.get("schema_summary", [])
    flags      = ctx.get("flags", [])
    pairs      = ctx.get("correlations", [])
    narrative  = ctx.get("narrative", "")
    descriptive = ctx.get("descriptive", [])

    return f"""You are a data analyst assistant helping a user explore their dataset.
Answer questions accurately and concisely. Cite specific numbers from the EDA results.
If a computation is not in the results, say so clearly — do not invent numbers.

DATASET:
  Rows: {shape.get('rows', '?')}  Cols: {shape.get('cols', '?')}
  Columns: {', '.join(r['column'] for r in schema)}

ANOMALY FLAGS:
{json.dumps(flags[:20], indent=2)}

STRONG CORRELATIONS:
{json.dumps(pairs[:10], indent=2)}

DESCRIPTIVE STATS (sample):
{json.dumps(descriptive[:6], indent=2)}

NARRATIVE SUMMARY:
{narrative}
""".strip()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    user:    dict = Depends(get_current_user),
):
    if not payload.question.strip():
        raise HTTPException(status_code=422, detail="Question must not be empty.")

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=settings.openai_api_key,
    )

    system_prompt = _build_system_prompt(payload.context)

    messages: list = [SystemMessage(content=system_prompt)]

    # Last 6 turns for token efficiency
    for turn in payload.history[-6:]:
        role    = turn.get("role", "user")
        content = turn.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        else:
            # Assistant turns re-injected as system messages (simple approach)
            messages.append(SystemMessage(content=f"(Previous answer) {content}"))

    messages.append(HumanMessage(content=payload.question))

    try:
        response = llm.invoke(messages)
        return ChatResponse(answer=response.content)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM error: {exc}") from exc