from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from routers import auth, upload, chat

app = FastAPI(
    title="EDA Agent API",
    version="1.0.0",
    description="Automated Exploratory Data Analysis powered by GPT-4o and LangGraph",
)

# ── CORS ───────────────────────────────────────────────────────────
# In production replace allow_origins with your actual Vercel domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins + ["*"],   # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────
app.include_router(auth.router,   prefix="/api/auth")
app.include_router(upload.router, prefix="/api")
app.include_router(chat.router,   prefix="/api")


# ── Health check ───────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "message": "EDA Agent API is running."}