# AI-powered-EDA-report
# EDA Agent — Full Stack

Automated Exploratory Data Analysis web app powered by FastAPI, React, GPT-4o, and Supabase.

---

## Stack

| Layer | Tool |
|---|---|
| Backend | FastAPI + LangGraph |
| Frontend | React + Vite |
| Auth | Supabase |
| Backend hosting | Railway |
| Frontend hosting | Vercel |

---

## Local Development

### 1. Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# → fill in OPENAI_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY, SUPABASE_ANON_KEY

uvicorn main:app --reload
# API runs at http://localhost:8000
# Docs at  http://localhost:8000/docs
```

### 2. Frontend

```bash
cd frontend
npm install

cp .env.example .env
# → fill in VITE_SUPABASE_URL, VITE_SUPABASE_ANON_KEY, VITE_API_URL=http://localhost:8000

npm run dev
# App runs at http://localhost:5173
```

---

## Supabase Setup

1. Create a free project at https://supabase.com
2. Go to **Settings → API** and copy:
   - `URL` → `SUPABASE_URL` (backend) + `VITE_SUPABASE_URL` (frontend)
   - `anon public` key → `SUPABASE_ANON_KEY` + `VITE_SUPABASE_ANON_KEY`
   - `service_role` key → `SUPABASE_SERVICE_KEY` (backend only — never expose this)
3. Auth is ready out of the box. Email confirmation can be toggled in **Auth → Settings**.

---

## Deployment

### Backend → Railway

1. Push `backend/` to a GitHub repo
2. New project → Deploy from GitHub at https://railway.app
3. Add env vars under **Variables**:
   - `OPENAI_API_KEY`
   - `SUPABASE_URL`
   - `SUPABASE_SERVICE_KEY`
4. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Copy the Railway-provided URL (e.g. `https://eda-agent.up.railway.app`)

### Frontend → Vercel

1. Push `frontend/` to GitHub
2. Import at https://vercel.com
3. Add env vars:
   - `VITE_SUPABASE_URL`
   - `VITE_SUPABASE_ANON_KEY`
   - `VITE_API_URL` = your Railway URL
4. Click Deploy

### CORS — update for production

In `backend/config.py`, replace `"*"` with your Vercel domain:

```python
allowed_origins: list[str] = ["https://your-app.vercel.app"]
```

---

## File Structure

```
eda-app/
├── backend/
│   ├── main.py              # FastAPI app
│   ├── config.py            # Settings (reads .env)
│   ├── requirements.txt
│   ├── Procfile             # Railway start command
│   ├── core/
│   │   ├── loader.py        # DataLoader + SchemaDetector
│   │   ├── engine.py        # EDAEngine
│   │   ├── visualizer.py    # Plotly charts as JSON
│   │   ├── agent.py         # LangGraph agent
│   │   └── report.py        # Markdown report generator
│   └── routers/
│       ├── auth.py          # JWT verification dependency
│       ├── upload.py        # POST /api/analyze
│       └── chat.py          # POST /api/chat
└── frontend/
    ├── index.html
    ├── vite.config.js
    ├── vercel.json
    └── src/
        ├── main.jsx
        ├── App.jsx          # Routing + auth state
        ├── index.css        # Global design system
        ├── lib/
        │   ├── supabase.js  # Supabase client
        │   └── api.js       # Axios + JWT interceptor
        └── pages/
            ├── LoginPage.jsx / .module.css
            ├── Dashboard.jsx / .module.css
            └── ResultsPage.jsx / .module.css
```
