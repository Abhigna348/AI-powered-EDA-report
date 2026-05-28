from fastapi import APIRouter, Header, HTTPException, Depends
from supabase import create_client, Client

from config import settings

router = APIRouter(tags=["auth"])


def get_supabase() -> Client:
    return create_client(settings.supabase_url, settings.supabase_service_key)


def get_current_user(authorization: str = Header(...)) -> dict:
    """
    Dependency — validates the Supabase JWT sent in the Authorization header.
    Usage: user = Depends(get_current_user)
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid auth header format.")

    token = authorization.removeprefix("Bearer ").strip()

    try:
        supabase = get_supabase()
        result   = supabase.auth.get_user(token)
        if result is None or result.user is None:
            raise HTTPException(status_code=401, detail="Invalid or expired token.")
        return {"id": result.user.id, "email": result.user.email}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Auth error: {exc}") from exc


# ── /api/auth/me — quick health-check for auth ────────────────────
@router.get("/me")
def me(user: dict = Depends(get_current_user)):
    return {"user": user}