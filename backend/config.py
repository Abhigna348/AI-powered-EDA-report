from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str
    supabase_url: str
    supabase_service_key: str
    supabase_anon_key: str

    allowed_origins: list[str] = [
        "http://localhost:5173",
        "http://localhost:3000"
    ]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()


# Optional debug check (remove in production)
print("Loaded SUPABASE_URL:", settings.supabase_url)