import os
from typing import List, Union
from pydantic import AnyHttpUrl, field_validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "RecycleVision AI"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    DESCRIPTION: str = "Smart Waste Classification & Recycling Analytics API"
    
    # Model Configuration
    MODEL_PATH: str = os.getenv("MODEL_PATH", "backend/models/best_model.h5")
    IMAGE_SIZE: int = 224
    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "10"))
    
    # Supported waste classes (exact alphabetical order matching training ImageDataGenerator)
    CLASSES: List[str] = [
        "cardboard",
        "glass",
        "metal",
        "paper",
        "plastic",
        "trash"
    ]
    
    # CORS Configuration
    ALLOWED_ORIGINS: Union[List[str], str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "https://recyclevision-ai.vercel.app",
        "*"
    ]
    
    @field_validator("ALLOWED_ORIGINS", mode="before")
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            if v.startswith("[") and v.endswith("]"):
                import json
                try:
                    return json.loads(v)
                except Exception:
                    pass
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    # Supabase Configuration
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", os.getenv("SUPABASE_ANON_KEY", ""))
    SUPABASE_SERVICE_ROLE_KEY: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    
    # Admin Security
    ADMIN_SECRET_KEY: str = os.getenv("ADMIN_SECRET_KEY", "recyclevision-admin-dev-secret-key-2026")
    
    # OpenAI ChatGPT Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Local SQLite / Storage Fallback
    SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH", "backend/data/recyclevision.db")
    REPORTS_DIR: str = os.getenv("REPORTS_DIR", "backend/reports")

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
