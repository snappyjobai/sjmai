import os
from typing import Optional
from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ENVIRONMENT: str = "development"
    DB_HOST: str = "localhost"
    DB_USER: str = "root"
    DB_PASSWORD: str = "Sjmrootpass"
    DB_NAME: str = "sjm_db"
    REDIS_HOST: Optional[str] = "localhost"
    REDIS_PORT: Optional[int] = 6379
    REDIS_PASSWORD: Optional[str] = None
    API_PORT: int = 8000
    BASE_URL: str = "http://localhost:8000"

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()