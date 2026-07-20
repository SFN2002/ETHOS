"""
Application-wide settings for ETHOS.

Loads configuration from environment variables and a `.env` file using
pydantic-settings, with sensible defaults for local development.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centralised, validated configuration.

    Environment variables take precedence; missing values are loaded from `.env`.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # DeepSeek / OpenAI-compatible API
    deepseek_api_key: str | None = Field(default=None, alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com/v1", alias="DEEPSEEK_BASE_URL"
    )
    deepseek_model: str = Field(default="deepseek-chat", alias="DEEPSEEK_MODEL")
    deepseek_timeout: float = Field(default=30.0, alias="DEEPSEEK_TIMEOUT")
    deepseek_max_retries: int = Field(default=3, alias="DEEPSEEK_MAX_RETRIES")
    deepseek_temperature: float = Field(default=0.7, alias="DEEPSEEK_TEMPERATURE")

    # Simulation control
    simulation_days: int = Field(default=30, alias="SIMULATION_DAYS")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", alias="LOG_LEVEL"
    )
    data_dir: str = Field(default="data", alias="DATA_DIR")
    citizens_file: str = Field(default="citizens.json", alias="CITIZENS_FILE")

    @property
    def citizens_path(self) -> str:
        """Resolved path to the citizen registry JSON file."""
        from pathlib import Path

        return str(Path(self.data_dir) / self.citizens_file)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached, validated settings instance."""
    return Settings()
