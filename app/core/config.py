import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings loaded from environment variables and .env file."""
    
    # Path to the pipeline config YAML file
    APP_CONFIG_PATH: str = "config/settings.yaml"
    
    # API Host and Port
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    
    # Debug mode (enables reload, detailed logs, etc.)
    DEBUG: bool = False
    
    # Workers count for production
    MAX_WORKERS: int = 4
    
    # Automatically load values from a .env file if it exists
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


# Instantiate the global settings object
settings = Settings()
