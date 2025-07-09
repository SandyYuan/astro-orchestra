"""Configuration settings for Astro Orchestra."""

from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    google_api_key: str
    arxiv_api_key: Optional[str] = None
    
    # Agent Configuration
    orchestrator_model: str = "gemini-2.5-pro"
    specialist_model: str = "gemini-2.5-flash"
    
    # Data Sources
    desi_base_url: str = "https://data.desi.lbl.gov/public/"
    lsst_base_url: str = "https://lsst.ncsa.illinois.edu/"
    desi_api_key: Optional[str] = None
    lsst_api_key: Optional[str] = None
    
    # Workflow Configuration
    max_iterations: int = 10
    timeout_seconds: int = 300
    
    # Development
    log_level: str = "INFO"
    debug: bool = False
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"  # Ignore extra environment variables
    )


# Global settings instance
settings = Settings() 