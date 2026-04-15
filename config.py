import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # LLM Configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "phi3"
    
    # Storage Configuration
    CHROMA_PATH: str = "./chroma_data"
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = True
    
    # Project Info
    PROJECT_NAME: str = "RAG Deep-Dive Inspector"
    VERSION: str = "1.0.0"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
