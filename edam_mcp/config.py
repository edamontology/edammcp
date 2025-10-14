"""Configuration management for the EDAM MCP server."""

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # EDAM Ontology Configuration
    edam_ontology_url: str = Field(
        default="https://edamontology.org/EDAM.owl",
        description="URL to the default version of the EDAM ontology OWL file",
    )

    # Matching Configuration
    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for concept mappings",
    )

    max_suggestions: int = Field(default=5, ge=1, le=20, description="Maximum number of suggestions to return")

    # Model Configuration
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for text embeddings",
    )

    # Cache Configuration
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds for ontology data")

    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")

    model_config = {"env_prefix": "EDAM_", "case_sensitive": False}


# Global settings instance
settings = Settings()
