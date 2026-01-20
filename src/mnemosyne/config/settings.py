"""Pydantic settings models for Mnemosyne configuration."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class StorageSettings(BaseSettings):
    """ChromaDB storage configuration."""

    persist_directory: str = Field(
        default="./data/memory",
        description="Directory for ChromaDB persistence",
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="SentenceTransformer model for embeddings",
    )
    short_term_collection: str = Field(
        default="short_term_memory",
        description="Name of short-term memory collection",
    )
    long_term_collection: str = Field(
        default="long_term_memory",
        description="Name of long-term memory collection",
    )

    model_config = SettingsConfigDict(env_prefix="MNEMOSYNE_STORAGE_")


class ConsolidationSettings(BaseSettings):
    """Memory consolidation configuration."""

    importance_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum importance score for promotion to long-term",
    )
    min_age_hours: int = Field(
        default=1,
        ge=0,
        description="Minimum hours before memory eligible for consolidation",
    )
    max_batch_size: int = Field(
        default=50,
        ge=1,
        description="Maximum memories per consolidation cycle",
    )
    buffer_threshold: int = Field(
        default=100,
        ge=1,
        description="Memory count that triggers consolidation recommendation",
    )
    similarity_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for clustering",
    )
    min_cluster_size: int = Field(
        default=2,
        ge=1,
        description="Minimum memories to form a cluster",
    )

    model_config = SettingsConfigDict(env_prefix="MNEMOSYNE_CONSOLIDATION_")


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    level: str = Field(
        default="INFO",
        description="Log level (DEBUG, INFO, WARNING, ERROR)",
    )
    quiet: bool = Field(
        default=False,
        description="Suppress stderr output (used when running as subprocess)",
    )

    model_config = SettingsConfigDict(env_prefix="MNEMOSYNE_LOGGING_")


class Settings(BaseSettings):
    """Root configuration for Mnemosyne."""

    storage: StorageSettings = Field(default_factory=StorageSettings)
    consolidation: ConsolidationSettings = Field(default_factory=ConsolidationSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        case_sensitive=False,
    )
