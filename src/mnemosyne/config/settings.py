"""Pydantic settings models for Mnemosyne configuration."""

from typing import Tuple, Type

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from mnemosyne.core.constants import (
    CONSOLIDATION_IMPORTANCE_THRESHOLD,
    CONSOLIDATION_SIMILARITY_THRESHOLD,
)


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
        default=CONSOLIDATION_IMPORTANCE_THRESHOLD,
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
        default=CONSOLIDATION_SIMILARITY_THRESHOLD,
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


class RetrievalSettings(BaseSettings):
    """Memory retrieval configuration."""

    # Hybrid search weights
    emotion_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Default weight for emotional similarity in mood-congruent retrieval",
    )
    candidate_multiplier: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Retrieve N * multiplier candidates when reranking",
    )
    vector_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for vector similarity in hybrid search RRF",
    )
    bm25_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for BM25 keyword matching in hybrid search RRF",
    )

    # Quality scoring weights
    recency_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for recency in quality scoring",
    )
    importance_weight: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Weight for importance score in quality scoring",
    )
    relevance_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for base relevance in quality scoring",
    )

    # Quality scoring factors
    recency_decay_rate: float = Field(
        default=0.995,
        ge=0.9,
        le=1.0,
        description="Exponential decay rate per hour (0.995 = ~89% after 1 day)",
    )
    max_recency_hours: int = Field(
        default=720,
        ge=24,
        description="Cap for recency calculation (default 30 days)",
    )
    max_content_length: int = Field(
        default=500,
        ge=100,
        description="Content length at which length factor reaches 1.0",
    )
    semantic_type_factor: float = Field(
        default=1.2,
        ge=1.0,
        le=2.0,
        description="Quality multiplier for semantic memories vs episodic",
    )
    assistant_role_factor: float = Field(
        default=1.1,
        ge=0.5,
        le=2.0,
        description="Quality multiplier for assistant messages",
    )
    user_role_factor: float = Field(
        default=0.9,
        ge=0.5,
        le=2.0,
        description="Quality multiplier for user messages",
    )

    # Emotional similarity
    use_angular_similarity: bool = Field(
        default=False,
        description="Use angular (cosine) similarity instead of Euclidean for emotions",
    )

    model_config = SettingsConfigDict(env_prefix="MNEMOSYNE_RETRIEVAL_")


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
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    model_config = SettingsConfigDict(
        toml_file="configs/mnemosyne.toml",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        """Configure settings sources with TOML support.

        Priority (highest to lowest):
        1. Init settings (constructor arguments)
        2. Environment variables
        3. TOML config file
        4. Default values
        """
        return (
            init_settings,
            env_settings,
            TomlConfigSettingsSource(settings_cls),
        )
