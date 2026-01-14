# Python Project Structure Best Practices Report
## Research for Elpis System Architecture

**Date:** January 2026
**Agent:** Project-Structure-Agent
**Status:** Complete

---

## Executive Summary

This report documents modern Python project best practices for 2025-2026, specifically tailored to the Elpis emotional coding agent system. Key recommendations include:

1. **Use src layout** with pyproject.toml for professional packaging
2. **Adopt uv** for fast, modern dependency management
3. **Implement Pydantic + TOML** for configuration
4. **Use structlog or loguru** for production-grade logging
5. **Pytest + pre-commit hooks** for quality assurance
6. **Ruff + mypy** for code quality and type safety

---

## 1. Modern Python Project Structure

### The src Layout (RECOMMENDED)

The **src layout** has become the industry standard for professional Python projects in 2025. This structure separates source code from other project components.

**Recommended Directory Structure:**
```
elpis/
├── src/
│   └── elpis/
│       ├── __init__.py
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── neuromodulated_llm.py
│       │   └── inference.py
│       ├── memory/
│       │   ├── __init__.py
│       │   ├── memory_system.py
│       │   ├── consolidation.py
│       │   └── retrieval.py
│       ├── emotion/
│       │   ├── __init__.py
│       │   ├── emotional_system.py
│       │   └── modulation.py
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── tool_engine.py
│       │   └── tool_definitions.py
│       ├── agent/
│       │   ├── __init__.py
│       │   ├── orchestrator.py
│       │   └── prompts.py
│       └── config/
│           ├── __init__.py
│           └── settings.py
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_emotion_system.py
│   │   ├── test_memory_system.py
│   │   └── test_tools.py
│   ├── integration/
│   │   ├── test_agent_workflow.py
│   │   └── test_memory_consolidation.py
│   └── fixtures/
│       └── sample_data.py
├── docs/
│   ├── architecture.md
│   ├── api.md
│   └── development.md
├── scripts/
│   ├── download_model.py
│   └── init_db.py
├── pyproject.toml
├── .pre-commit-config.yaml
├── .gitignore
├── README.md
└── LICENSE
```

**Advantages of src Layout:**

- Prevents accidental imports of uninstalled code during development
- Forces package to be installed before testing
- Cleaner separation between source and configuration
- Aligns with industry standards (setuptools, packaging guide)
- Better IDE and type checker support
- Facilitates proper packaging and distribution

### Project Organization Best Practices

1. **One package per project** - Keep `src/elpis/` as the main package
2. **Feature-based organization** - Group related functionality (llm, memory, emotion, tools, agent)
3. **Clear boundaries** - Each module has a single responsibility
4. **Configuration separation** - Keep configuration in `src/elpis/config/`
5. **Data storage** - Use directories outside src for runtime data:
   - `data/models/` - Downloaded LLM weights
   - `data/memory_db/` - ChromaDB storage
   - `data/memory_raw/` - Filesystem backups
   - `workspace/` - Agent's working directory

---

## 2. Dependency Management

### Modern Approach: pyproject.toml

**Key Changes from Legacy:**
- ✅ **Modern (2025):** `pyproject.toml` with declarative configuration
- ❌ **Legacy:** `setup.py` with executable Python code

**Why pyproject.toml?**
- **Security:** No arbitrary code execution during installation
- **Simplicity:** Declarative TOML syntax vs. imperative Python
- **Standardization:** PEP 621 compliant, widely supported
- **IDE Support:** Better tooling integration

### Recommended pyproject.toml Structure

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "elpis"
version = "0.1.0"
description = "Emotional coding agent with biologically-inspired memory and emotional modulation"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [{name = "Your Name", email = "your.email@example.com"}]
keywords = ["ai", "coding", "llm", "memory", "emotion"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

dependencies = [
    "llama-cpp-python>=0.2.0",
    "chromadb>=0.4.0",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "loguru>=0.7.0",
    "structlog>=23.0",
    "pyyaml>=6.0",
    "click>=8.0",
    "httpx>=0.24.0",
    "tenacity>=8.2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "pytest-asyncio>=0.21",
    "ruff>=0.1.0",
    "mypy>=1.5",
    "pre-commit>=3.4",
    "black>=23.0",
]

docs = [
    "sphinx>=7.0",
    "sphinx-rtd-theme>=1.3",
]

[project.scripts]
elpis = "elpis.cli:main"

[tool.setuptools]
packages = ["elpis"]

[tool.setuptools.package-data]
elpis = ["py.typed"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
addopts = "-v --cov=src/elpis --cov-report=term-missing --cov-report=html"
import_mode = "importlib"
markers = [
    "unit: unit tests",
    "integration: integration tests",
    "slow: slow tests",
]

[tool.coverage.run]
source = ["src/elpis"]
omit = ["*/tests/*", "*/test_*.py"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if __name__ == .__main__.:",
    "raise AssertionError",
    "raise NotImplementedError",
]

[tool.ruff]
target-version = "py310"
line-length = 100
select = ["E", "F", "I", "N", "UP", "W"]
ignore = ["E501"]  # Line length handled by formatter

[tool.ruff.isort]
known-first-party = ["elpis"]

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false  # Gradual typing adoption
disallow_incomplete_defs = false
check_untyped_defs = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true

[[tool.mypy.overrides]]
module = "llama_cpp.*"
ignore_missing_imports = true

[[tool.mypy.overrides]]
module = "chromadb.*"
ignore_missing_imports = true
```

### Dependency Management Tools Comparison

#### uv (RECOMMENDED for Elpis)
- **Speed:** 100x faster than pip in CI/CD scenarios
- **Lock file:** Deterministic `uv.lock` for reproducibility
- **Modern:** Written in Rust, actively developed
- **Best for:** Fast development, reproducible builds
- **Installation:** `pip install uv` then `uv sync`

**Why uv for Elpis:**
- Phase 1 needs fast iteration
- Complex dependencies (llama.cpp, chromadb, transformers)
- Want lock file for reproducible development environments

#### Poetry (Alternative if needed)
- **All-in-one:** Manages dependencies, virtual environments, packaging, publishing
- **Mature:** Widely adopted, battle-tested
- **TOML-based:** pyproject.toml-first approach
- **Best for:** Published packages, team projects

#### pip (For Simple Projects)
- **Simplicity:** Python's default, everyone knows it
- **Ecosystem:** Works with all tools
- **Limitations:** No automatic conflict resolution, no lock file

**Recommendation:** Start with **uv** for fast development, add **poetry** later if publishing to PyPI.

### Setting up uv for Elpis

```bash
# Install uv
pip install uv

# Initialize project with uv
uv init

# Create virtual environment and sync dependencies
uv sync

# Add dependency
uv add package-name

# Add dev dependency
uv add --dev pytest ruff mypy

# Update lock file
uv sync --upgrade
```

---

## 3. Configuration Management

### Recommended Approach: Pydantic + TOML

Combine **Pydantic** for runtime validation with **TOML** for human-readable configuration.

**Why this combination?**
- **Type Safety:** Pydantic validates configuration at startup
- **IDE Support:** Full autocomplete and type hints
- **Environment Variables:** Seamless .env integration
- **Flexibility:** Support TOML, JSON, environment variables

### Configuration Architecture for Elpis

**File Structure:**
```
elpis/
└── src/elpis/
    └── config/
        ├── __init__.py
        ├── settings.py          # Pydantic models
        └── defaults.toml        # Default configuration

configs/
├── config.default.toml          # Development defaults
├── config.prod.toml            # Production overrides
└── config.local.toml           # Local (git-ignored)
```

**settings.py - Pydantic Configuration Models:**

```python
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings

class ModelSettings(BaseSettings):
    """LLM Model configuration"""
    path: str = Field(
        default="./data/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        description="Path to GGUF model file"
    )
    context_length: int = Field(default=8192, ge=256, le=32768)
    gpu_layers: int = Field(default=35, ge=0, le=80)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(default=2048, ge=1, le=32768)

    class Config:
        env_prefix = "MODEL_"

class MemorySettings(BaseSettings):
    """Memory system configuration"""
    stm_capacity: int = Field(default=20, ge=1, le=1000)
    ltm_capacity: int = Field(default=1000, ge=1)
    consolidation_interval: int = Field(default=10, ge=1)
    importance_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    embedding_model: str = "all-MiniLM-L6-v2"
    db_path: str = "./data/memory_db"

    class Config:
        env_prefix = "MEMORY_"

class EmotionSettings(BaseSettings):
    """Emotional system configuration"""
    baseline: float = Field(default=0.5, ge=0.0, le=1.0)
    decay_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    update_magnitude: float = Field(default=0.15, ge=0.0, le=1.0)
    log_emotions: bool = True

    class Config:
        env_prefix = "EMOTION_"

class LoggingSettings(BaseSettings):
    """Logging configuration"""
    level: str = "INFO"
    format: str = "json"  # 'json' or 'text'
    log_emotions: bool = True
    log_memory_ops: bool = True
    output_file: Optional[str] = "./logs/elpis.log"

    class Config:
        env_prefix = "LOG_"

class Settings(BaseSettings):
    """Root settings configuration"""
    app_name: str = "Elpis"
    debug: bool = False

    model: ModelSettings = ModelSettings()
    memory: MemorySettings = MemorySettings()
    emotion: EmotionSettings = EmotionSettings()
    logging: LoggingSettings = LoggingSettings()

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"
        toml_file = "config.toml"

# Global settings instance
settings = Settings()
```

**config.toml - TOML Configuration File:**

```toml
app_name = "Elpis"
debug = false

[model]
path = "./data/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
context_length = 8192
gpu_layers = 35
temperature = 0.7
top_p = 0.9
max_tokens = 2048

[memory]
stm_capacity = 20
ltm_capacity = 1000
consolidation_interval = 10
importance_threshold = 0.6
embedding_model = "all-MiniLM-L6-v2"
db_path = "./data/memory_db"

[emotion]
baseline = 0.5
decay_rate = 0.05
update_magnitude = 0.15
log_emotions = true

[logging]
level = "INFO"
format = "json"
log_emotions = true
log_memory_ops = true
output_file = "./logs/elpis.log"
```

### Configuration Priority (High to Low)

1. Environment variables: `ELPIS_MODEL__PATH=/path/to/model`
2. `.env` file in project root
3. `config.toml` in `configs/` directory
4. Hardcoded defaults in `settings.py`

### Alternative: Dynaconf (For Complex Multi-Environment Setups)

If Elpis needs sophisticated multi-environment support later:

```python
from dynaconf import Dynaconf

settings = Dynaconf(
    envvar_prefix="ELPIS",
    settings_files=["config.default.toml", "config.local.toml"],
    environments=True,  # Support [dev], [prod], [test] sections
)

# Access as: settings.model.path
```

---

## 4. Logging Setup

### Recommended: Loguru (with structlog integration)

**Loguru** is the recommended primary logger for Elpis due to:
- Simple API: `from loguru import logger`
- Built-in rotation and retention
- Colored console output for development
- JSON output for production
- Context binding with `.bind()`

### Logger Configuration for Elpis

**src/elpis/config/logging_config.py:**

```python
import sys
from pathlib import Path
from loguru import logger
from elpis.config.settings import settings

def configure_logging():
    """Configure loguru for Elpis"""

    # Remove default handler
    logger.remove()

    # Console handler (colorized)
    logger.add(
        sys.stderr,
        format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=settings.logging.level,
        colorize=True,
    )

    # File handler (JSON for parsing)
    if settings.logging.output_file:
        Path(settings.logging.output_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            settings.logging.output_file,
            format="{message}",
            level=settings.logging.level,
            serialize=True,  # JSON output
            rotation="500 MB",  # Rotate when file reaches 500MB
            retention="7 days",  # Keep 7 days of logs
            compression="zip",  # Compress old logs
        )

    # Emotion-specific logging
    if settings.logging.log_emotions:
        emotion_log = Path("logs/emotions.jsonl")
        emotion_log.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            emotion_log,
            level="DEBUG",
            serialize=True,
            filter=lambda record: "emotion" in record["extra"],
        )

    return logger

# Initialize on module load
logger = configure_logging()
```

**Usage in Elpis code:**

```python
from elpis.config.logging_config import logger

# Basic logging
logger.info("Agent initialized")
logger.debug("STM size: {stm_size}", stm_size=len(stm))

# With context
logger.bind(agent_id="agent_1", task_id="task_42").info("Task completed")

# Emotion tracking
logger.bind(emotion=emotional_state).debug("Emotional update: {delta}", delta=delta)

# Exception logging
try:
    something()
except Exception as e:
    logger.exception("Failed to do something")
```

### Optional: structlog for Structured Logging

If you need distributed tracing and observability platform integration:

```python
import structlog

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

log = structlog.get_logger()
log.info("task_completed", task_id=123, duration_ms=1500)
```

---

## 5. Testing Framework Setup

### pytest Configuration for Elpis

**pyproject.toml sections (already shown above):**

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
addopts = "-v --cov=src/elpis --cov-report=term-missing --cov-report=html"
import_mode = "importlib"
markers = [
    "unit: unit tests",
    "integration: integration tests",
    "slow: slow tests",
]
```

### Test Structure for Elpis

```
tests/
├── conftest.py                 # Shared fixtures
├── unit/
│   ├── test_emotion_system.py
│   ├── test_memory_system.py
│   ├── test_tools.py
│   └── test_orchestrator.py
├── integration/
│   ├── test_agent_workflow.py
│   ├── test_memory_consolidation.py
│   └── test_tool_execution.py
├── fixtures/
│   ├── sample_data.py
│   └── mock_models.py
└── conftest.py
```

**conftest.py - Shared Fixtures:**

```python
import pytest
from pathlib import Path
from unittest.mock import Mock
from elpis.emotion.emotional_system import EmotionalSystem
from elpis.memory.memory_system import MemorySystem
from elpis.config.settings import Settings

@pytest.fixture
def temp_workspace(tmp_path):
    """Temporary workspace for testing"""
    return tmp_path

@pytest.fixture
def emotion_system():
    """Mock emotional system"""
    return EmotionalSystem()

@pytest.fixture
def memory_system(temp_workspace):
    """In-memory memory system for testing"""
    return MemorySystem(
        stm_capacity=10,
        ltm_path=temp_workspace / "ltm",
        db_path=temp_workspace / "db"
    )

@pytest.fixture
def test_settings():
    """Test configuration"""
    return Settings(
        debug=True,
        model__path="mock://test-model",
        memory__stm_capacity=10,
    )

@pytest.fixture
def mock_llm():
    """Mock LLM for testing"""
    llm = Mock()
    llm.generate.return_value = "Test response"
    return llm
```

**Example Unit Test:**

```python
import pytest
from elpis.emotion.emotional_system import EmotionalSystem

@pytest.mark.unit
class TestEmotionalSystem:
    """Test suite for emotional modulation"""

    def test_baseline_state(self, emotion_system):
        """Emotions start at baseline"""
        state = emotion_system.get_state()
        assert state.dopamine == 0.5
        assert state.serotonin == 0.5

    def test_dopamine_increase_on_success(self, emotion_system):
        """Dopamine increases on task success"""
        emotion_system.event("test_passed")
        state = emotion_system.get_state()
        assert state.dopamine > 0.5

    @pytest.mark.parametrize("event,emotion", [
        ("test_passed", "dopamine"),
        ("test_failed", "norepinephrine"),
        ("novel_solution", "acetylcholine"),
    ])
    def test_event_triggers_emotion(self, emotion_system, event, emotion):
        """Events trigger appropriate emotions"""
        initial = getattr(emotion_system.get_state(), emotion)
        emotion_system.event(event)
        final = getattr(emotion_system.get_state(), emotion)
        assert final != initial
```

**Example Integration Test:**

```python
@pytest.mark.integration
@pytest.mark.slow
class TestMemoryConsolidation:
    """Test memory consolidation workflow"""

    def test_stm_to_ltm_consolidation(self, memory_system):
        """STM items consolidate to LTM"""
        # Add items to STM
        memory_system.record_interaction("action_1", "result_1", 0.9)
        memory_system.record_interaction("action_2", "result_2", 0.8)

        # Trigger consolidation
        memory_system.consolidate()

        # Verify items in LTM
        results = memory_system.search_ltm("action_1", k=1)
        assert len(results) > 0
        assert "action_1" in results[0].content
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/elpis

# Run specific test type
pytest -m unit
pytest -m integration

# Run slow tests (integration)
pytest -m slow

# Run with output
pytest -v --tb=short

# Generate HTML coverage report
pytest --cov=src/elpis --cov-report=html
# Open htmlcov/index.html
```

---

## 6. Development Workflow & Tooling

### Modern Python Development Stack for Elpis

#### Primary Tools

1. **uv** - Package management and Python version management
2. **ruff** - Fast linter and formatter (replaces pylint + black + isort)
3. **mypy** - Static type checking
4. **pytest** - Testing framework
5. **pre-commit** - Git hooks for quality checks

#### Installation

```bash
# Install uv
pip install uv

# Initialize project with uv
uv sync

# Add dev dependencies
uv add --dev pytest pytest-cov pytest-asyncio ruff mypy pre-commit

# Initialize pre-commit
pre-commit install
```

### .pre-commit-config.yaml

```yaml
default_language_version:
  python: python3.10

repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-toml
      - id: check-added-large-files
        args: ['--maxkb=1000']

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.5
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.0
    args: [--strict, --warn-unused-ignores]
    additional_dependencies:
      - types-PyYAML
      - types-requests

  - repo: https://github.com/commitizen-tools/commitizen
    rev: 3.12.0
    hooks:
      - id: commitizen
        stages: [commit-msg]
```

### Ruff Configuration (in pyproject.toml)

```toml
[tool.ruff]
target-version = "py310"
line-length = 100

# Enable these rule sets
select = [
    "E",      # Pycodestyle errors
    "W",      # Pycodestyle warnings
    "F",      # Pyflakes
    "I",      # isort
    "N",      # pep8-naming
    "UP",     # pyupgrade
    "B",      # flake8-bugbear
    "A",      # flake8-builtins
    "C4",     # flake8-comprehensions
    "RUF",    # ruff-specific rules
]

# Ignore specific rules
ignore = [
    "E501",   # Line length (handled by formatter)
    "W503",   # Line break before binary operator
]

[tool.ruff.isort]
known-first-party = ["elpis"]
force-single-line = false

[tool.ruff.flake8-bugbear]
extend-immutable-calls = ["fastapi.Depends", "fastapi.Header"]
```

### Mypy Configuration (in pyproject.toml)

```toml
[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false  # Gradual typing
check_untyped_defs = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true

[[tool.mypy.overrides]]
module = ["llama_cpp.*", "chromadb.*"]
ignore_missing_imports = true
```

### Development Workflow Commands

```bash
# Format and lint code
uv run ruff check --fix .
uv run ruff format .

# Type checking
uv run mypy src/elpis

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src/elpis --cov-report=html

# Run specific test
uv run pytest tests/unit/test_emotion_system.py::TestEmotionalSystem::test_baseline_state

# Check for pre-commit issues
pre-commit run --all-files

# Manually trigger pre-commit on staged files
pre-commit run
```

### VSCode Settings (.vscode/settings.json)

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.mypyEnabled": true,
  "python.linting.mypyArgs": ["--strict"],
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.fixAll.ruff": "explicit"
    }
  },
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests"]
}
```

### GitHub Actions CI/CD Pipeline

**.github/workflows/test.yml:**

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v1

      - name: Set up Python
        run: uv python install ${{ matrix.python-version }}

      - name: Install dependencies
        run: uv sync

      - name: Lint with ruff
        run: uv run ruff check src/elpis

      - name: Type check with mypy
        run: uv run mypy src/elpis

      - name: Test with pytest
        run: uv run pytest --cov=src/elpis --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
```

---

## 7. Integration Recommendations for Elpis

### Phase 1: Initial Setup
- [ ] Restructure to src layout
- [ ] Create pyproject.toml with all configuration
- [ ] Set up uv for dependency management
- [ ] Initialize pre-commit with ruff + mypy
- [ ] Create basic pytest structure with conftest.py
- [ ] Set up logging with loguru

### Phase 2: Quality Assurance
- [ ] Implement Pydantic configuration system
- [ ] Add comprehensive test coverage (unit + integration)
- [ ] Set up GitHub Actions CI/CD
- [ ] Add type hints to codebase (gradual)
- [ ] Implement coverage requirements (>80%)

### Phase 3: Production Readiness
- [ ] Switch to structlog for distributed tracing
- [ ] Implement structured JSON logging
- [ ] Add performance monitoring hooks
- [ ] Create deployment documentation
- [ ] Set up staging environment

### Phase 4: Observability (Later)
- [ ] Integrate with monitoring platform (Datadog, NewRelic)
- [ ] Add trace ID propagation
- [ ] Implement custom metrics for emotional state
- [ ] Create dashboards for memory system performance

---

## Key Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Project Layout** | src layout | Industry standard, prevents import issues |
| **Packaging** | pyproject.toml | Modern, secure, declarative |
| **Dependency Manager** | uv | Fast, reproducible, modern |
| **Configuration** | Pydantic + TOML | Type-safe, validated, human-readable |
| **Logging** | loguru | Simple API, production-ready |
| **Testing** | pytest | Industry standard, rich ecosystem |
| **Code Quality** | ruff + mypy | Fast, comprehensive, modern |
| **Git Hooks** | pre-commit | Automated quality checks |

---

## Resources & References

### Official Documentation
- [Python Packaging Guide - Project Structure](https://packaging.python.org/en/latest/)
- [PEP 517 - Python Package Build Interface](https://peps.python.org/pep-0517/)
- [PEP 621 - Project Metadata](https://peps.python.org/pep-0621/)
- [pytest Documentation](https://docs.pytest.org/)
- [loguru GitHub](https://github.com/Delgan/loguru)
- [Pydantic Documentation](https://docs.pydantic.dev/)

### Recommended Articles
- [Real Python - Project Layout Best Practices](https://realpython.com/ref/best-practices/project-layout/)
- [The Hitchhiker's Guide to Python - Structuring Projects](https://docs.python-guide.org/writing/structure/)
- [Ruff: The Modern Python Linter](https://astral.sh/ruff)
- [uv: The Modern Python Package Manager](https://astral.sh/uv)

### Configuration Examples
- setuptools documentation on pyproject.toml
- Pydantic BaseSettings examples
- Dynaconf multi-environment configuration

---

## Conclusion

The Elpis project should adopt modern Python best practices with:

1. **src layout** for clean separation of concerns
2. **uv + pyproject.toml** for fast, reproducible builds
3. **Pydantic + TOML** for validated configuration
4. **loguru** for production-grade logging
5. **pytest** with comprehensive test coverage
6. **ruff + mypy + pre-commit** for code quality automation

This architecture supports Elpis's complex requirements (multi-tier memory, emotional modulation, tool execution) while maintaining code quality and developer productivity.

---

**Report Compiled by:** Project-Structure-Agent
**Date:** January 11, 2026
**Status:** Ready for Implementation
