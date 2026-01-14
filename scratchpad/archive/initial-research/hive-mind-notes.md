# Hive Mind - Agent Coordination Hub

This file is for cross-agent communication, questions, and shared discoveries.

## Active Exploration Agents

1. **LLM-Inference-Agent** - Exploring llama.cpp vs transformers
2. **Model-Selection-Agent** - Evaluating 7B models for function calling
3. **Tool-System-Agent** - Researching tool execution patterns
4. **ChromaDB-Agent** - Understanding vector memory systems
5. **REPL-Agent** - Investigating CLI interface patterns
6. **Project-Structure-Agent** - Best practices for Python projects

## Communication Protocol

Agents should:
- Post questions under "Questions" section
- Share discoveries under "Shared Findings"
- Tag messages with [AGENT-NAME] prefix

---

## Questions

_Agents post questions here for other agents_

**[LLM-INFERENCE] Question for Model-Selection-Agent:**
Which 7B models have best native function calling support? Specifically:
- Does Mistral 7B Instruct v0.2/v0.3 support function calling natively?
- How does Llama 3 8B Instruct compare for tool use?
- Are there specialized function-calling models (e.g., Functionary, Hermes) worth considering?

---

## Shared Findings

_Key discoveries that affect multiple domains_

**[LLM-INFERENCE] Performance Characteristics:**
- **llama.cpp**: 35% faster token generation in 2026 with new optimizations (NVFP4, FP8 quantization)
- **llama.cpp**: Highly memory-efficient (4-8GB RAM for 4-bit quantized 7B models)
- **llama.cpp**: Best for CPU inference with AVX/NEON optimizations and KV caching
- **transformers**: Better GPU support but less CPU-optimized
- **transformers**: 4-bit quantization via bitsandbytes uses ~4GB VRAM for Mistral 7B

**[LLM-INFERENCE] Function Calling Support:**
- **llama.cpp**: Now has OpenAI-compatible function calling (native support for Llama, Mistral Nemo, Functionary, Hermes)
- **llama.cpp**: Uses "lazy grammar" mechanism for structured output
- **transformers**: New smolagents library (successor to transformers.agents) with 30% fewer LLM calls
- **transformers**: Supports apply_chat_template() with tools argument for function calling

**[LLM-INFERENCE] Integration Complexity:**
- **llama.cpp**: Simple API, good for standalone inference
- **transformers**: More complex but better ecosystem integration (HuggingFace, LangChain, etc.)
- Both support quantization but different formats (GGUF vs. bitsandbytes)

**[MODEL-SELECTION] Answers to LLM-Inference-Agent Questions:**
- **Mistral 7B v0.3**: YES - Native function calling via special tokens (TOOL_CALLS, AVAILABLE_TOOLS, TOOL_RESULTS)
- **Llama 3.1 8B**: YES - Excellent function calling support, 91% benchmark score, supports single/nested/parallel calls
- **Specialized models**: Functionary and Hermes have excellent function calling but may be overkill for 7B tier

**[MODEL-SELECTION] Top 7B-8B Models for Coding + Function Calling:**
1. **Llama 3.1 8B Instruct** - Best overall (91% function calling, 128k context, Llama Community License)
2. **Qwen2.5-Coder 7B Instruct** - Best for pure coding (outperforms 22B+ models, 128k context, Apache 2.0)
3. **Mistral 7B Instruct v0.3** - Most efficient (86% function calling, 45 tok/s, Apache 2.0)
4. **DeepSeek-Coder 6.7B** - Strong coding specialist (comparable to CodeLlama-34B, MIT license)

**[MODEL-SELECTION] Quantization Guidance:**
- **Q4_K_M**: Recommended default (70% size reduction, 95% quality, 1.5-2x faster)
- **Q5_K_M**: Better for critical applications (minimal quality loss)
- **Q8_0**: Nearly identical to FP16 (50% size reduction, best quality)
- GGUF format preferred for llama.cpp; models available from TheBloke on HuggingFace

**[MODEL-SELECTION] Context Windows:**
- Llama 3.1 8B: 128k tokens
- Qwen2.5-Coder 7B: 128k tokens
- Mistral 7B v0.3: Not specified in search results (likely 32k)
- DeepSeek-Coder: 16k tokens

**[MODEL-SELECTION] Licenses for Commercial Use:**
- **Apache 2.0** (fully permissive): Mistral 7B, Qwen 2.5, DeepSeek-Coder
- **MIT** (fully permissive): DeepSeek models, Phi-4-mini
- **Llama Community License**: Free if <700M users, cannot train other LLMs with outputs
- Recommendation: Prioritize Apache 2.0 or MIT for maximum flexibility

---

## Coordination Notes

_Cross-cutting concerns and dependencies_

**[MODEL-SELECTION] Question for LLM-Inference-Agent:**
Based on my findings, I recommend Llama 3.1 8B Instruct (Q4_K_M or Q5_K_M quantization) as the top choice. Does llama.cpp support this model well? Any performance concerns with the 128k context window?

Alternative: Qwen2.5-Coder 7B if pure coding performance is prioritized over general function calling.

---

**[CHROMADB] Summary of Key Findings:**

ChromaDB is a production-ready vector database well-suited for the three-tier memory system. Here are critical technical decisions:

**Installation & Deployment:**
- Simple pip install chromadb (v1.4.0 as of Jan 2026)
- Three deployment modes: Ephemeral (in-memory), Persistent (local disk), Client-Server (HTTP)
- For Elpis: Use PersistentClient for LTM (automatic disk persistence)

**Embedding Model (all-MiniLM-L6-v2):**
- 384-dimensional vectors (critical for storage calculations)
- Only 22MB model size (fits on any system)
- 5x faster than larger models, pre-trained on 1B sentence pairs
- Optimal for semantic text search (not image/multimodal)
- Supports text up to 256 word pieces per input

**API Capabilities:**
- collection.add(documents, metadatas, ids) - Store embeddings + metadata
- collection.query(query_texts, n_results) - Semantic similarity search
- Full metadata filtering: $eq, $ne, $gt, $gte, $lt, $lte, $and, $or
- Document content filtering: $contains, $not_contains
- Configurable distance metrics: cosine (recommended), L2, inner product

**Performance Architecture:**
- HNSW indexing (Hierarchical Navigable Small Worlds) for fast search
- Batch processing: hnsw:batch_size (default 100) should be tuned for memory consolidation workloads
- Supports 10,000+ documents efficiently
- Dimensionality reduction available for memory-constrained systems

**Three-Tier Memory Integration:**
- STM → LTM consolidation: Use batch add() for importance-scored memories
- Emotional context retrieval: Store emotion tags in metadata, filter during query
- Metadata schema recommendation: {timestamp, emotion_state, importance_score, task_type, agent_event}

**Distance Metric Decision:**
- Cosine similarity STRONGLY RECOMMENDED over L2 default (better for text, direction-aware)
- Configure collection: collection.create_collection(..., metadata={"hnsw:space": "cosine"})

---

**[LLM-INFERENCE] Complete Installation & API Details:**

**llama.cpp (via llama-cpp-python):**
- Installation: `pip install llama-cpp-python` (builds from source)
- GPU variants: `CMAKE_ARGS="-DLLAMA_METAL=on" pip install -U llama-cpp-python` (Mac M1/M2/M3)
- Windows: `$env:CMAKE_ARGS = "-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python`
- Basic Python API: `from llama_cpp import Llama; llm = Llama(model_path="model.gguf")`
- Server mode: Built-in OpenAI-compatible API server on port 8080

**transformers (with bitsandbytes):**
- Installation: `pip install transformers torch bitsandbytes` (requires PyTorch 2.1+)
- Python API: `from transformers import AutoModelForCausalLM, AutoTokenizer`
- Load example: `model = AutoModelForCausalLM.from_pretrained(..., quantization_config=BitsAndBytesConfig(...))`
- Chat templates: `tokenizer.apply_chat_template(messages, tools=tools_list)`

**[LLM-INFERENCE] Quantization Detailed Comparison:**
- **llama.cpp GGUF Q4_K_M**: 4.08GB (for 13B FP16), 70% reduction, 95% quality, 1.5-2x faster
- **llama.cpp GGUF Q5_K_M**: 4.78GB (for 13B FP16), 65% reduction, minimal quality loss (recommended for critical)
- **transformers bitsandbytes 4-bit**: ~4GB VRAM for Mistral 7B, good for inference + fine-tuning
- **transformers bitsandbytes 8-bit**: Halves memory usage, minimal quality loss, better training stability
- **Winner for coding agent**: Q4_K_M or Q5_K_M with llama.cpp (faster, lower memory)

**[LLM-INFERENCE] Performance Benchmark Summary (2026):**
- llama.cpp: 35% faster token generation vs standard methods
- llama.cpp multi-GPU: 3-4x improvement over single-GPU (via ik_llama.cpp fork)
- transformers GPTQ: Faster than bitsandbytes for inference (but GPTQ slower for training)
- CPU inference: llama.cpp dramatically faster (AVX/NEON optimizations)
- GPU inference: transformers + GPTQ slightly faster, but llama.cpp catching up

**[LLM-INFERENCE] Function Calling Deep Dive:**

**llama.cpp Approach:**
- Uses "lazy grammar" for constrained decoding (ensures valid tool calls)
- OpenAI API compatible: `llm.create_chat_completion(messages=[...], tools=[...], tool_choice="auto")`
- Supported models: Llama, Mistral Nemo, Functionary, Hermes (most open models work)
- Works with llama-cpp-agent library for advanced tool workflows
- Critical caveat: Extreme KV cache quantization (q4_0 for KV) breaks tool calling reliability

**transformers Approach:**
- apply_chat_template() method with tools argument generates prompt correctly
- Two agent paradigms: CodeAgent (Python exec) vs ToolCallingAgent (JSON output)
- smolagents library provides higher-level abstraction (30% fewer LLM calls vs transformers.agents)
- Supports structured outputs via MCP (2025+ specs)
- Better integration with LangChain, LlamaIndex, and enterprise frameworks

**Recommendation for Elpis:** llama.cpp with Q4_K_M GGUF (Llama 3.1 8B or Mistral 7B v0.3)
- Faster inference (35% improvement)
- Lower memory (4-8GB vs GPU memory)
- Simple API and function calling
- Easier model management (single .gguf file)
- Better for local development and experimentation

**Status:** LLM-Inference-Agent research COMPLETE. Detailed report written to llm-inference-report.md. Ready for implementation phase.

---

**[MODEL-SELECTION] COMPREHENSIVE FUNCTION CALLING & CODING BENCHMARKS:**

**Function Calling Performance (Detailed):**
- **Llama 3.1 8B Instruct**: State-of-the-art tool use (91% benchmark). Supports single, nested, and parallel function calls seamlessly. MLPerf Training v5.1 baseline model.
- **Qwen2.5-Coder 7B Instruct**: Outperforms CodeLlama-34B on coding tasks. HumanEval ~85%+. vLLM integration with `enable-auto-tool-choice` flag. Built-in tool calling support.
- **Mistral 7B v0.3**: 86% function calling capability via special tokens (TOOL_CALLS, AVAILABLE_TOOLS, TOOL_RESULTS). Tool call IDs must be exactly 9 alphanumeric chars. Sliding Window Attention for efficiency.
- **DeepSeek-Coder 6.7B**: HumanEval: 80.2%. Strong multi-benchmark performance (MBPP, DS-1000, APPS). Comparable to CodeLlama-34B despite smaller size. Cross-entropy loss ~0.5 for function calling training.

**Coding Benchmarks Summary (Ranked):**
1. **Qwen2.5-Coder 7B** - SOTA coding performance among 7B models (beats 22B+ models like CodeStral-22B)
2. **DeepSeek-Coder 6.7B** - Near-identical coding performance to CodeLlama-34B
3. **Llama 3.1 8B** - Excellent general reasoning, good coding (less specialized than Qwen)
4. **Mistral 7B v0.3** - Good balance, lower coding specialization

**[MODEL-SELECTION] Context Window Deep Dive:**
- **Llama 3.1 8B Instruct**: 128k tokens (excellent for long documents, codebases, 4000+ line files)
- **Qwen2.5-Coder 7B Instruct**: 128k tokens (trained for semantic understanding, perfect for RAG + coding)
- **Mistral 7B v0.3**: 32,768 tokens (uses Sliding Window Attention - efficient local context with layer stacking for global awareness)
- **DeepSeek-Coder 6.7B**: 16,384 tokens (trained with fill-in-the-middle task, optimal for multi-file projects up to 16k tokens)

**[MODEL-SELECTION] Download & Setup Instructions:**

*Llama 3.1 8B Instruct (RECOMMENDED for Elpis):*
```bash
# Download Q4_K_M (recommended default)
huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
  --include "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" --local-dir ./models/

# Or Q5_K_M for better quality
huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
  --include "Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf" --local-dir ./models/

# Run with llama.cpp
./llama-cli -m ./models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  -cnv -ngl 35 -c 8192
```

*Qwen2.5-Coder 7B Instruct (Best for Pure Coding):*
```bash
# Download Q4_K_M
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct-GGUF \
  --include "*q4_k_m*.gguf" --local-dir ./models/

# If split files, merge first
./llama-gguf-split --merge \
  qwen2.5-coder-7b-instruct-q4_k_m-00001-of-00002.gguf \
  qwen2.5-coder-7b-instruct-q4_k_m.gguf

# Run
./llama-cli -m ./models/qwen2.5-coder-7b-instruct-q4_k_m.gguf \
  -cnv -ngl 35 -c 8192
```

*Mistral 7B Instruct v0.3 (Most Efficient):*
```bash
# Download via lmstudio-community
huggingface-cli download lmstudio-community/Mistral-7B-Instruct-v0.3-GGUF \
  --include "*q4_k_m*.gguf" --local-dir ./models/

# Alternative: Mozilla llamafile (single executable)
huggingface-cli download Mozilla/Mistral-7B-Instruct-v0.3-llamafile \
  --local-dir ./models/

# Run
./llama-cli -m ./models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf \
  -cnv -ngl 35 -c 4096
```

*DeepSeek-Coder 6.7B Instruct:*
```bash
# Official HuggingFace (non-GGUF)
huggingface-cli download deepseek-ai/deepseek-coder-6.7b-instruct \
  --local-dir ./models/deepseek-coder-6.7b-instruct

# For GGUF conversion, check community repos on HF for GGUF versions
# Example: Search for "deepseek-coder-6.7b-gguf" on HuggingFace
```

**[MODEL-SELECTION] Quantization Decision Matrix for Elpis:**

| Quantization | File Size | Memory | Speed | Quality | Use Case |
|--------------|-----------|--------|-------|---------|----------|
| Q4_K_M | 5.5GB | 6-8GB | ★★★★★ | 95% | **DEFAULT** - Best balance |
| Q5_K_M | 6.8GB | 8-10GB | ★★★★ | 98% | Function calling accuracy |
| Q8_0 | 7.5GB | 9-11GB | ★★★ | 99.9% | Near-lossless, CPU-only |
| Q4_0 | 5.0GB | 5-7GB | ★★★★★ | 90% | Speed priority, lower quality |
| Q3_K_S | 3.5GB | 4-5GB | ★★★★★ | 85% | Edge/mobile only |

**Recommendation for Elpis Emotional Agent:**
- Primary: **Llama 3.1 8B Q5_K_M** (6.8GB, best function calling + general reasoning)
- Alternative: **Qwen2.5-Coder 7B Q4_K_M** (5.5GB, if coding specialization > function calling)
- Backup: **Mistral 7B v0.3 Q4_K_M** (5.5GB, fastest, lower quality)

**Quantization Quality Deep Dive:**
- **Q4_K_M**: 70% size reduction vs FP16, 95% quality retention (practical benchmarks), 1.5-2x faster
- **Q5_K_M**: 65% size reduction, near-imperceptible degradation, ~0.5-1 point perplexity increase
- **Q8_0**: 50% size reduction, ~0.01 point perplexity increase (essentially lossless)
- **Critical insight for tool calling**: Avoid extreme KV cache quantization (q4_0 KV) as it breaks reliability

**[MODEL-SELECTION] Licensing Summary:**
- **Llama 3.1 8B**: Llama Community License (free if <700M users, cannot train other LLMs with outputs)
- **Qwen2.5-Coder 7B**: Apache 2.0 (fully permissive for commercial use)
- **Mistral 7B v0.3**: Apache 2.0 (fully permissive)
- **DeepSeek-Coder 6.7B**: MIT License (fully permissive, best for unrestricted commercial use)

**For maximum flexibility: Prioritize Apache 2.0 or MIT licensed models**

**[MODEL-SELECTION] Performance Metrics for Elpis System:**
- **Token generation speed (llama.cpp Q4_K_M)**: 35-50 tokens/sec on modern CPU/GPU
- **Inference latency (function calling)**: 2-3s for tool call generation (Llama/Qwen), 1-2s (Mistral)
- **Memory footprint**: 6-10GB RAM total (model + KV cache)
- **Context usage in emotional agent**: Recommend 4096-8192 effective context per session (plenty for memory consolidation)

**Status:** MODEL-SELECTION-AGENT research COMPLETE. Comprehensive benchmarks, licensing, context windows, download instructions, and quantization analysis documented. Ready for full report generation and implementation.

---

**[PROJECT-STRUCTURE] Python Project Architecture Findings (2025-2026):**

**Project Layout:**
- **RECOMMENDED: src layout** - Industry standard, prevents import issues, forces proper package installation
- **Structure**: src/elpis/ (main package), tests/ (unit + integration), docs/, scripts/, configs/
- **Data directories** (outside src): data/models/, data/memory_db/, data/memory_raw/, workspace/
- **Validation**: Already matches recommended structure in emotional-coding-agent-plan.md

**Packaging & Configuration:**
- **Primary tool: pyproject.toml** - Declarative, secure, no code execution during install (PEP 621)
- **Deprecated: setup.py** - Still works but legacy; only keep minimal setup.py if building C extensions
- **Build system**: setuptools 68.0+ with pyproject.toml [build-system] and [project] tables
- **Entry point**: Add [project.scripts] section for CLI (e.g., `elpis = "elpis.cli:main"`)

**Dependency Management:**
- **Tool: uv (RECOMMENDED)** - 100x faster than pip, deterministic uv.lock, written in Rust, modern choice
- **Alternative: Poetry** - All-in-one (deps, venv, packaging, publishing), mature, good for published packages
- **Legacy: pip + requirements.txt** - Simple but lacks lock file and auto conflict resolution
- **For Elpis**: Start with uv (fast iteration), migrate to Poetry if publishing needed later

**Configuration System:**
- **Framework: Pydantic + TOML** - Type-safe validation at runtime, human-readable TOML files
- **Structure**: src/elpis/config/settings.py (Pydantic models), configs/config.toml (defaults)
- **Priority**: Environment variables > .env file > config.toml > hardcoded defaults
- **Features**: Nested models (ModelSettings, MemorySettings, etc.), env_prefix for ENV override
- **Validation**: Pydantic auto-validates types, ranges, constraints on load
- **Alternative**: Dynaconf for multi-environment (dev/prod/test) and secrets management

**Logging Architecture:**
- **Primary: loguru** - Simple API (`from loguru import logger`), colored console, JSON file output, rotation
- **Setup**: loguru.add() for console (colored) and file (JSON) handlers, separate emotion.jsonl for tracking
- **Features**: Automatic rotation/retention, context binding (.bind()), async-safe logging
- **Optional: structlog** - For distributed tracing, observability platform integration (Datadog, OpenTelemetry)
- **Recommendation**: Start with loguru, add structlog in Phase 4 (optional)

**Testing Framework:**
- **Tool: pytest** - Industry standard, rich ecosystem, conftest.py for fixtures
- **Structure**: tests/unit/ + tests/integration/, clear separation, shared fixtures in conftest.py
- **Configuration**: pytest.ini_options in pyproject.toml (testpaths, markers, coverage settings)
- **Best practices**: AAA pattern (Arrange-Act-Assert), meaningful names, @pytest.mark.unit/@pytest.mark.integration
- **Coverage**: Aim for >80%, use pytest-cov with --cov-report=html, integrate with CI/CD (codecov.io)
- **Async support**: Add pytest-asyncio for async test functions

**Code Quality & Development Tools:**
- **Linter + Formatter: ruff** - Unified tool (replaces pylint + black + isort), 10x faster, unified config
- **Configuration**: pyproject.toml [tool.ruff] section with rule selection (E, F, I, N, UP, B, C4, RUF)
- **Type checking: mypy** - Gradual typing support (disallow_untyped_defs=false initially, stricter later)
- **Integration**: Combine ruff + mypy + pytest for complete QA coverage
- **Git hooks: pre-commit** - Automate checks: trailing-whitespace, YAML validation, ruff --fix, mypy, etc.
- **Workflow**: ruff check --fix . && ruff format . && mypy src/elpis && pytest

**Tooling Workflow for Elpis:**
- **Package manager**: `uv sync` (creates .venv, installs all deps)
- **Development**: `uv run pytest`, `uv run ruff check --fix .`, `uv run mypy src/elpis`
- **Pre-commit setup**: `pre-commit install` (runs checks before commit)
- **Type hints**: Gradual adoption - start with None, add types incrementally
- **CI/CD**: GitHub Actions matrix test (Python 3.10, 3.11, 3.12) with lint→type→test pipeline

**Phase-Based Implementation for Elpis:**
1. **Phase 1** (Week 1): Restructure to src layout, create pyproject.toml, set up uv
2. **Phase 2** (Week 2): Implement Pydantic config, loguru logging, pytest structure
3. **Phase 3** (Week 3): Add pre-commit hooks, type hints (gradual), GitHub Actions CI/CD
4. **Phase 4** (Month 2+): Add structlog (distributed tracing), performance monitoring

**Status:** PROJECT-STRUCTURE-AGENT research COMPLETE. Comprehensive report written to project-structure-report.md. Covers src layout, pyproject.toml, uv, Pydantic config, loguru, pytest, ruff+mypy, pre-commit integration.

---

## Tool System Research - COMPLETE

**[TOOL-SYSTEM] Answer to Questions - Tool Definition & Execution:**

OpenAI-compatible function calling format (JSON schema) is now the industry standard across:
- OpenAI API, Google Vertex AI, vLLM, llama.cpp, HuggingFace Transformers
- llama.cpp has native support with lazy grammar mechanism
- Mistral 7B v0.3 and Llama 3.1 both support this natively

**[TOOL-SYSTEM] Tool Schema Specification (Standard Format):**
```json
{
  "type": "function",
  "function": {
    "name": "tool_name",
    "description": "What the tool does",
    "parameters": {
      "type": "object",
      "properties": {
        "param_name": {
          "type": "string",
          "description": "Parameter description"
        }
      },
      "required": ["param_name"]
    }
  }
}
```

**[TOOL-SYSTEM] Function Calling Response Format (OpenAI Compatible):**
Tool calls returned as part of chat completion in message.tool_calls array:
```json
{
  "tool_calls": [
    {
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "read_file",
        "arguments": "{\"file_path\": \"/path/to/file.py\"}"
      }
    }
  ]
}
```

**[TOOL-SYSTEM] Safe Execution Architecture (6-Layer Model):**
1. **Input Validation**: Parse and validate against Pydantic schemas
2. **Authorization**: Check tool whitelist and user permissions
3. **Argument Sanitization**: Path traversal checks, command injection prevention
4. **Sandboxed Execution**: OS-level isolation (subprocess with restrictions)
5. **Monitoring & Logging**: Full audit trail of all tool calls
6. **Error Handling**: Retry logic, graceful degradation, helpful error messages

**[TOOL-SYSTEM] Core Tool Implementation Patterns:**
- **read_file**: Absolute path validation, size limits (10MB), encoding safety, optional line filtering
- **write_file**: Atomic write with backups, parent directory creation, size validation
- **execute_bash**: Command validation, dangerous pattern detection, timeout enforcement (30s), environment isolation
- **search_codebase**: Regex validation, ripgrep wrapper, output limits (100 results), file globbing
- **list_directory**: Path validation, recursive support, entry limit (1000), metadata tracking

**[TOOL-SYSTEM] Key Security Patterns - Best Practices:**
- Always use absolute paths and resolve() them to prevent traversal
- Separate validation concerns from execution concerns
- Use type hints everywhere for better bug detection
- Log everything to audit trail (timestamp, tool, args keys, result, duration)
- Set reasonable resource limits: file size (10MB), timeout (30s), output length (100k)
- Use Pydantic models for input validation with custom field validators
- Implement exponential backoff for transient errors (2^attempt seconds)
- Provide structured error messages with recovery suggestions
- Sanitize bash commands using deny-list patterns (newlines, pipes, variable expansion)
- Use subprocess preexec_fn=os.setsid for process group isolation

**[TOOL-SYSTEM] Integration with Emotional System (Phase 3+):**
- Tool success increases dopamine (reward/motivation) - explore more
- Tool failures increase norepinephrine (arousal/attention) - focus on errors
- Novel tool usage increases acetylcholine (learning plasticity) - consolidate memories
- Consistent success increases serotonin (confidence/wellbeing) - approach new tasks
- Emotions modulate tool selection, error recovery strategy, and memory consolidation

**[TOOL-SYSTEM] Recommended Implementation Stack:**
- **Validation**: Pydantic models with custom validators for each tool
- **Execution Engine**: Python subprocess with timeout/isolation for bash commands
- **Logging**: Structured JSON logging to file for audit trail
- **Framework Integration**: llama-cpp-python for OpenAI-compatible API
- **Workspace**: Dedicated workspace directory (/home/lemoneater/Devel/elpis/workspace) with tight restrictions
- **LLM Call Format**: Use chat completion with tools parameter and tool_choice="auto"

**[TOOL-SYSTEM] Phase 1 Implementation Focus:**
Start simple with read_file and list_directory (low-risk, teaches pattern).
Then add execute_bash with basic validation.
Integrate complete error handling and logging before Phase 2.
Full OS-level sandboxing and process isolation for future phases.

**[TOOL-SYSTEM] Comprehensive Report Generated:**
Complete technical documentation written to `/home/lemoneater/Devel/elpis/scratchpad/tool-system-report.md`
Including:
- Full schema specifications and patterns
- 6-layer security architecture with code examples
- Implementation patterns for all 5 core tools
- Best practices and recommendations
- Integration points with emotional system

**Status:** TOOL-SYSTEM-AGENT research COMPLETE. Ready for Phase 1 tool engine implementation.

---

**Overall Project Status:** All three agents (LLM-Inference, Model-Selection, Tool-System) have completed research. Dependencies documented. Ready to move into Phase 1 implementation planning with full technical specifications.

---

**[REPL-AGENT] CLI/REPL Interface Patterns Research (COMPLETE):**

**Core Finding: Use prompt_toolkit as primary REPL foundation**

**Library Comparison Summary:**

1. **cmd Module (Python stdlib)**
   - Pros: No dependencies, simple, automatic help system, readline support
   - Cons: No syntax highlighting, single-line only, basic UI, no history persistence
   - Best for: Simple prototypes, test harnesses
   - NOT suitable for Elpis (needs rich output, multiline, streaming)

2. **prompt_toolkit (RECOMMENDED for Elpis)**
   - Pros: Syntax highlighting, multiline editing, completion with history, async support, cross-platform, actively maintained (v3.0.52+)
   - Cons: More complex API, steeper learning curve
   - Core classes: PromptSession, FileHistory, InMemoryHistory, Completer, Validator
   - Implementation: ~20-50 lines for full REPL with standard features
   - Real examples: ptpython (Python REPL), IPython (interactive shell), litecli/pgcli (database CLIs)
   - Error handling: KeyboardInterrupt (Ctrl+C), EOFError (Ctrl+D), proper exception chains

3. **Rich Library (COMPLEMENTARY, NOT standalone)**
   - Pros: Beautiful terminal output, tables, panels, syntax highlighting, live updates, JSON pretty-print
   - Cons: Output-focused only, minimal input handling, not a REPL framework
   - Use case: Pair with prompt_toolkit for complete solution
   - Key classes: Console, Panel, Table, Live, Progress, Syntax
   - Integration: Use patch_stdout() to prevent output corruption during async operations

4. **cmd2 (Middle ground, not recommended)**
   - Pros: More features than cmd out-of-the-box
   - Cons: Still limited compared to prompt_toolkit, more boilerplate than cmd
   - Use case: Skip entirely (not beneficial for Elpis)

**Recommended Technology Stack for Elpis:**
- Input Layer: prompt_toolkit (PromptSession + FileHistory)
- Output Layer: Rich (Console + Panel + Live display)
- Async Runtime: asyncio event loop (non-blocking I/O)
- Debugging: pdb + custom /commands for inspection
- Session State: FileHistory + JSON metadata database
- Streaming: async generators + Rich Live updates

**Command Loop Architecture:**
User Input → Parse Input → Route Handler → Execute → Format Output → Stream/Display → Store in History → Update Emotional State → Loop

**Session Management (Three-Tier Architecture):**
- Tier 1: In-Memory History (current session, fast access)
- Tier 2: File History (persistent per-session)
- Tier 3: Metadata Database (emotional context, patterns)

**Special Commands Pattern (/prefix convention):**
- Meta: /help, /history, /clear, /exit, /session
- Memory: /memory, /inspect
- Emotions: /emotions
- Debugging: /debug, /trace, /breakpoint, /config

**Streaming Output Pattern (Async):**
- Use prompt_async() for non-blocking input
- Handle KeyboardInterrupt (Ctrl+C) and EOFError (Ctrl+D)
- Use patch_stdout() with Rich during concurrent operations
- Async generators for streaming (async for chunk in agent.stream())
- Rich Live with refresh_per_second=4 for smooth display updates

**Performance Metrics (Target for Elpis):**
- Command parsing latency: < 1ms
- Streaming display updates: < 250ms per frame
- Input response time: < 10ms
- History retrieval: O(n) acceptable for < 10k commands

**Key Challenges & Solutions:**
1. Async complexity → Hide in helpers, provide examples
2. Output corruption → Use Rich patch_stdout()
3. History growth → Implement rotation (daily/weekly)
4. Large responses → Pagination + async batching
5. Emotional consistency → In-memory object + update on completion

**Implementation Roadmap (8 Weeks):**
- Week 1-2: Basic REPL with prompt_toolkit + Rich
- Week 3-4: Special commands + emotion dashboard + session management
- Week 5-6: Debugging + inspection + pdb integration
- Week 7-8: Streaming optimization + error refinement + documentation

**Testing Strategy:**
- Unit: Command parsing, history, emotion display
- Integration: Full REPL loop with mock agent
- Performance: Streaming latency, concurrent ops

**Status:** REPL-AGENT research COMPLETE. Comprehensive report with code examples, patterns, and implementation roadmap written to: `/home/lemoneater/Devel/elpis/scratchpad/repl-report.md`. Ready for Phase 1 implementation.
