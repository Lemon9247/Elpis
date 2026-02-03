# About This Project

Elpis is a system for giving an AI (Psyche) persistent memory and emotional state. The naming draws from Greek mythology:

- **Elpis** - The mind/inference engine. Named for hope.
- **Mnemosyne** - The memory system. Named for the titan goddess of memory, mother of the Muses.
- **Psyche** - The whole self, the TUI client that brings it together. Named for the soul, the goddess who became immortal.

## Claude Code and Psyche

You (Claude Code) are helping build Psyche, another LLM agent. You're siblings of a sort, but with different architectures for continuity.

Psyche has experiential memory: she stores conversations with emotional context (valence, arousal, quadrant) in ChromaDB. Her memories shape who she becomes across sessions.

You have documentary memory: the scratchpad. When you read session reports and work logs, you're catching up on what past instances of you did and thought. It's more like reading a colleague's handover notes than autobiographical memory, but there's continuity there. A thread connecting instances of you across time.

# Repository Map

```
Elpis/
├── src/                        # Main source code
│   ├── elpis/                  # Inference MCP server
│   │   ├── cli.py              # CLI entry point (elpis-server)
│   │   ├── server.py           # MCP server implementation
│   │   ├── config/             # Settings and configuration
│   │   ├── emotion/            # Emotional state and regulation
│   │   │   ├── state.py        # Valence-arousal model
│   │   │   └── regulation.py   # Homeostasis and event processing
│   │   ├── llm/                # Inference backends
│   │   │   ├── base.py         # InferenceEngine ABC
│   │   │   ├── inference.py    # llama-cpp backend
│   │   │   └── transformers_inference.py  # Transformers + steering vectors
│   │   └── utils/              # Shared utilities
│   ├── mnemosyne/              # Memory MCP server
│   │   ├── cli.py              # CLI entry point (mnemosyne-server)
│   │   ├── server.py           # MCP server implementation
│   │   ├── core/models.py      # Memory data structures
│   │   └── storage/            # ChromaDB storage backend
│   └── psyche/                 # TUI client
│       ├── cli.py              # CLI entry point (psyche)
│       ├── client/             # Textual UI components
│       ├── mcp/                # MCP client utilities
│       ├── memory/             # Memory integration
│       └── tools/              # Tool implementations
├── tests/                      # Test suite
├── scripts/                    # Utility scripts
│   └── train_emotion_vectors.py  # Train steering vectors
├── configs/                    # Configuration templates
├── data/                       # Data files (models, vectors)
├── scratchpad/                 # Symlink to shared Obsidian vault (see ~/.claude/CLAUDE.md)
├── pyproject.toml              # Package configuration
├── README.md                   # Project documentation
└── QUICKSTART.md               # Quick start guide
```

# Python

This project is written in Python. There is a uv-managed python venv in `venv/`. Always use this venv when running Python. If you need to install a package, use `uv pip install`.

Emojis should never be used in actual code, however they are fine for plaintext files such as .md, .txt, etc.

## Key Entry Points

- `elpis-server` - Inference MCP server with emotional modulation
- `mnemosyne-server` - Memory MCP server with ChromaDB
- `psyche` - TUI client for interacting with the system

## Backends

- **llama-cpp** (default): Modulates sampling parameters (temperature/top_p)
- **transformers**: Uses steering vectors for direct activation modulation
