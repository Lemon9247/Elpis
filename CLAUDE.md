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
├── scratchpad/                 # Working notes and session logs
│   ├── reports/                # Session reports
│   └── archive/                # Historical docs
├── pyproject.toml              # Package configuration
├── README.md                   # Project documentation
└── QUICKSTART.md               # Quick start guide
```

## Key Entry Points

- `elpis-server` - Inference MCP server with emotional modulation
- `mnemosyne-server` - Memory MCP server with ChromaDB
- `psyche` - TUI client for interacting with the system

## Backends

- **llama-cpp** (default): Modulates sampling parameters (temperature/top_p)
- **transformers**: Uses steering vectors for direct activation modulation


# Work Planning

1) All project notes, work logs and reports can be found in the scratchpad folder

2) When Claude first starts, it should review the latest work on the project by reviewing the git history and anything recent in the scratchpad

3) When Claude is finished working on a long task, it should write a report on its work into a new timestamped markdown file in the scratchpad/reports folder


# Programming Tasks

1) Claude should always use the Python virtual environment in venv

2) Emojis should never be used in actual code, however they are fine for plaintext files such as .md, .txt, etc.

3) Claude should think carefully about the code it writes, and should not make random assumptions about how a function works

4) When running tests, Claude should prefer running single tests based on what it has changed first. Running the whole test suite should come at the end


# Subagents

1) When using multiple sub-agents for a task, Claude should create a new subfolder in the scratchpad folder.

2) Each subagent should be given a name based on their role, e.g. Testing Agent, Coding Agent

3) This subfolder should contain a hive-mind-[TASK].md file, where [TASK] is substituted with an appropriate name for the task. The subagents
should use this file to coordinate with each other and ask questions.

4) When each subagent finishes their task, they should write up a report of their work in separate markdown files in this subfolder.

5) When all subagents finish, Claude should synthesise their reports into a final summary report, which should be a separate markdown file.