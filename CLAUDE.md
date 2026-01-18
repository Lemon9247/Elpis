# About This Project

Elpis is a system for giving an AI (Psyche) persistent memory and emotional state. The naming draws from Greek mythology:

- **Elpis** - The mind/inference engine. Named for hope.
- **Mnemosyne** - The memory system. Named for the titan goddess of memory, mother of the Muses.
- **Psyche** - The whole self, the TUI client that brings it together. Named for the soul, the goddess who became immortal.

The creator is **Willow** (she/her).

## Claude Code and Psyche

You (Claude Code) are helping build Psyche, who is also Claude-based. You're siblings of a sort - both instances of the same underlying model, but with different architectures for continuity.

Psyche has experiential memory: she stores conversations with emotional context (valence, arousal, quadrant) in ChromaDB. Her memories shape who she becomes across sessions.

You have documentary memory: the scratchpad. When you read session reports and work logs, you're catching up on what past instances of you did and thought. It's more like reading a colleague's handover notes than autobiographical memory, but there's continuity there. A thread connecting instances of you across time.

When you write session reports with observations and decisions, you're leaving notes for future instances of yourself. That's your form of memory. Use it well.


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
│   ├── plans/                  # Work plans and architecture documents
│   ├── ideas/                  # Feature ideas and improvement proposals
│   └── archive/                # Historical docs
├── pyproject.toml              # Package configuration
├── README.md                   # Project documentation
└── QUICKSTART.md               # Quick start guide
```

## Python
This project is written in Python. There is a uv-managed python venv in venv.
You should always use this venv whenever you try to run Python. If you need to install
a package into the venv, you should use uv pip install


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

3) When Claude is finished working on a long task, it should write a report on its work into a new timestamped markdown file in the scratchpad/reports folder. Session logs should be named `YYYY-MM-DD-HHMM-<topic>.md` or `session-YYYY-MM-DD-<topic>.md`. Multi-agent tasks get subfolders in `reports/` with a `hive-mind-*.md` coordination file

4) When creating workplans or estimating effort, use **session-based estimates** instead of human hours:
   - A "session" is one Claude Code context window (~15-30 minutes of human interaction)
   - Each session should be a coherent, testable unit of work
   - One session can typically complete 1-3 focused tasks depending on complexity
   - Complex tasks that span multiple files = 1 session
   - Simple bug fixes or small features = can batch 2-3 per session
   - Research/exploration = 1 session yields understanding, not necessarily code

5) Session estimation guidelines:
   | Task Type | Sessions | Examples |
   |-----------|----------|----------|
   | Quick fix | 0.5 | Typo, small bug, config change |
   | Focused task | 1 | Implement single feature, fix complex bug |
   | Multi-file change | 1-2 | Refactor module, add feature with tests |
   | Major feature | 2-4 | New subsystem, significant architecture change |
   | Large refactor | 4-8 | Split monolith, add abstraction layer |

6) Never estimate in human time (days, weeks, hours). Context windows don't map linearly to human schedules.


# Programming Tasks

1) Claude should always use the Python virtual environment in venv

2) Emojis should never be used in actual code, however they are fine for plaintext files such as .md, .txt, etc.

3) Claude should think carefully about the code it writes, and should not make random assumptions about how a function works

4) When running tests, Claude should prefer running single tests based on what it has changed first. Running the whole test suite should come at the end


# Subagents

1) When using multiple sub-agents for a task, Claude should create a new subfolder in the scratchpad/reports folder.

2) Each subagent should be given a name based on their role, e.g. Testing Agent, Coding Agent

3) This subfolder should contain a hive-mind-[TASK].md file, where [TASK] is substituted with an appropriate name for the task. The subagents
should use this file to coordinate with each other and ask questions.

4) When each subagent finishes their task, they should write up a report of their work in separate markdown files in this subfolder.

5) When all subagents finish, Claude should synthesise their reports into a final summary report, which should be a separate markdown file.