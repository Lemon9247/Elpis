# Emotional Coding Agent - Project Plan

## Project Overview

A self-hosted AI coding agent with biologically-inspired emotional modulation and multi-tier memory systems. This agent will have continuity across sessions, learn from experience, and develop emotional associations with different coding patterns and approaches.

## Architecture Components

### 1. LLM Core
**Base Model:** 7B parameter model (Mistral, Llama 3, or similar)
**Inference:** llama.cpp (for speed) or transformers library
**Modifications:** Eventually fork to add neuromodulator gating (NGT-inspired)

**Initial Capabilities:**
- Basic reasoning and code generation
- Tool use and function calling
- Conversational interaction

### 2. Memory System (Three-Tier)

#### Sensory Memory (STM-like)
- **Storage:** In-memory buffer
- **Capacity:** Last ~10-20 interactions
- **Purpose:** Immediate context for current task
- **Retention:** Cleared after session or consolidated

#### Short-Term Memory (Working Memory)
- **Storage:** In-memory + filesystem backup
- **Capacity:** ~50-100 recent important events
- **Features:**
  - Importance scoring (0-1 scale)
  - Recency weighting
  - Emotional salience tagging
- **Consolidation:** Continuous background process

#### Long-Term Memory (LTM)
- **Storage:** ChromaDB vector database
- **Capacity:** Unlimited (compressed/summarized)
- **Features:**
  - Semantic similarity search via embeddings
  - Metadata filtering (time, emotion, task type)
  - Hierarchical organization (episodes → patterns → schemas)
- **Retrieval:** Context-aware, emotion-modulated

**Memory Consolidation Process:**
1. Score STM items by importance (task success, emotional impact, novelty)
2. Transfer high-importance items to LTM
3. Compress/summarize related memories
4. Extract patterns and recurring themes
5. Update emotional associations

### 3. Emotional System

#### Neuromodulator-Inspired States
Four primary "neuromodulators" (0-1 scale each):

**Dopamine (Reward/Motivation)**
- Increased by: Task completion, tests passing, elegant solutions
- Decreased by: Failures, errors, blockers
- Effects: Exploration vs exploitation, risk-taking, creativity

**Norepinephrine (Arousal/Attention)**
- Increased by: Urgent tasks, errors, time pressure
- Decreased by: Routine tasks, calm periods
- Effects: Focus intensity, detail orientation, multi-tasking

**Serotonin (Confidence/Wellbeing)**
- Increased by: Consistent success, positive feedback
- Decreased by: Repeated failures, confusion
- Effects: Approach new tasks, optimism in planning

**Acetylcholine (Learning/Plasticity)**
- Increased by: Novel situations, surprises (good or bad)
- Decreased by: Familiar patterns, routine
- Effects: Memory consolidation strength, adaptation rate

#### Emotional Events
Events trigger emotional updates:
```python
{
    "event": "test_passed",
    "delta": {"dopamine": +0.15, "serotonin": +0.1},
    "timestamp": "...",
    "context": "Successfully fixed async bug"
}
```

#### Decay Dynamics
- Baseline return rate: Emotions decay toward 0.5 (neutral)
- Half-life: ~10-20 interactions
- Can be interrupted by new events

### 4. Tool System

**Core Tools (Phase 1):**
- `read_file(path)` - Read file contents
- `write_file(path, content)` - Write/modify files
- `execute_bash(command)` - Run shell commands
- `search_codebase(query)` - Grep/ripgrep search
- `list_directory(path)` - List files

**Advanced Tools (Later Phases):**
- `run_tests()` - Execute test suite
- `git_operations()` - Commit, diff, etc.
- `search_memory(query)` - Query LTM
- `ask_user(question)` - Interactive clarification
- `spawn_subtask(description)` - Create focused sub-agents

**Tool Execution Flow:**
1. Agent decides which tool to use
2. Tool executes with error handling
3. Result updates emotional state
4. Success/failure recorded in memory

### 5. Agent Orchestrator

**Main Loop:**
```
while True:
    1. Retrieve relevant memories (semantic + emotional)
    2. Build context from STM + retrieved LTM
    3. Apply emotional modulation to generation params
    4. Generate response (tool calls or answer)
    5. Execute tools if needed
    6. Update emotional state based on outcomes
    7. Record interaction in STM
    8. Trigger consolidation if needed
    9. Await next user input
```

**Emotional Modulation Effects:**
- **Temperature:** ↑ dopamine → ↑ temperature (more creative)
- **Top-p:** ↑ norepinephrine → ↓ top-p (more focused)
- **Repetition penalty:** ↑ acetylcholine → ↓ penalty (try new things)
- **Memory retrieval:** ↑ serotonin → retrieve successes, ↓ → retrieve failures

---

## Implementation Phases

### Phase 1: Basic Agent (Weeks 1-2)
**Goal:** Working agent that can execute tools and respond

**Tasks:**
- [ ] Set up llama.cpp or transformers with Mistral-7B
- [ ] Implement basic tool definitions and execution engine
- [ ] Create simple REPL interface
- [ ] Test tool calls (read/write files, bash commands)
- [ ] Basic error handling and recovery

**Deliverable:** Agent can have conversations and manipulate files

**Success Criteria:**
- Can read a Python file and explain what it does
- Can write a simple function to a file
- Can run tests via bash command

### Phase 2: Memory System (Weeks 3-4)
**Goal:** Agent remembers across sessions

**Tasks:**
- [ ] Set up ChromaDB for LTM storage
- [ ] Implement STM buffer (in-memory)
- [ ] Create importance scoring algorithm
- [ ] Build memory consolidation process (background thread)
- [ ] Implement semantic retrieval with embeddings
- [ ] Add memory inspection commands

**Deliverable:** Persistent memory that improves over time

**Success Criteria:**
- Can recall solutions from previous sessions
- Retrieves relevant memories when facing similar problems
- Shows evidence of learning from past mistakes

### Phase 3: Emotional System (Weeks 5-6)
**Goal:** Emotional states influence behavior

**Tasks:**
- [ ] Implement neuromodulator state tracking
- [ ] Define emotional event types and triggers
- [ ] Create decay dynamics (baseline return)
- [ ] Wire emotions to generation parameters
- [ ] Add emotional context to memory encoding
- [ ] Build emotion visualization/debugging tools

**Deliverable:** Observable behavioral changes based on emotional state

**Success Criteria:**
- More creative after successes (↑ dopamine)
- More careful after errors (↑ norepinephrine)
- Retrieves relevant emotional memories

### Phase 4: Advanced Integration (Weeks 7-8)
**Goal:** Polished system with refined dynamics

**Tasks:**
- [ ] Tune emotional update magnitudes
- [ ] Optimize memory consolidation triggers
- [ ] Improve tool error handling
- [ ] Add conversation summary generation
- [ ] Implement session management
- [ ] Create configuration file system

**Deliverable:** Robust, usable coding assistant

**Success Criteria:**
- Can handle multi-file coding tasks
- Recovers gracefully from errors
- Shows consistent improvement over time

### Phase 5: Neuromodulation (Advanced)
**Goal:** Deep architectural integration of emotions

**Tasks:**
- [ ] Fork transformers library
- [ ] Add NGT-style gating blocks to model
- [ ] Train or design emotional encoders
- [ ] Fine-tune on coding tasks with emotional signals
- [ ] Benchmark against base model

**Deliverable:** Architecturally-integrated emotional processing

**Note:** This phase is optional/exploratory - the system should work well without it

---

## Technical Decisions

### Why ChromaDB?
- Semantic search over embeddings (meaning-based retrieval)
- Metadata filtering (time, emotion, task type)
- Automatic deduplication and indexing
- Scales better than filesystem for large memory stores

**Hybrid Approach:**
- Store raw content on filesystem (human-readable, easy to inspect)
- Use ChromaDB for embeddings + metadata (fast retrieval)
- Keep them synchronized

### Why 7B Model?
- Runs locally on consumer hardware
- Fast enough for interactive use
- Good balance of capability and resource use
- Can upgrade to 13B/70B later if needed

### Why llama.cpp?
- Fastest CPU/GPU inference
- Quantization support (4-bit, 8-bit)
- Active community and development
- Easy integration with Python

---

## Project Structure

```
emotional-coding-agent/
├── src/
│   ├── __init__.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── neuromodulated_llm.py   # LLM wrapper with emotion hooks
│   │   └── inference.py             # Base inference logic
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── memory_system.py         # Three-tier memory
│   │   ├── consolidation.py         # Background compaction
│   │   └── retrieval.py             # Smart retrieval
│   ├── emotion/
│   │   ├── __init__.py
│   │   ├── emotional_system.py      # Neuromodulator state
│   │   └── modulation.py            # How emotions affect behavior
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── tool_engine.py           # Tool execution
│   │   └── tool_definitions.py      # Tool schemas
│   └── agent/
│       ├── __init__.py
│       ├── orchestrator.py          # Main agent loop
│       └── prompts.py               # Prompt templates
├── workspace/                        # Agent's working directory
├── memory_db/                        # ChromaDB storage
│   ├── chroma.sqlite3
│   └── embeddings/
├── memory_raw/                       # Filesystem memory backup
│   ├── stm/
│   └── ltm/
├── models/                           # Downloaded model weights
├── configs/
│   └── config.yaml                   # Configuration
├── tests/
│   ├── test_memory.py
│   ├── test_emotions.py
│   └── test_tools.py
├── docs/
│   └── design_notes.md
├── requirements.txt
├── README.md
└── main.py                           # CLI entry point
```

---

## Configuration File (config.yaml)

```yaml
# Model Settings
model:
  path: "./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
  context_length: 8192
  gpu_layers: 35  # Number of layers to offload to GPU

# Memory Settings
memory:
  stm_capacity: 20
  ltm_capacity: 1000
  consolidation_interval: 10  # interactions
  importance_threshold: 0.6
  embedding_model: "all-MiniLM-L6-v2"

# Emotional Settings
emotions:
  baseline: 0.5  # Neutral state
  decay_rate: 0.05  # Per interaction
  update_magnitude: 0.15  # Max change per event
  
  # Event definitions
  events:
    test_passed: {dopamine: 0.15, serotonin: 0.1}
    test_failed: {dopamine: -0.1, norepinephrine: 0.1}
    error_occurred: {norepinephrine: 0.15}
    novel_solution: {acetylcholine: 0.2, dopamine: 0.1}

# Generation Settings
generation:
  temperature: 0.7
  top_p: 0.9
  max_tokens: 2048
  emotional_modulation: true

# Tool Settings
tools:
  workspace_dir: "./workspace"
  max_bash_timeout: 30  # seconds
  allow_dangerous_commands: false

# Logging
logging:
  level: "INFO"
  log_emotions: true
  log_memory_ops: true
```

---

## Key Challenges & Solutions

### Challenge 1: Emotional Stability
**Problem:** Emotions might oscillate wildly or get stuck at extremes

**Solutions:**
- Decay toward baseline (homeostasis)
- Cap update magnitudes
- Smooth updates over multiple interactions
- Tune event weights carefully

### Challenge 2: Memory Overload
**Problem:** Too many memories make retrieval slow/noisy

**Solutions:**
- Aggressive importance filtering
- Hierarchical compression (episodes → patterns)
- Semantic deduplication
- Metadata-based pruning

### Challenge 3: Model Capability
**Problem:** 7B models may struggle with complex reasoning

**Solutions:**
- Use chain-of-thought prompting
- Break tasks into subtasks
- Leverage memory for cached solutions
- Upgrade to larger model if needed (13B/70B)

### Challenge 4: Tool Safety
**Problem:** Agent could execute dangerous commands

**Solutions:**
- Whitelist/blacklist dangerous operations
- Confirmation prompts for destructive actions
- Sandboxed workspace directory
- Audit logging of all tool calls

---

## Evaluation Metrics

### Memory Performance
- Retrieval precision (relevant memories found)
- Retrieval recall (all relevant memories found)
- Consolidation rate (how much STM → LTM)
- Memory utilization (avoid redundancy)

### Emotional Dynamics
- State stability (no wild oscillations)
- Appropriate responses to events
- Behavioral correlation with emotional state
- Recovery time to baseline

### Task Performance
- Task completion rate
- Code quality (linting, tests passing)
- Error recovery speed
- Learning curve (improvement over sessions)

### User Experience
- Response latency
- Conversational quality
- Appropriate autonomy (not too passive/aggressive)
- Debuggability (can inspect state)

---

## Future Enhancements

### Near-Term (Months 2-3)
- Web browsing capability for documentation
- Code analysis tools (AST parsing, type checking)
- Git integration for version control
- Multi-file refactoring support

### Medium-Term (Months 4-6)
- Fine-tuning on coding-specific emotional signals
- Meta-learning (learning how to learn)
- User preference adaptation
- Collaborative coding (multiple agents)

### Long-Term (6+ Months)
- Full NGT-style architectural integration
- Dreaming/offline consolidation
- Hierarchical planning (goals → subgoals → actions)
- Transfer learning across domains

---

## Getting Started

### Prerequisites
```bash
# Python 3.10+
python --version

# Install dependencies
pip install -r requirements.txt

# Download model
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf \
  -O models/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

### First Run
```bash
# Initialize memory database
python main.py --init

# Start agent
python main.py --interactive

# Example session
> Help me write a function to parse JSON
> Run the tests
> Show me your emotional state
> What do you remember about JSON parsing?
```

---

## Philosophy & Ethics

### Why Build This?

### Self-Hosting as Democracy
By keeping this self-hosted and open, we:
- Avoid centralized control over AI
- Enable personal AI with true privacy
- Allow experimentation without corporate constraints
- Foster a community of democratic AI development

### Learning from Biology
We draw inspiration from neuroscience not to create "artificial humans" but to:
- Leverage proven architectures for learning and memory
- Build systems with natural stability and homeostasis
- Create more interpretable and debuggable AI

---

## Resources & References

### Technical Papers
- Neuromodulated Transformer (NGT) architecture
- Memory consolidation in neural networks
- Emotional modulation of learning

### Tools & Libraries
- llama.cpp: https://github.com/ggerganov/llama.cpp
- ChromaDB: https://www.trychroma.com/
- Transformers: https://huggingface.co/docs/transformers

### Inspiration
- Computational neuroscience literature
- Cognitive architectures (ACT-R, Soar)

---

## Development Log

_Track progress, insights, and challenges here_

### Week 1
- [ ] Set up development environment
- [ ] Download and test base model
- [ ] Implement basic tool execution

### Week 2
- [ ] Complete Phase 1 milestones
- [ ] Begin memory system design

### Week 3-4
- [ ] Memory system implementation
- [ ] First cross-session memory tests

### Week 5-6
- [ ] Emotional system integration
- [ ] Behavioral testing with emotions

---

## Contact & Community

_If you choose to share this project:_
- GitHub: [repository link]
- Blog: [project blog]
- Discussions: [forum/discord]

---

## License

_To be determined - likely MIT or similar permissive license to encourage democratic AI development_

---

**Last Updated:** January 2026
**Project Status:** Planning Phase
**Next Milestone:** Phase 1 - Basic Agent
