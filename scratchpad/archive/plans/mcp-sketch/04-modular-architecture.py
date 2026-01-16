"""
Modular Elpis Architecture
==========================

Elpis as a set of composable MCP servers that any agent harness can use.

Design Goals:
- Each component is an independent MCP server
- Components can be used together or separately
- Any MCP-compatible harness (Psyche, opencode, aider, etc) can use them
- No tight coupling between inference, emotion, and memory
- Graceful degradation (works without optional components)

Architecture:

    ┌─────────────────────────────────────────────────────────────┐
    │           Agent Harness (any MCP client)                    │
    │      (Psyche, opencode, aider, Claude Code, custom)         │
    └─────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                    MCP Protocol Layer                        │
    └─────────────────────────────────────────────────────────────┘
            │                   │                   │
            ▼                   ▼                   ▼
    ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
    │    elpis      │   │    mnemosyne  │   │    other      │
    │   (emotion +  │   │   (memory)    │   │   MCP tools   │
    │   inference)  │   │               │   │               │
    └───────────────┘   └───────────────┘   └───────────────┘
            │                   │
            ▼                   ▼
    ┌───────────────┐   ┌───────────────┐
    │  Local LLM    │   │   ChromaDB    │
    │ (transformers)│   │               │
    └───────────────┘   └───────────────┘

Components:
1. elpis-inference  - Emotional LLM inference (can work standalone)
2. mnemosyne        - Memory system (can work with any LLM backend)
3. psyche           - Reference harness that uses both (optional)

"""

# =============================================================================
# 1. ELPIS INFERENCE SERVER (elpis-inference)
# =============================================================================

"""
elpis-inference: Emotional LLM inference as an MCP server

This is the core Elpis offering - local LLM inference with emotional
steering vectors. It exposes a simple MCP interface that any client can use.

Key Design Decisions:
- Emotion state lives IN the server (not the client)
- Clients send events, server manages state + homeostasis
- Inference automatically uses current emotional state
- Clients can query state but don't need to track it
- Works with ANY MCP client, no special integration needed
"""

# ----- MCP Tool Definitions -----

ELPIS_INFERENCE_TOOLS = """
Tools exposed by elpis-inference MCP server:

┌─────────────────────────────────────────────────────────────────┐
│ INFERENCE TOOLS                                                 │
├─────────────────────────────────────────────────────────────────┤
│ generate                                                        │
│   Generate text with emotional modulation                       │
│   Params:                                                       │
│     - messages: List[{role, content}]                          │
│     - max_tokens: int (optional)                               │
│     - temperature: float (optional, overrides emotion)         │
│     - top_p: float (optional, overrides emotion)               │
│   Returns: {content: str, emotion: EmotionState}               │
│                                                                 │
│ generate_stream                                                 │
│   Streaming generation with emotional modulation               │
│   Same params as generate, streams tokens                      │
│                                                                 │
│ function_call                                                   │
│   Generate tool/function calls                                 │
│   Params:                                                       │
│     - messages: List[{role, content}]                          │
│     - tools: List[ToolDefinition]                              │
│   Returns: {tool_calls: List[ToolCall], emotion: EmotionState} │
├─────────────────────────────────────────────────────────────────┤
│ EMOTION TOOLS                                                   │
├─────────────────────────────────────────────────────────────────┤
│ emotion_event                                                   │
│   Trigger an emotional event                                   │
│   Params:                                                       │
│     - event_type: str (success, failure, novelty, etc)         │
│     - intensity: float (0.0 to 2.0, default 1.0)              │
│   Returns: {emotion: EmotionState}                             │
│                                                                 │
│ emotion_get                                                     │
│   Get current emotional state                                  │
│   Returns: {emotion: EmotionState}                             │
│                                                                 │
│ emotion_reset                                                   │
│   Reset to baseline emotional state                            │
│   Returns: {emotion: EmotionState}                             │
│                                                                 │
│ emotion_set_baseline                                            │
│   Set the baseline/personality (optional)                      │
│   Params:                                                       │
│     - valence: float (-1 to 1)                                 │
│     - arousal: float (-1 to 1)                                 │
│   Returns: {emotion: EmotionState}                             │
├─────────────────────────────────────────────────────────────────┤
│ RESOURCES                                                       │
├─────────────────────────────────────────────────────────────────┤
│ emotion://state      - Current emotional state                 │
│ emotion://events     - Available event types                   │
│ emotion://config     - Current configuration                   │
└─────────────────────────────────────────────────────────────────┘

EmotionState schema:
{
    "valence": float,      // -1.0 to 1.0
    "arousal": float,      // -1.0 to 1.0
    "quadrant": str,       // "excited", "frustrated", "calm", "depleted"
    "steering": {          // Current steering coefficients
        "excited": float,
        "frustrated": float,
        "calm": float,
        "depleted": float
    }
}
"""

# ----- Server Implementation Sketch -----

ELPIS_SERVER_SKETCH = '''
"""elpis-inference MCP server implementation."""

from mcp.server import Server
from mcp.types import Tool, TextContent

from elpis.emotion.state import EmotionalState
from elpis.emotion.regulator import HomeostasisRegulator
from elpis.llm.transformers_inference import TransformersInference


class ElpisInferenceServer:
    """
    MCP server providing emotional LLM inference.

    Stateful: maintains emotional state across requests.
    Any MCP client can use this without special integration.
    """

    def __init__(self, config: ElpisConfig):
        self.config = config

        # Emotional state (lives in server)
        self.emotional_state = EmotionalState(
            baseline_valence=config.baseline_valence,
            baseline_arousal=config.baseline_arousal,
        )
        self.regulator = HomeostasisRegulator(self.emotional_state)

        # Inference engine
        self.inference = TransformersInference(
            model_name_or_path=config.model_path,
            # ... other config
        )

        # MCP server
        self.server = Server("elpis-inference")
        self._register_tools()

    def _register_tools(self):
        """Register MCP tools."""

        @self.server.tool()
        async def generate(
            messages: list,
            max_tokens: int = None,
            temperature: float = None,
            top_p: float = None,
        ) -> dict:
            """Generate text with emotional modulation."""
            # Apply homeostatic decay
            self.regulator._apply_decay()

            # Get emotional parameters
            steering = self.emotional_state.get_steering_coefficients()
            params = self.emotional_state.get_modulated_params()

            # Generate (steering vectors applied internally)
            content = await self.inference.chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature or params["temperature"],
                top_p=top_p or params["top_p"],
                emotion_coefficients=steering,
            )

            # Optionally: analyze response for emotional feedback
            # self.regulator.process_response(content)

            return {
                "content": content,
                "emotion": self.emotional_state.to_dict(),
            }

        @self.server.tool()
        async def emotion_event(
            event_type: str,
            intensity: float = 1.0,
        ) -> dict:
            """Trigger an emotional event."""
            self.regulator.process_event(event_type, intensity)
            return {"emotion": self.emotional_state.to_dict()}

        @self.server.tool()
        async def emotion_get() -> dict:
            """Get current emotional state."""
            self.regulator._apply_decay()  # Apply time-based decay
            return {"emotion": self.emotional_state.to_dict()}

        @self.server.tool()
        async def emotion_reset() -> dict:
            """Reset to baseline."""
            self.emotional_state.reset()
            return {"emotion": self.emotional_state.to_dict()}


# Entry point
def main():
    import asyncio
    from mcp.server.stdio import stdio_server

    config = ElpisConfig.from_file("config.yaml")
    server = ElpisInferenceServer(config)

    asyncio.run(stdio_server(server.server))


if __name__ == "__main__":
    main()
'''


# =============================================================================
# 2. MNEMOSYNE MEMORY SERVER (mnemosyne)
# =============================================================================

"""
mnemosyne: Memory system as an independent MCP server

Named after the Greek goddess of memory (and mother of the Muses).

Key Design Decisions:
- Completely independent of inference backend
- Can be used with Elpis, Claude API, OpenAI, Ollama, anything
- Accepts emotional context as optional metadata
- Exposes consolidation as an explicit tool (client triggers "sleep")
- Can retrieve memories for context injection
"""

# ----- MCP Tool Definitions -----

MNEMOSYNE_TOOLS = """
Tools exposed by mnemosyne MCP server:

┌─────────────────────────────────────────────────────────────────┐
│ ENCODING TOOLS                                                  │
├─────────────────────────────────────────────────────────────────┤
│ memory_store                                                    │
│   Store a new memory                                           │
│   Params:                                                       │
│     - content: str (the memory content)                        │
│     - memory_type: str ("episodic", "semantic", "procedural")  │
│     - summary: str (optional, compressed version)              │
│     - emotional_context: {valence, arousal} (optional)         │
│     - tags: List[str] (optional)                               │
│     - metadata: dict (optional)                                │
│   Returns: {memory_id: str, status: str}                       │
│                                                                 │
│ memory_store_conversation                                       │
│   Store a conversation as episodic memory                      │
│   Params:                                                       │
│     - messages: List[{role, content}]                          │
│     - emotional_context: {valence, arousal} (optional)         │
│     - metadata: dict (optional)                                │
│   Returns: {memory_id: str, status: str}                       │
├─────────────────────────────────────────────────────────────────┤
│ RETRIEVAL TOOLS                                                 │
├─────────────────────────────────────────────────────────────────┤
│ memory_query                                                    │
│   Query memories by semantic similarity                        │
│   Params:                                                       │
│     - query: str (text to match against)                       │
│     - n_results: int (default 5)                               │
│     - memory_types: List[str] (optional filter)                │
│     - min_importance: float (optional threshold)               │
│   Returns: {memories: List[Memory]}                            │
│                                                                 │
│ memory_recent                                                   │
│   Get recent memories                                          │
│   Params:                                                       │
│     - n_results: int (default 10)                              │
│   Returns: {memories: List[Memory]}                            │
│                                                                 │
│ memory_get_context                                              │
│   Get formatted memories for context injection                 │
│   Params:                                                       │
│     - query: str (current context/input)                       │
│     - emotional_state: {valence, arousal} (optional)           │
│     - max_tokens: int (approximate budget)                     │
│   Returns: {context: str, memory_ids: List[str]}               │
├─────────────────────────────────────────────────────────────────┤
│ CONSOLIDATION TOOLS                                             │
├─────────────────────────────────────────────────────────────────┤
│ memory_consolidate                                              │
│   Run sleep consolidation cycle                                │
│   Params:                                                       │
│     - emotional_state: {valence, arousal} (optional)           │
│   Returns: {report: ConsolidationReport}                       │
│                                                                 │
│ memory_should_consolidate                                       │
│   Check if consolidation is recommended                        │
│   Returns: {should_consolidate: bool, reason: str}             │
├─────────────────────────────────────────────────────────────────┤
│ MANAGEMENT TOOLS                                                │
├─────────────────────────────────────────────────────────────────┤
│ memory_delete                                                   │
│   Delete a memory by ID                                        │
│   Params:                                                       │
│     - memory_id: str                                           │
│   Returns: {deleted: bool}                                     │
│                                                                 │
│ memory_stats                                                    │
│   Get memory system statistics                                 │
│   Returns: {total, by_type, by_status, buffer_size, etc}       │
├─────────────────────────────────────────────────────────────────┤
│ RESOURCES                                                       │
├─────────────────────────────────────────────────────────────────┤
│ memory://stats       - Current statistics                      │
│ memory://recent      - Recent memories                         │
│ memory://config      - Current configuration                   │
└─────────────────────────────────────────────────────────────────┘
"""


# =============================================================================
# 3. USAGE PATTERNS
# =============================================================================

USAGE_PATTERNS = """
Usage Patterns for Modular Elpis
================================

Pattern 1: Full Stack (Psyche-style)
------------------------------------
Use both elpis-inference and mnemosyne together for a complete
emotionally-aware agent with persistent memory.

    MCP Config:
    ```json
    {
        "mcpServers": {
            "elpis": {
                "command": "elpis-server",
                "args": ["--config", "elpis.yaml"]
            },
            "memory": {
                "command": "mnemosyne-server",
                "args": ["--config", "mnemosyne.yaml"]
            }
        }
    }
    ```

    Client pseudocode:
    ```python
    # Before generating
    context = await mcp.call("memory", "memory_get_context", {
        "query": user_input,
        "emotional_state": await mcp.call("elpis", "emotion_get")
    })

    # Generate with emotion
    response = await mcp.call("elpis", "generate", {
        "messages": [
            {"role": "system", "content": f"You are Psyche.\\n{context}"},
            {"role": "user", "content": user_input},
        ]
    })

    # Store conversation
    await mcp.call("memory", "memory_store_conversation", {
        "messages": messages,
        "emotional_context": response["emotion"],
    })

    # Update emotion based on outcome
    if task_succeeded:
        await mcp.call("elpis", "emotion_event", {"event_type": "success"})
    ```


Pattern 2: Emotional Inference Only
-----------------------------------
Use elpis-inference with any harness for emotional LLM responses,
no memory system.

    MCP Config:
    ```json
    {
        "mcpServers": {
            "elpis": {
                "command": "elpis-server",
                "args": ["--config", "elpis.yaml"]
            }
        }
    }
    ```

    Works with: opencode, aider, custom harnesses

    The harness just calls generate/emotion_event as needed.
    Elpis handles all emotional state internally.


Pattern 3: Memory Only (with external LLM)
------------------------------------------
Use mnemosyne with Claude API, OpenAI, or any other backend.

    MCP Config:
    ```json
    {
        "mcpServers": {
            "memory": {
                "command": "mnemosyne-server",
                "args": ["--config", "mnemosyne.yaml"]
            }
        }
    }
    ```

    Client pseudocode:
    ```python
    # Get relevant memories
    context = await mcp.call("memory", "memory_get_context", {
        "query": user_input,
    })

    # Use any LLM backend
    response = await claude_client.messages.create(
        messages=[
            {"role": "system", "content": f"Context:\\n{context}"},
            {"role": "user", "content": user_input},
        ]
    )

    # Store the conversation
    await mcp.call("memory", "memory_store_conversation", {
        "messages": [...],
    })
    ```


Pattern 4: Hybrid (External LLM + Elpis Emotion)
------------------------------------------------
Use Claude API for inference but Elpis for emotional tracking.
Emotion affects system prompts rather than steering vectors.

    Client pseudocode:
    ```python
    # Get emotional state
    emotion = await mcp.call("elpis", "emotion_get")

    # Build emotionally-informed system prompt
    system = build_system_prompt(emotion["emotion"])

    # Use external LLM
    response = await claude_client.messages.create(...)

    # Update emotion based on response
    await mcp.call("elpis", "emotion_event", {"event_type": "success"})
    ```

    Note: This doesn't get steering vector benefits, but still
    provides emotional continuity and can modulate prompts.
"""


# =============================================================================
# 4. PACKAGE STRUCTURE
# =============================================================================

PACKAGE_STRUCTURE = """
Proposed Package Structure
==========================

Two separate PyPI packages that can be installed independently:

elpis-inference/
├── pyproject.toml
├── README.md
├── src/
│   └── elpis/
│       ├── __init__.py
│       ├── server.py           # MCP server entry point
│       ├── emotion/
│       │   ├── __init__.py
│       │   ├── state.py        # EmotionalState
│       │   └── regulator.py    # HomeostasisRegulator
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── transformers_inference.py
│       │   └── steering.py     # Steering vector management
│       └── config/
│           ├── __init__.py
│           └── settings.py
└── tests/

mnemosyne/
├── pyproject.toml
├── README.md
├── src/
│   └── mnemosyne/
│       ├── __init__.py
│       ├── server.py           # MCP server entry point
│       ├── types.py            # Memory, EmotionalContext, etc.
│       ├── store.py            # ChromaDB storage
│       ├── encoder.py          # Memory encoding
│       ├── retriever.py        # Memory retrieval
│       ├── consolidator.py     # Sleep consolidation
│       └── config/
│           ├── __init__.py
│           └── settings.py
└── tests/

psyche/  (optional reference harness)
├── pyproject.toml
├── README.md
├── src/
│   └── psyche/
│       ├── __init__.py
│       ├── harness.py          # Main loop
│       ├── repl.py             # User interface
│       └── mcp_client.py       # Connects to elpis + mnemosyne
└── tests/


Installation:
    # Just emotional inference
    pip install elpis-inference

    # Just memory
    pip install mnemosyne-memory

    # Full stack
    pip install elpis-inference mnemosyne-memory

    # With reference harness
    pip install elpis-inference mnemosyne-memory psyche-harness


CLI Entry Points:
    # Start inference server
    elpis-server --config config.yaml

    # Start memory server
    mnemosyne-server --config config.yaml

    # Start reference harness (auto-spawns servers via MCP)
    psyche
"""


# =============================================================================
# 5. SUMMARY
# =============================================================================

SUMMARY = """
Summary: Modular Elpis Architecture
===================================

BEFORE (Current):
    ┌─────────────────────────────────────┐
    │              Elpis                   │
    │  ┌─────────────────────────────────┐│
    │  │ Psyche (harness + memory + UI)  ││
    │  │  ┌───────────────────────────┐  ││
    │  │  │ Elpis Server              │  ││
    │  │  │  - Inference              │  ││
    │  │  │  - Emotion                │  ││
    │  │  └───────────────────────────┘  ││
    │  └─────────────────────────────────┘│
    └─────────────────────────────────────┘

    Tightly coupled, only works with Psyche


AFTER (Proposed):
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │   elpis     │  │  mnemosyne  │  │   other     │
    │ (emotion +  │  │  (memory)   │  │  MCP tools  │
    │  inference) │  │             │  │             │
    └─────────────┘  └─────────────┘  └─────────────┘
          ▲                ▲                ▲
          │                │                │
          └────────────────┼────────────────┘
                           │
                    ┌──────┴──────┐
                    │ Any MCP     │
                    │ Client      │
                    │             │
                    │ - Psyche    │
                    │ - opencode  │
                    │ - aider     │
                    │ - custom    │
                    └─────────────┘

    Composable, works with any harness


Key Benefits:
1. Use emotional inference with any harness
2. Use memory system with any LLM backend
3. Mix and match components
4. Clear API boundaries
5. Independent versioning/releases
6. Easier testing and maintenance
7. Community can build alternative components

Names:
- elpis-inference: Emotional LLM inference server
- mnemosyne: Memory system server (goddess of memory)
- psyche: Reference harness (soul/mind, also the agent's name)
"""

if __name__ == "__main__":
    print(SUMMARY)
