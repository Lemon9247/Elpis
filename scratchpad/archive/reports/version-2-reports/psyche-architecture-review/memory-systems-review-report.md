# Memory Systems Review Report

**Agent**: Memory Systems Review Agent
**Date**: 2026-01-16
**Task**: Research memory systems and approaches for LLM applications

---

## Executive Summary

This report reviews memory systems for LLM applications, with a focus on the FocusLLM paper and best practices for context compaction. The current Psyche memory system has fundamental workflow issues that cause memories not to be stored on shutdown. This report provides recommendations for a robust memory workflow that addresses these issues while leveraging ChromaDB as the underlying storage layer.

**Key Findings:**

1. **FocusLLM** is designed for extending context length at training time, not for runtime memory management - it is **not directly applicable** to Psyche's use case
2. The current Psyche memory system has a **sound architectural foundation** but flawed execution workflow
3. The main issues are: lack of explicit save triggers, shutdown storage failures, and missing summarization
4. The recommended approach is **agent-driven memory management** (inspired by Letta/MemGPT) with explicit storage decisions

---

## 1. FocusLLM Paper Analysis

### 1.1 Paper Overview

**Title**: FocusLLM: Scaling LLM's Context by Parallel Decoding
**ArXiv**: https://arxiv.org/abs/2408.11745

FocusLLM is a training-time method to extend the effective context length of LLMs. It works by:

1. **Parallel Context Encoding**: Divides long context into chunks and encodes them in parallel
2. **Focus Tokens**: Extracts "focus tokens" from each chunk that capture salient information
3. **Decoder Integration**: Integrates focus tokens into the decoder for generation

### 1.2 Technical Approach

The core mechanism involves:

- Splitting input context into fixed-size chunks (e.g., 4K tokens each)
- Using a local encoder to process each chunk in parallel
- Extracting a small set of "focus tokens" from each chunk via attention-based selection
- Concatenating focus tokens with the query for final generation

This is fundamentally a **training-time architecture modification** that requires:
- Retraining or fine-tuning the model
- Modified inference architecture with parallel chunk processing
- Custom focus token extraction layers

### 1.3 Applicability to Psyche

**Verdict: NOT DIRECTLY APPLICABLE**

FocusLLM is designed for a different problem:

| FocusLLM | Psyche Needs |
|----------|--------------|
| Training-time context extension | Runtime memory management |
| Parallel encoder architecture | Sequential conversation flow |
| Fixed focus token extraction | Dynamic relevance-based retrieval |
| Model architecture changes | Works with any LLM backend |

Psyche uses external models (llama-cpp, transformers) that cannot be modified at the architecture level. FocusLLM would require retraining these models with the focus token mechanism.

**However**, the conceptual insight is useful:

> **Key Insight**: "Focus tokens" - the idea of extracting salient information from context chunks - maps conceptually to what a good summarization system should do.

This reinforces the importance of intelligent summarization when compacting context.

---

## 2. Alternative Memory Approaches Reviewed

### 2.1 Letta/MemGPT Architecture (Most Relevant)

The Letta (formerly MemGPT) architecture is highly relevant to Psyche. Key concepts:

#### Hierarchical Memory System
```
+------------------------+
|   In-Context Memory    |  <- Editable system prompt section
|   (Core Memory)        |     Character-limited blocks
+------------------------+
           |
+------------------------+
|    Message Buffer      |  <- Recent conversation
|   (Working Memory)     |     Sliding window
+------------------------+
           |
+------------------------+
|   External Storage     |  <- ChromaDB/Vector DB
|   (Archival + Recall)  |     Semantic retrieval
+------------------------+
```

#### Agent-Driven Memory Management
The LLM itself decides what to store using explicit tools:
- `memory_replace`: Update existing memory blocks
- `memory_insert`: Add new information
- `memory_rethink`: Consolidate and reorganize
- `archival_memory_insert`: Store to long-term
- `archival_memory_search`: Retrieve from long-term

This is the **opposite** of passive compaction - the agent actively manages its own memory.

#### Sleep-Time Compute (Consolidation)
Background processing during idle periods:
- Identify contradictions in stored memories
- Abstract patterns from specific experiences
- Pre-compute associations
- Recursive summarization

**Relevance to Psyche**: This maps directly to Psyche's idle reflection and consolidation system.

### 2.2 RAG (Retrieval-Augmented Generation)

Standard RAG pattern for memory:

1. **Store**: Embed messages/facts and store in vector database
2. **Retrieve**: On new query, search for semantically relevant memories
3. **Augment**: Include retrieved context in prompt
4. **Generate**: LLM responds with augmented context

**Current Psyche Implementation**: Already uses ChromaDB with this pattern via `recall_memory` tool.

### 2.3 Conversation Summarization

Multiple approaches exist:

1. **Naive Truncation**: Just keep last N messages (current fallback)
2. **LLM Summarization**: Use LLM to generate summary (partially implemented)
3. **Structured Summarization**: Force specific sections (files modified, decisions, etc.)
4. **Hierarchical Summarization**: Progressive compression over time

### 2.4 Token-Budget-Aware Reasoning

Research from ACL 2025 shows:
- Dynamic token budget allocation based on task complexity
- Can reduce costs 68% while maintaining performance
- Relevant for Psyche's context management

---

## 3. Current Psyche Memory System Analysis

### 3.1 Architecture Overview

```
Psyche Memory Flow:

User Input -> ContextCompactor -> [Sliding Window]
                   |
                   v (on compaction)
            Staged Messages -> (delayed store) -> Mnemosyne Short-Term
                                                         |
                                                         v (consolidation)
                                                  Mnemosyne Long-Term
```

**Components**:
- `ContextCompactor` (`src/psyche/memory/compaction.py`): Token-based sliding window
- `MemoryServer` (`src/psyche/memory/server.py`): Manages the inference loop and memory
- `ChromaMemoryStore` (`src/mnemosyne/storage/chroma_store.py`): Persistent vector storage
- `MemoryConsolidator` (`src/mnemosyne/core/consolidator.py`): Promotes important memories

### 3.2 Identified Issues

#### Issue 1: Shutdown Memory Loss
**Location**: `server.py` lines 1272-1330 (`shutdown_with_consolidation`)

The shutdown flow has race conditions:
```python
# Problem: all_messages includes already-cleared staged_messages
all_messages = self._staged_messages + remaining  # _staged_messages may be empty
if all_messages:
    await self._store_conversation_summary(all_messages)
```

Also, if Mnemosyne is not connected at shutdown, all memories are lost silently.

#### Issue 2: Passive Compaction Only
The system only stores memories when context overflows:
- If conversation ends before overflow, nothing is stored
- No explicit "save this" triggers
- Agent cannot decide what's worth remembering

#### Issue 3: Naive Summarization
Currently uses simple truncation:
```python
summary=msg.content[:100]  # Just first 100 chars
```

While `_summarize_conversation` exists, it's only called on shutdown.

#### Issue 4: No Semantic Deduplication
Consolidation clusters by embedding similarity but doesn't merge semantically identical information effectively.

#### Issue 5: Timing Dependencies
Consolidation requires `min_age_hours=1` before memories are eligible - short sessions won't consolidate.

### 3.3 What Works Well

1. **ChromaDB Integration**: Solid vector storage with semantic search
2. **Emotional Context**: Memories tagged with emotional state (valence/arousal)
3. **Importance Scoring**: Multi-factor importance calculation
4. **Memory Types**: Episodic, semantic, procedural, emotional classification
5. **Tool-Based Access**: `recall_memory` and `store_memory` tools

---

## 4. Recommended Memory Workflow

### 4.1 Core Principles

1. **Agent-Driven Storage**: The agent decides what to remember, not passive overflow
2. **Explicit Save Points**: Regular checkpoints, not just on shutdown
3. **Structured Summaries**: Use LLM to generate meaningful summaries
4. **Graceful Degradation**: Handle disconnections without data loss
5. **Immediate Retrieval**: Retrieved memories go into context for the current turn

### 4.2 Proposed Memory Workflow

```
+------------------------------------------------------------------+
|                        MEMORY WORKFLOW                            |
+------------------------------------------------------------------+
                              |
                    [User Message Arrives]
                              |
                              v
                    +-------------------+
                    | Retrieve Relevant |  <- Automatic RAG
                    | Long-Term Memory  |
                    +-------------------+
                              |
                              v
                    +-------------------+
                    | Include in Prompt |
                    +-------------------+
                              |
                              v
                    +-------------------+
                    | Generate Response |
                    +-------------------+
                              |
                              v
           +------------------+------------------+
           |                                     |
           v                                     v
   [Agent uses store_memory]            [No explicit store]
           |                                     |
           v                                     v
   +---------------+                    +------------------+
   | Store to      |                    | Message stays in |
   | Short-Term    |                    | Working Memory   |
   +---------------+                    +------------------+
           |
           |
           +---> [Periodic Checkpoint] ---> Store Conversation Checkpoint
           |
           +---> [Idle Consolidation] ---> Promote Important Memories
           |
           +---> [Shutdown] ---> Store Summary + Remaining Context

```

### 4.3 Specific Recommendations

#### Recommendation 1: Automatic Context Retrieval
On each user message, automatically query Mnemosyne for relevant memories and inject them into context:

```
When user sends: "Continue working on the auth system"
                              |
                              v
Query: "auth system" --> ChromaDB --> [Relevant memories]
                              |
                              v
Inject as: "[Memory] Previous work on auth: implemented JWT..."
```

This should be automatic, not requiring explicit `recall_memory` calls.

#### Recommendation 2: Periodic Memory Checkpoints
Don't wait for shutdown. Save conversation checkpoints:
- Every N messages (e.g., 20)
- After significant tool use
- On explicit user trigger (`/save`)
- On connection issues

#### Recommendation 3: Structured Conversation Summaries
Replace truncation with structured summarization:

```markdown
## Conversation Summary - [Timestamp]

### Topics Discussed
- Topic 1: Brief description
- Topic 2: Brief description

### Key Facts Learned
- User preference: X
- Project context: Y

### Decisions Made
- Decided to use approach X because...

### Unresolved Items
- Still need to figure out...
```

#### Recommendation 4: Local Checkpoint Fallback
When Mnemosyne is disconnected, write checkpoints to local JSON:

```python
# On compaction when Mnemosyne unavailable:
checkpoint = {
    "timestamp": datetime.now().isoformat(),
    "messages": [m.to_dict() for m in dropped_messages],
    "pending_storage": True
}
Path("~/.psyche/pending_memories.json").write_text(json.dumps(checkpoint))
```

On reconnection, process pending memories.

#### Recommendation 5: Reduce Consolidation Delay
Change `min_age_hours` from 1 hour to 0:
- Allow immediate consolidation for short sessions
- Importance threshold provides filtering
- Better than losing memories entirely

#### Recommendation 6: Enhanced Store Memory Tool
Make `store_memory` smarter:

```python
store_memory(
    content="...",
    summary="...",  # Auto-generate if not provided
    importance=0.8,  # Explicit importance hint
    merge_similar=True,  # Merge with similar existing memory
)
```

### 4.4 When to Compact

Current trigger: When total tokens exceed `available_tokens` (max - reserve)

**Recommended Enhancement**:

1. **Proactive Compaction** at 70% capacity (not 100%)
2. **Time-Based Compaction**: After 5+ minutes of inactivity
3. **Semantic Compaction**: When many messages are semantically similar
4. **Explicit Compaction**: `/compact` command

### 4.5 What to Store in Long-Term Memory

**Definitely Store**:
- User preferences and patterns
- Key facts learned about user/project
- Decisions and their rationale
- Errors encountered and solutions
- Successful interaction patterns

**Maybe Store** (based on importance):
- Technical discussions
- Problem-solving steps
- Questions asked and answered

**Don't Store**:
- Greetings and small talk
- Transient tool outputs
- Error messages without resolution
- Duplicate information

### 4.6 How to Retrieve Effectively

**Current**: Explicit `recall_memory` tool calls

**Recommended Enhancement**:

1. **Automatic Pre-Retrieval**:
   - Before processing user input, search for relevant memories
   - Inject top-K (e.g., 3) as system context

2. **Query Expansion**:
   - Don't just search for user's words
   - Extract entities and topics, search for those too

3. **Recency Weighting**:
   - Boost recent memories in search results
   - Decay older memories unless explicitly important

4. **Type-Based Filtering**:
   - For code tasks: prioritize procedural memories
   - For discussion: prioritize semantic memories
   - For emotional support: prioritize emotional memories

---

## 5. Implementation Considerations with ChromaDB

### 5.1 Current ChromaDB Usage (Good)

The current implementation has solid ChromaDB integration:
- Separate collections for short-term and long-term
- Embedding with SentenceTransformer (all-MiniLM-L6-v2)
- Metadata storage for memory attributes
- Semantic search with cosine similarity

### 5.2 Recommended ChromaDB Enhancements

#### Enhanced Metadata Schema
```python
metadata = {
    # Current
    "summary": str,
    "memory_type": str,  # episodic, semantic, procedural, emotional
    "status": str,
    "importance_score": float,
    "created_at": str,
    "tags": json_list,
    "emotional_context": json_dict,

    # Recommended additions
    "session_id": str,      # Track which session created this
    "access_count": int,    # Track retrieval frequency
    "last_accessed": str,   # For recency weighting
    "source_type": str,     # "user", "agent", "tool", "summary"
    "topic_cluster": str,   # For grouping related memories
}
```

#### Collection Restructuring
Consider adding specialized collections:
```python
collections = {
    "short_term": "recent, unprocessed memories",
    "long_term": "consolidated important memories",
    "facts": "distilled factual knowledge",        # NEW
    "preferences": "user preferences & patterns",  # NEW
    "session_summaries": "per-session summaries",  # NEW
}
```

#### Optimized Retrieval Query
```python
def retrieve_relevant_context(query: str, n_results: int = 5):
    # Multi-collection search with weighting
    results = []

    # Search facts first (highest priority)
    results += search_collection("facts", query, n=2)

    # Then preferences
    results += search_collection("preferences", query, n=1)

    # Then long-term memories
    results += search_collection("long_term", query, n=3)

    # Rank by combined score: similarity * recency * importance
    return rank_results(results, n_results)
```

---

## 6. Comparison with Other Approaches

| Aspect | Current Psyche | Letta/MemGPT | Standard RAG | FocusLLM |
|--------|---------------|--------------|--------------|----------|
| Storage | ChromaDB | Database + Vector | Vector DB | N/A (training) |
| Compaction | Passive overflow | Active agent-driven | None | Architecture |
| Summarization | Truncation | LLM-based | None | Focus tokens |
| Consolidation | Time + importance | Sleep-time compute | None | N/A |
| Retrieval | Explicit tool | Agent-controlled | Query-based | Built-in |

**Recommendation**: Adopt Letta's agent-driven philosophy while keeping ChromaDB infrastructure.

---

## 7. Priority Implementation Order

### Phase 1: Fix Critical Issues (Immediate)
1. Fix shutdown memory loss - ensure all messages are stored
2. Add connection check before all storage operations
3. Implement local checkpoint fallback

### Phase 2: Improve Workflow (Short-term)
4. Implement automatic context retrieval on user messages
5. Add structured conversation summarization
6. Create periodic checkpoint system (every N messages)

### Phase 3: Enhance Intelligence (Medium-term)
7. Add semantic deduplication in consolidation
8. Implement query expansion for retrieval
9. Add importance hints to store_memory tool

### Phase 4: Advanced Features (Long-term)
10. Implement topic clustering for memories
11. Add specialized collections (facts, preferences)
12. Implement memory merging and refinement

---

## 8. Conclusion

The current Psyche memory system has solid foundations but flawed execution. The FocusLLM paper, while interesting for context extension research, is not applicable to Psyche's runtime memory management needs.

**Key Recommendations Summary:**

1. **Fix the shutdown storage bug** - critical data loss issue
2. **Add automatic context retrieval** - don't require explicit recall
3. **Implement periodic checkpoints** - don't wait for shutdown
4. **Use LLM for structured summarization** - not just truncation
5. **Adopt agent-driven memory philosophy** - let the LLM decide importance
6. **Add local fallback storage** - handle disconnections gracefully

The ChromaDB backend is well-suited for this task. The focus should be on the workflow layer, not the storage layer.

---

## References

### Papers and Documentation
- FocusLLM: https://arxiv.org/abs/2408.11745
- Letta/MemGPT Documentation: https://docs.letta.com/
- MemGPT Paper: "MemGPT: Towards LLMs as Operating Systems"
- Token-Budget-Aware LLM Reasoning (ACL 2025): https://arxiv.org/abs/2412.18547

### Existing Project Research
- `/home/lemoneater/Projects/Personal/Elpis/scratchpad/archive/phase-2/context-compaction-research.md`
- `/home/lemoneater/Projects/Personal/Elpis/scratchpad/archive/phase-2/letta-architecture-research.md`
- `/home/lemoneater/Projects/Personal/Elpis/scratchpad/plans/20260114-memory-summarization-plan.md`

### Key Source Files
- `/home/lemoneater/Projects/Personal/Elpis/src/psyche/memory/compaction.py`
- `/home/lemoneater/Projects/Personal/Elpis/src/psyche/memory/server.py`
- `/home/lemoneater/Projects/Personal/Elpis/src/mnemosyne/core/consolidator.py`
- `/home/lemoneater/Projects/Personal/Elpis/src/mnemosyne/storage/chroma_store.py`

---

**Report Status**: Complete
**Next Steps**: Integrate findings into architecture review synthesis
