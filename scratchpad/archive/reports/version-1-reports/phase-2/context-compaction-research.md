# Context Window Management and Compaction Strategies Research

**Research Date**: 2026-01-12
**Researcher**: Context Compaction Researcher
**Project**: Elpis Phase 2 Architecture

## Executive Summary

This report provides comprehensive research on context window management and compaction strategies for long-running LLM conversations. Key findings include:

1. **Anthropic's Approach**: Claude uses a multi-layered strategy including context editing, automatic compaction, prompt caching, and file-based memory tools to manage context windows up to 1M tokens.

2. **Primary Strategies**: The three main approaches are:
   - **Reversible Compaction**: Strip redundant information that exists in the environment (files, databases) - zero data loss, can be retrieved on demand
   - **Lossy Summarization**: LLM-generated summaries of conversation history - some information loss but reduces tokens
   - **Hybrid Approaches**: Combine multiple techniques with preference hierarchy (raw > compaction > summarization)

3. **Performance Impact**: Token-budget-aware reasoning can reduce costs by 68.64% while maintaining competitive performance. Smart caching can reduce API calls by 70%+ and improve response times from 2-3s to <100ms.

4. **Key Trade-off**: Balance between context preservation and token efficiency. The "sweet spot" is providing just enough relevant context without overwhelming the model or inflating costs.

5. **Recommendations for Elpis**:
   - Implement tiered context management (core/working/transient buckets)
   - Use reversible compaction as primary strategy with file-based memory
   - Apply lossy summarization only when compaction insufficient
   - Integrate ChromaDB for semantic memory retrieval
   - Monitor context usage and implement proactive compaction at 70% threshold

---

## 1. Context Window Management Strategies

### 1.1 Sliding Window (Drop Oldest Messages)

**Overview**: A FIFO (first in, first out) queue that discards oldest interactions as the context window approaches capacity.

**How It Works**:
- Process text in overlapping segments
- Example: With 1000-token capacity, first segment covers tokens 1-1000, second segment 501-1500, etc.
- Overlap ensures key information from one segment is available at the beginning of the next
- Loop through message history from newest to oldest to preserve recent messages

**Advantages**:
- Simple to implement
- Low computational overhead
- Predictable behavior
- No additional API calls required

**Disadvantages**:
- Permanent information loss of older context
- No consideration of message importance
- Can lose critical information from early conversation
- Limited applicability for tasks requiring long-term memory

**Best Use Cases**:
- Short-lived conversations
- Tasks where only recent context matters
- Prototyping and development
- Cost-sensitive applications with simple requirements

**Implementation Notes**:
- LangChain provides `ConversationBufferMemory` that keeps only the last N messages
- Maintains recency bias (LLMs weigh recent content more heavily)
- Can be enhanced with overlap to preserve continuity

---

### 1.2 Summarization (LLM Summarizes Old Context)

**Overview**: Use an LLM to generate compressed summaries of conversation history, replacing verbose transcripts with condensed versions.

**How It Works**:
- Triggered at context threshold (commonly 128k tokens)
- LLM analyzes conversation history and generates summary
- Most recent tool calls kept in raw, full-detail format
- Older segments progressively compressed into summaries

**Advantages**:
- Preserves semantic meaning better than simple truncation
- Can identify and retain important information
- Enables very long conversations beyond raw context limits
- Maintains conversation coherence

**Disadvantages**:
- **Lossy**: Information cannot be fully recovered once summarized
- **Costly**: Every summarization requires expensive API call (can be 7%+ of total cost)
- **Quality variance**: Generic summarization can lose critical details
- **Latency**: Adds pause during summary generation (unless pre-computed)

**Types of Summarization**:

1. **Freeform Summarization**: LLM generates open-ended summary
   - Flexible but risks silently dropping important details
   - Prone to "gradual information loss"

2. **Structured Summarization**: Dedicated sections for specific information types
   - Forces preservation of critical elements (file paths, decisions, etc.)
   - Better consistency and completeness
   - Example structure:
     ```
     ## Key Decisions Made
     ## Files Created/Modified
     ## Unresolved Issues
     ## Action Items
     ## Technical Details
     ```

3. **Hierarchical Summarization**: Progressive compression as information ages
   - Recent exchanges remain verbatim
   - Older content gets increasingly compressed
   - Multi-tier approach balances detail and efficiency

**Best Use Cases**:
- Long-running conversations requiring historical context
- Complex multi-step tasks spanning multiple sessions
- Scenarios where semantic understanding of past context is critical
- Applications where cost of summarization < cost of raw token usage

**Performance Data**:
- Hybrid approach (observation masking + LLM summarization) reduced costs by 11% vs pure summarization
- Improved answer accuracy by ~2.6%

**Implementation Considerations**:
- Consider background processes that pre-compute summaries proactively
- Keep recent tool calls in raw format (don't summarize immediately)
- Use structured templates to prevent information loss
- Monitor summary quality and adjust prompts accordingly

---

### 1.3 Importance-Based (Keep Important Messages, Drop Others)

**Overview**: Intelligently select which context to retain based on semantic importance, relevance, and functional role rather than just chronological order.

**How It Works**:

**Token-Level Importance**:
- Aggregate multi-head attention weights to identify critical tokens
- Calculate semantic alignment with task/query
- Apply facility-location diversity objectives
- Assign importance scores to each token

**Function-Aware Pruning**:
- Group neurons by functional roles
- Prune groups independently
- Weight tokens most semantically aligned with group's function
- Preserve structural coherence

**Semantic Similarity**:
- Embed messages/chunks using vector embeddings
- Calculate similarity to current query/task
- Retain semantically relevant context
- Use vector databases (ChromaDB) for efficient retrieval

**Advantages**:
- Preserves most relevant information regardless of age
- Can dramatically reduce context size (up to 80%) with minimal quality loss
- Adapts to task requirements dynamically
- Prevents "lost-in-the-middle" effect

**Disadvantages**:
- Complex implementation requiring additional infrastructure
- Computational overhead for importance calculation
- Risk of incorrectly assessing importance
- May discard context that becomes relevant later

**Performance Data**:
- Dynamic context pruning enables up to 80% context reduction
- 2x inference speedup
- Negligible perplexity/performance drop on benchmarks (GLUE, WinoGrande, HellaSwag)

**Best Use Cases**:
- RAG (Retrieval-Augmented Generation) systems
- Multi-turn conversations with diverse topics
- Scenarios with large knowledge bases
- Applications requiring selective context from extensive history

**Implementation Patterns**:
- Combine with vector databases (ChromaDB) for semantic search
- Use embedding models to convert text to semantic vectors
- Implement similarity thresholds for retrieval
- Cache common queries/responses to avoid redundant computation

---

### 1.4 Hybrid Approaches

**Overview**: Combine multiple strategies to leverage strengths and mitigate weaknesses of individual approaches.

**Recommended Hierarchy**:
```
Preference: Raw > Compaction > Summarization
```

**Strategy**: Use most information-preserving approach that fits within limits.

**Multi-Technique Integration**:

1. **Sliding Window + Vector Database** (MemoryLLM pattern):
   - Keep last N turns in raw sliding window (e.g., 10 recent turns)
   - Store older conversations in vector database
   - Retrieve semantically relevant past context on demand
   - Best of both worlds: recent context + selective long-term memory

2. **Observation Masking + LLM Summarization**:
   - Mask/strip redundant observations from tool outputs
   - Summarize remaining context with LLM
   - 7% cost reduction vs masking alone
   - 11% cost reduction vs summarization alone
   - ~2.6% accuracy improvement

3. **Context Bucketing** (Core/Working/Transient):
   - **Core**: Must persist (system prompts, key decisions, file structure)
   - **Working**: Needed now, summarize later (active task context)
   - **Transient**: Use once, discard (tool outputs, temporary data)
   - Explicit management of what goes where

4. **Tiered Compression**:
   - Recent (0-50 messages): Full verbatim text
   - Mid-term (50-200 messages): Structured summaries
   - Long-term (200+ messages): Semantic vectors in database
   - Retrieve on-demand based on relevance

**Advantages**:
- Flexibility to adapt to different scenarios
- Optimizes for both quality and cost
- Reduces risk of critical information loss
- Better overall performance than single-strategy approaches

**Best Practices**:
- Start with reversible approaches (compaction)
- Progress to lossy approaches (summarization) only when necessary
- Use semantic search for long-term memory retrieval
- Maintain explicit context budgets and monitor usage

---

## 2. How Anthropic Handles Context

### 2.1 Context Window Sizes

**Current Capacity** (2026):
- **Claude Sonnet 4/4.5**: Up to 1 million tokens
- **Standard users**: 200,000 tokens per session
- **Claude.ai Enterprise**: 500,000 tokens per session
- **Beta (eligible orgs)**: 1 million tokens per session

**Context Awareness**:
- Claude Sonnet 4.5 and Haiku 4.5 feature built-in context awareness
- Models track their remaining context window ("token budget") throughout conversation
- Intelligent handling when approaching limits instead of hard errors

### 2.2 Context Editing

**Automatic Stale Content Removal**:
- Automatically clears stale tool calls and results when approaching token limits
- Preserves conversation flow while removing redundant content
- Agents executing tasks accumulate tool results; context editing removes stale content
- No manual intervention required

**Thinking Block Clearing** (Beta):
- `context-management-2025-06-27` beta feature
- Claude automatically clears older thinking blocks from previous turns
- Previous thinking blocks automatically ignored in context usage calculations
- No need to manually remove old thinking blocks
- Cache invalidates when non-tool-result user content added

### 2.3 Context Compaction

**Anthropic's Definition**:
> "Compaction is the practice of taking a conversation nearing the context window limit, summarizing its contents, and reinitiating a new context window with the summary."

**Implementation in Claude Code**:
- Message history passed to model to summarize and compress
- Preserves critical details:
  - Architectural decisions
  - Unresolved bugs
  - Implementation details
  - File paths and structure
- Discards redundant elements:
  - Verbose tool outputs
  - Repetitive messages
  - Fully-resolved temporary context

**Reversible vs. Lossy**:
- **Reversible Compaction**: Strip information that exists in environment (files)
  - Example: 500-line code file → store only file path in history
  - Can retrieve full content later using file read tools

- **Lossy Summarization**: Use when compaction insufficient
  - LLM generates summary of conversation
  - Some information permanently compressed
  - Last resort when reversible compaction doesn't free enough space

### 2.4 Memory Tool

**File-Based Persistent Memory**:
- Store information outside context window
- Dedicated memory directory in user's infrastructure
- Persists across conversations
- CRUD operations: Create, Read, Update, Delete memory files

**Benefits**:
- Infinite long-term memory potential
- Survives session restarts
- Can be version-controlled
- Shared across multiple agents/sessions

**Use Cases**:
- User preferences and settings
- Project structure and architecture notes
- Long-term goals and constraints
- Domain knowledge and patterns

### 2.5 Prompt Caching

**Cache Breakpoints**:
- Developer-controlled caching with explicit breakpoints
- Specify which portions of prompt should be cached
- Configurable time-to-live (TTL) options
- Reduces redundant processing of static context

**Cost Savings**:
- 90% cost reduction for cached content reads
- Cached tokens count as input tokens in usage metrics
- 5-minute default TTL, customizable

**Extended Thinking Integration**:
- Thinking blocks cached as part of request content
- Subsequent API calls with tool results use cached thinking
- Cache invalidates when:
  - Thinking parameters change (enable/disable, budget allocation)
  - Non-tool-result user content added
  - Previous thinking blocks stripped
- System prompts and tools remain cached despite thinking block changes

**Best Practices**:
- Place stable context (system prompts, tools, knowledge base) before cache breakpoints
- Don't break cache with frequently-changing content
- Monitor cache hit rates
- Be aware that cache invalidation can increase costs unexpectedly

### 2.6 Claude Code Best Practices

**CLAUDE.md File**:
- Repository root file for persistent project context
- Define goals, allowed tools, style guides, escalation rules
- Read automatically every session
- Acts as "always-on" memory for project fundamentals

**Context Management Commands**:
- `/compact`: Manually trigger compaction at ~70% capacity
- `/clear`: Clear context between tasks to prevent drift
- Monitor context meter proactively

**Subagent Pattern**:
- Each subagent gets own context window and tool permissions
- Keeps main session from being polluted
- Investigate specific questions without context overhead
- Store in `.claude/agents/` and version-control
- Alternative: Put all context in CLAUDE.md and let main agent delegate

**.claudeignore File**:
- Exclude irrelevant directories (node_modules, build artifacts, data files)
- Prevents wasting context on non-relevant files
- Similar to .gitignore pattern

**Session Hygiene**:
- Start fresh sessions for different contexts
- Prevents context mixing
- Maintains focus on specific areas
- Clear separation of concerns

**Proactive Planning**:
- Ask Claude to plan before implementation
- Use "think" keyword to trigger extended thinking mode
- Additional computation time to evaluate alternatives
- Reduces trial-and-error cycles

---

## 3. Best Practices

### 3.1 Treat Context as Finite Resource

**Core Principle**: Context has diminishing marginal returns. Every token depletes the "attention budget."

**Key Insights**:
- LLMs have limited "working memory" like humans
- Not all tokens receive equal attention
- More context ≠ better performance
- Balance between "enough" and "too much"

**Optimization Goals**:
- Include not less (nothing critical missing)
- Include not more (no overwhelm or distraction)
- Find the sweet spot for optimal performance

### 3.2 Context Quality Over Quantity

**Common Pitfalls**:

1. **Context Bloat**: Filling context window with maximum information
   - Worse performance despite more data
   - Higher costs
   - Model distraction from key information

2. **Insufficient Context**: Not providing enough background
   - #1 mistake in LLM usage
   - Leads to hallucinations
   - Generic, low-quality responses
   - Incorrect/nonexistent APIs and packages

3. **Lost-in-the-Middle Effect**: Important context in middle of prompt
   - LLMs exhibit primacy and recency bias
   - Beginning and end weighted more heavily
   - Middle content often undervalued
   - Solution: Place critical information at boundaries

**Context Problems**:

- **Context Distraction**: Context so long model over-focuses on it, neglecting training knowledge
- **Context Confusion**: Superfluous information generates low-quality response
- **Context Clash**: New information conflicts with other prompt information
- **Context Rot**: As tokens increase, ability to recall information decreases

### 3.3 Context Curation Strategy

**Proactive Management**:
- Constantly curate and shape context shared with LLM
- Regular review and pruning of conversation history
- Remove obsolete, redundant, or resolved items
- Focus on active, relevant, and future-needed information

**RAG (Retrieval-Augmented Generation)**:
- Selectively add relevant information for better responses
- Search knowledge base for pertinent chunks
- Include only semantically similar content
- Combine LLM's trained knowledge with current data

**Tool Loadout Optimization**:
- Select only relevant tool definitions
- Don't include entire tool library if only using subset
- Match tools to current task/phase
- Reduce token overhead from unused tools

**Context Bucketing** (revisited):
```
Core Context:
  - System prompts
  - Project architecture
  - Key decisions
  - Persistent constraints

Working Context:
  - Active task details
  - Recent tool outputs
  - Current code being modified
  - Immediate conversation

Transient Context:
  - One-time tool results
  - Exploratory queries
  - Temporary calculations
  - Fully-resolved issues
```

### 3.4 Cost Optimization

**Token Budget Management**:
- Dynamic adjustment of reasoning tokens based on problem complexity
- Can reduce output token costs by 68.64% while maintaining performance
- Monitor usage against budgets
- Set alerts/throttling at thresholds

**Smart Caching**:
- Single highest-ROI optimization for LLM applications
- Reduce API calls by 70%+
- Improve response times from 2-3s to <100ms
- Cache semantically similar queries
- Return cached responses above similarity threshold

**Model Routing**:
- Route requests to appropriate model tier
- Use smaller/cheaper models for simple tasks
- Reserve powerful models for complex reasoning
- Reduces processing time and token usage

**Prompt Optimization**:
- A/B testing different prompt variants
- Continuous refinement to reduce token consumption
- Track quality metrics alongside token reduction
- Gradual rollouts of optimized prompts

**Batch Processing**:
- Up to 50% cost discounts (Claude, OpenAI)
- Use for latency-tolerant workloads
- Process multiple requests in single API call
- Trade latency for cost savings

**Common Token Waste Sources**:
- Overly long prompts from unoptimized design
- Excessive context windows (full documents vs. relevant chunks)
- Repeated information across turns
- Unused tool definitions

**Systematic Cost Reduction**:
> Organizations routinely achieve 60-80% cost reductions through token management, intelligent caching, prompt optimization, and strategic model selection while maintaining or improving performance.

### 3.5 Performance Considerations

**Response Quality**:
- Specific instructions improve success rate significantly
- Clear directions reduce course corrections
- Well-structured prompts > lengthy prompts
- Context relevance > context volume

**Latency Management**:
- Summarization adds pause unless pre-computed
- Background processes can prepare summaries proactively
- Semantic search adds retrieval overhead
- Balance between latency and context quality

**Scalability**:
- Vector databases (ChromaDB) enable semantic search at scale
- Hierarchical budget management for multi-tenant systems
- Virtual keys for granular cost tracking
- Throttling and alerts for budget control

**Reliability**:
- Avoid cache invalidation with stable context structure
- Monitor hit rates and adjust breakpoints
- Handle edge cases (context overflow, retrieval failures)
- Graceful degradation when context limits reached

---

## 4. Implementation Patterns

### 4.1 Sliding Window Implementation (Python)

**Basic Pattern**:

```python
from collections import deque
from typing import List, Dict

class SlidingWindowMemory:
    """Simple sliding window for conversation history."""

    def __init__(self, window_size: int = 10):
        """
        Initialize sliding window memory.

        Args:
            window_size: Number of message pairs to keep
        """
        self.window_size = window_size
        self.messages = deque(maxlen=window_size * 2)  # user + assistant

    def add_message(self, role: str, content: str):
        """Add a message to the window."""
        self.messages.append({"role": role, "content": content})

    def get_context(self) -> List[Dict[str, str]]:
        """Get current context window."""
        return list(self.messages)

    def clear(self):
        """Clear the window."""
        self.messages.clear()
```

**With Token Counting**:

```python
import tiktoken
from collections import deque

class TokenAwareSlidingWindow:
    """Sliding window with token budget management."""

    def __init__(self, max_tokens: int = 4000, model: str = "claude-3-5-sonnet-20241022"):
        self.max_tokens = max_tokens
        self.encoding = tiktoken.encoding_for_model(model)
        self.messages = deque()

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))

    def add_message(self, role: str, content: str):
        """Add message and remove old ones if exceeding token limit."""
        self.messages.append({"role": role, "content": content})

        # Remove oldest messages until under token limit
        while self._total_tokens() > self.max_tokens and len(self.messages) > 1:
            self.messages.popleft()

    def _total_tokens(self) -> int:
        """Calculate total tokens in current messages."""
        return sum(
            self.count_tokens(msg["content"])
            for msg in self.messages
        )

    def get_context(self) -> List[Dict[str, str]]:
        """Get current context."""
        return list(self.messages)
```

### 4.2 Structured Summarization (Python)

```python
from typing import List, Dict
import anthropic

class StructuredSummarizer:
    """Summarize conversation with structured template."""

    SUMMARY_TEMPLATE = """
Summarize the following conversation using this structure:

## Key Decisions Made
- List important decisions and their rationale

## Files Created/Modified
- List all files created or modified with brief description

## Unresolved Issues
- List any bugs, errors, or problems not yet resolved

## Action Items
- List pending tasks or next steps

## Technical Context
- Important technical details, dependencies, configurations

Conversation to summarize:
{conversation}

Provide a concise, structured summary:
"""

    def __init__(self, client: anthropic.Anthropic):
        self.client = client

    def summarize(
        self,
        messages: List[Dict[str, str]],
        model: str = "claude-3-5-sonnet-20241022"
    ) -> str:
        """Generate structured summary of conversation."""

        # Format conversation
        conversation = "\n\n".join(
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in messages
        )

        # Generate summary
        response = self.client.messages.create(
            model=model,
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": self.SUMMARY_TEMPLATE.format(
                    conversation=conversation
                )
            }]
        )

        return response.content[0].text

    def compact_conversation(
        self,
        messages: List[Dict[str, str]],
        keep_recent: int = 5
    ) -> List[Dict[str, str]]:
        """
        Compact conversation by summarizing older messages.

        Args:
            messages: Full conversation history
            keep_recent: Number of recent messages to keep verbatim

        Returns:
            Compacted message list with summary + recent messages
        """
        if len(messages) <= keep_recent:
            return messages

        # Split into old (to summarize) and recent (to keep)
        old_messages = messages[:-keep_recent]
        recent_messages = messages[-keep_recent:]

        # Summarize old messages
        summary = self.summarize(old_messages)

        # Create new context with summary + recent
        return [
            {"role": "user", "content": f"[Previous conversation summary]\n\n{summary}"}
        ] + recent_messages
```

### 4.3 Hybrid Context Manager (Python)

```python
from typing import List, Dict, Optional
import chromadb
from collections import deque

class HybridContextManager:
    """
    Hybrid context management with:
    - Sliding window for recent context
    - Vector database for long-term semantic memory
    - Structured summarization when needed
    """

    def __init__(
        self,
        recent_window_size: int = 10,
        max_tokens: int = 100000,
        compaction_threshold: float = 0.7,
        chroma_path: str = "./chroma_db"
    ):
        # Recent context (sliding window)
        self.recent_messages = deque(maxlen=recent_window_size * 2)
        self.max_tokens = max_tokens
        self.compaction_threshold = compaction_threshold

        # Long-term memory (ChromaDB)
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.memory_collection = self.chroma_client.get_or_create_collection(
            name="conversation_memory",
            metadata={"hnsw:space": "cosine"}
        )

        # Core context (always included)
        self.core_context: List[Dict[str, str]] = []

        # Current token usage
        self.current_tokens = 0

    def set_core_context(self, messages: List[Dict[str, str]]):
        """Set core context that always persists."""
        self.core_context = messages

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add message to recent context."""
        message = {"role": role, "content": content}
        self.recent_messages.append(message)

        # Update token count (simplified - use tiktoken in production)
        self.current_tokens = self._estimate_tokens()

        # Check if compaction needed
        if self.current_tokens > self.max_tokens * self.compaction_threshold:
            self._compact()

    def _estimate_tokens(self) -> int:
        """Estimate total tokens (simplified)."""
        total = sum(len(msg["content"]) // 4 for msg in self.core_context)
        total += sum(len(msg["content"]) // 4 for msg in self.recent_messages)
        return total

    def _compact(self):
        """Move older messages to vector store and optionally summarize."""
        if len(self.recent_messages) <= 4:
            return  # Too few to compact

        # Move oldest half to long-term memory
        to_archive = len(self.recent_messages) // 2
        archived = []

        for _ in range(to_archive):
            if self.recent_messages:
                msg = self.recent_messages.popleft()
                archived.append(msg)

        # Store in ChromaDB for semantic retrieval
        if archived:
            self._store_in_vector_db(archived)

        # Recalculate tokens
        self.current_tokens = self._estimate_tokens()

    def _store_in_vector_db(self, messages: List[Dict[str, str]]):
        """Store messages in ChromaDB for later retrieval."""
        import hashlib
        import time

        for i, msg in enumerate(messages):
            # Create unique ID
            msg_id = hashlib.md5(
                f"{msg['content']}{time.time()}{i}".encode()
            ).hexdigest()

            # Store in collection
            self.memory_collection.add(
                documents=[msg["content"]],
                metadatas=[{"role": msg["role"]}],
                ids=[msg_id]
            )

    def retrieve_relevant_memory(
        self,
        query: str,
        n_results: int = 3
    ) -> List[Dict[str, str]]:
        """Retrieve semantically relevant past messages."""
        results = self.memory_collection.query(
            query_texts=[query],
            n_results=n_results
        )

        if not results["documents"]:
            return []

        # Format as messages
        memories = []
        for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
            memories.append({
                "role": metadata["role"],
                "content": doc
            })

        return memories

    def get_full_context(
        self,
        current_query: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Get complete context for LLM call.

        Combines:
        - Core context (always)
        - Relevant long-term memory (if query provided)
        - Recent messages (sliding window)
        """
        context = list(self.core_context)

        # Add relevant memories if query provided
        if current_query:
            relevant_memories = self.retrieve_relevant_memory(current_query)
            if relevant_memories:
                context.append({
                    "role": "user",
                    "content": "[Relevant context from past conversations]\n\n" +
                              "\n\n".join(m["content"] for m in relevant_memories)
                })

        # Add recent messages
        context.extend(list(self.recent_messages))

        return context

    def clear_recent(self):
        """Clear recent context (e.g., between tasks)."""
        self.recent_messages.clear()
        self.current_tokens = self._estimate_tokens()
```

### 4.4 Context Bucketing Pattern (Python)

```python
from enum import Enum
from typing import List, Dict
from dataclasses import dataclass

class ContextBucket(Enum):
    """Context bucket types."""
    CORE = "core"          # Must persist
    WORKING = "working"    # Needed now, summarize later
    TRANSIENT = "transient"  # Use once, discard

@dataclass
class CategorizedMessage:
    """Message with bucket category."""
    role: str
    content: str
    bucket: ContextBucket
    metadata: Dict = None

class BucketedContextManager:
    """Manage context using bucket categorization."""

    def __init__(self):
        self.core: List[CategorizedMessage] = []
        self.working: List[CategorizedMessage] = []
        self.transient: List[CategorizedMessage] = []

    def add_message(
        self,
        role: str,
        content: str,
        bucket: ContextBucket,
        metadata: Dict = None
    ):
        """Add categorized message to appropriate bucket."""
        msg = CategorizedMessage(role, content, bucket, metadata)

        if bucket == ContextBucket.CORE:
            self.core.append(msg)
        elif bucket == ContextBucket.WORKING:
            self.working.append(msg)
        else:  # TRANSIENT
            self.transient.append(msg)

    def get_context_for_llm(
        self,
        include_transient: bool = True
    ) -> List[Dict[str, str]]:
        """Build context from buckets."""
        messages = []

        # Always include core
        messages.extend(
            {"role": msg.role, "content": msg.content}
            for msg in self.core
        )

        # Include working context
        messages.extend(
            {"role": msg.role, "content": msg.content}
            for msg in self.working
        )

        # Optionally include transient
        if include_transient:
            messages.extend(
                {"role": msg.role, "content": msg.content}
                for msg in self.transient
            )

        return messages

    def clear_transient(self):
        """Clear transient bucket after use."""
        self.transient.clear()

    def summarize_working(self, summarizer) -> str:
        """Summarize working context and move to core."""
        if not self.working:
            return ""

        # Convert to message format
        messages = [
            {"role": msg.role, "content": msg.content}
            for msg in self.working
        ]

        # Generate summary
        summary = summarizer.summarize(messages)

        # Add summary to core
        self.add_message(
            role="user",
            content=f"[Previous working context summary]\n\n{summary}",
            bucket=ContextBucket.CORE
        )

        # Clear working bucket
        self.working.clear()

        return summary
```

### 4.5 Anthropic Prompt Caching Pattern (Python)

```python
import anthropic

def create_cached_system_prompt(
    client: anthropic.Anthropic,
    system_content: str,
    tools: List[Dict],
    conversation: List[Dict[str, str]]
) -> Dict:
    """
    Use prompt caching for stable system context.

    Cache structure:
    1. System prompt (cached)
    2. Tools (cached)
    3. Recent conversation (not cached)
    """

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4096,
        system=[
            {
                "type": "text",
                "text": system_content,
                "cache_control": {"type": "ephemeral"}  # Cache system prompt
            }
        ],
        tools=tools,  # Tools automatically eligible for caching
        messages=conversation  # Recent conversation not cached
    )

    return response

# Example usage with project context
def create_project_context(client: anthropic.Anthropic):
    """Create reusable project context with caching."""

    system_prompt = """You are a coding assistant for the Elpis project.

Project Architecture:
- Python 3.10+ codebase
- Uses Pydantic for configuration
- ChromaDB for memory storage
- FastAPI for API layer

Code Style:
- Type hints required
- Docstrings for all public functions
- pytest for testing
- Black for formatting

Current Task Context:
Working on Phase 2 - implementing context management system.
"""

    tools = [
        {
            "name": "read_file",
            "description": "Read a file from the codebase",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"}
                },
                "required": ["path"]
            }
        },
        # More tools...
    ]

    # First call - establishes cache
    response1 = create_cached_system_prompt(
        client=client,
        system_content=system_prompt,
        tools=tools,
        conversation=[
            {"role": "user", "content": "Show me the config structure"}
        ]
    )

    # Subsequent calls - use cached system prompt & tools
    # Only pay for new conversation tokens
    response2 = create_cached_system_prompt(
        client=client,
        system_content=system_prompt,  # Read from cache (90% cheaper)
        tools=tools,  # Read from cache
        conversation=[
            {"role": "user", "content": "Show me the config structure"},
            {"role": "assistant", "content": response1.content[0].text},
            {"role": "user", "content": "Now show me the memory system"}
        ]
    )

    return response2
```

---

## 5. Libraries and Tools

### 5.1 Letta (formerly MemGPT)

**Overview**: Memory-first framework for stateful LLM agents with self-editing memory capabilities.

**Key Features**:
- Two-tier memory architecture (in-context + out-of-context)
- Self-editing memory through tool use
- LLM Operating System (OS) concept
- Moves data in/out of context window to manage memory

**Memory Architecture**:
```
Main Context (In-Context):
  - Active working memory
  - Currently loaded information
  - Limited by context window

External Context (Out-of-Context):
  - Long-term storage
  - Retrieved on-demand
  - Unlimited capacity
```

**Python Integration**:
```bash
pip install letta
```

**Recent Updates** (v0.4.1):
- Integration with Composio, LangChain, CrewAI tools
- Python + TypeScript SDKs
- API-first architecture
- Scalable agent deployment

**Use Cases**:
- Long-running agents requiring persistent memory
- Multi-session conversations
- Agents that learn and improve over time
- Context-aware intelligent systems

**References**:
- Paper: "MemGPT: Towards LLMs as Operating Systems"
- Framework: Letta (www.letta.com)

### 5.2 LangChain

**Overview**: Comprehensive framework for building LLM applications with extensive memory management utilities.

**Memory Types**:

1. **ConversationBufferMemory**:
   - Simple in-memory buffer of last N messages
   - Zero setup, great for prototyping
   - No persistence, chronological buffering only

2. **ConversationSummaryMemory**:
   - Automatically summarizes conversation over time
   - Uses LLM to generate summaries
   - Balances detail with token efficiency

3. **ConversationBufferWindowMemory**:
   - Sliding window implementation
   - Keeps only last K interactions
   - Simple and predictable

4. **ConversationVectorStoreMemory**:
   - Stores conversation in vector database
   - Retrieves relevant context based on semantic similarity
   - Best for long conversations with diverse topics

**Integration with ChromaDB**:
```python
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Vector store for semantic memory
vectorstore = Chroma(
    embedding_function=OpenAIEmbeddings(),
    persist_directory="./chroma_db"
)

# Combine with LangChain memory
memory = ConversationVectorStoreMemory(
    vectorstore=vectorstore,
    memory_key="chat_history"
)
```

**RAG Implementation**:
```python
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.llms import Anthropic

# Load documents into ChromaDB
vectorstore = Chroma.from_documents(
    documents=split_documents,
    embedding=embedding_model
)

# Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=Anthropic(model="claude-3-5-sonnet-20241022"),
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)
```

### 5.3 ChromaDB

**Overview**: Open-source vector database designed for LLM applications, particularly RAG workflows.

**Key Features**:
- Stores and retrieves high-dimensional embeddings efficiently
- Built-in embedding, text search, vector search
- Document storage and multimodal search
- Persistent and in-memory modes

**Installation**:
```bash
pip install chromadb
```

**Basic Usage**:
```python
import chromadb

# Create client
client = chromadb.PersistentClient(path="./chroma_db")

# Create collection
collection = client.get_or_create_collection(
    name="conversation_memory",
    metadata={"hnsw:space": "cosine"}
)

# Add documents
collection.add(
    documents=["This is a conversation message"],
    metadatas=[{"role": "user", "timestamp": "2026-01-12"}],
    ids=["msg_001"]
)

# Query for relevant context
results = collection.query(
    query_texts=["What did we discuss?"],
    n_results=3
)
```

**RAG Pattern**:
1. Split documents into chunks
2. Generate embeddings for each chunk
3. Store embeddings in ChromaDB
4. On user query:
   - Embed query
   - Search ChromaDB for similar chunks
   - Retrieve top-k most relevant chunks
   - Include in LLM context with query
   - Generate response with augmented context

**Semantic Caching**:
```python
def semantic_cache_lookup(query: str, collection, threshold: float = 0.9):
    """Check cache for semantically similar queries."""
    results = collection.query(
        query_texts=[query],
        n_results=1
    )

    if results["distances"][0][0] < (1 - threshold):  # Cosine similarity
        # Cache hit - return cached response
        return results["metadatas"][0][0]["response"]

    # Cache miss - need to call LLM
    return None
```

**Benefits for Elpis**:
- Semantic search over conversation history
- Long-term memory beyond context window
- Efficient retrieval of relevant past context
- Persistent storage across sessions
- Native Python integration

### 5.4 Other Memory Frameworks

**Zep**:
- Integrates with LangChain, LangGraph, various LLM APIs
- Focus on conversational memory and user session management
- Production-ready with scalability features

**Mem0**:
- Native integration with OpenAI, Claude, LangChain
- Simplified API for memory management
- Good for rapid prototyping

**Comparison for Elpis**:
```
Letta: Best for complex agents with sophisticated memory needs
LangChain: Best for standard integrations and quick development
ChromaDB: Best for semantic search and RAG patterns
Zep: Best for production deployments with session management
Mem0: Best for simple use cases and prototyping
```

---

## 6. Common Pitfalls to Avoid

### 6.1 Context Bloat

**Problem**: Filling context window with maximum possible information.

**Consequences**:
- Degraded performance despite more data
- Higher API costs
- Model distraction from critical information
- Slower response times

**Solution**:
- Curate context aggressively
- Include only relevant information
- Remove redundant or resolved content
- Monitor token usage continuously

### 6.2 Insufficient Context

**Problem**: Not providing enough background information.

**Consequences**:
- LLM hallucinations
- Generic, low-quality responses
- Incorrect assumptions about APIs/packages
- Boilerplate code instead of specific solutions

**Solution**:
- Include project structure and architecture
- Provide relevant code context
- Specify constraints and requirements clearly
- Use CLAUDE.md or similar for persistent context

### 6.3 Lost-in-the-Middle Effect

**Problem**: Placing critical information in middle of long prompt.

**Consequences**:
- Important context underweighted by model
- Primacy and recency bias favor beginning/end
- Information effectively "invisible" to model

**Solution**:
- Place critical info at beginning or end
- Use explicit markers ("IMPORTANT:", "KEY REQUIREMENT:")
- Structure prompts with clear sections
- Keep most important content in first/last 20%

### 6.4 Context Rot

**Problem**: As conversation grows, model's recall degrades across entire context.

**Consequences**:
- Inconsistent responses
- Forgotten earlier decisions
- Repeated questions or contradictions
- Quality degradation over long sessions

**Solution**:
- Proactive compaction at 70% threshold
- Regular summarization of completed topics
- Clear session boundaries
- Use `/compact` or `/clear` commands

### 6.5 Naive Summarization

**Problem**: Generic, unstructured summarization loses critical details.

**Consequences**:
- File paths, decisions silently dropped
- Gradual information loss over multiple summaries
- Cannot recover specific details later
- "Telephone game" effect with repeated summarization

**Solution**:
- Use structured summary templates
- Dedicate sections to specific information types
- Keep recent messages in raw format
- Validate summaries before discarding originals

### 6.6 Cache Invalidation

**Problem**: Frequent changes to cached content negate caching benefits.

**Consequences**:
- Unexpected cost increases
- No cache hit rate improvement
- Wasted cache setup effort
- Slower responses than uncached

**Solution**:
- Separate stable (cached) from dynamic (uncached) content
- Place cache breakpoints strategically
- Monitor cache hit rates
- Avoid changing cached system prompts mid-conversation

### 6.7 Ignoring Tool Output Size

**Problem**: Including full verbose tool outputs in context.

**Consequences**:
- Rapid context window consumption
- Most tool output often redundant
- Difficult to find signal in noise

**Solution**:
- Reversible compaction: Store results externally, reference by path/ID
- Summarize tool outputs before adding to context
- Keep only essential information from tool results
- Use context editing to remove stale tool outputs

### 6.8 No Context Categorization

**Problem**: Treating all context as equally important.

**Consequences**:
- Important information compacted/discarded prematurely
- Temporary data persists unnecessarily
- Inefficient token usage

**Solution**:
- Implement context bucketing (Core/Working/Transient)
- Explicitly manage what goes where
- Different retention policies per bucket
- Clear criteria for categorization

### 6.9 Synchronous Summarization

**Problem**: Pausing conversation to generate summary.

**Consequences**:
- User-facing latency
- Poor user experience
- Wasted time during summarization

**Solution**:
- Background processes for proactive summarization
- Pre-compute summaries before threshold reached
- Use faster models for summarization
- Consider async summarization during user think time

### 6.10 Over-Reliance on Context Window Size

**Problem**: Assuming larger context window = better performance.

**Consequences**:
- False sense of security
- Lack of context management discipline
- Higher costs without proportional benefit
- Context rot still occurs

**Solution**:
- Manage context actively regardless of window size
- Quality over quantity principle
- Implement compaction even with large windows
- Monitor performance metrics, not just token counts

---

## 7. Recommendations for Elpis Implementation

### 7.1 Architecture Design

**Tiered Context Management System**:

```
Layer 1: Core Context (Persistent)
  - System prompts
  - Project architecture (CLAUDE.md equivalent)
  - User preferences
  - Configuration
  - Tool definitions
  Storage: File-based, version-controlled

Layer 2: Working Context (Active Session)
  - Recent conversation (sliding window: 10-20 turns)
  - Current task details
  - Active file contents
  - Recent tool outputs
  Storage: In-memory deque

Layer 3: Long-Term Memory (Semantic Retrieval)
  - Past conversations
  - Historical decisions
  - Learned patterns
  - User interaction history
  Storage: ChromaDB vector database

Layer 4: Transient Context (Temporary)
  - One-time tool results
  - Exploratory queries
  - Intermediate calculations
  Storage: In-memory, cleared after use
```

**Context Flow**:
```
New Message → Categorize → Appropriate Layer
             ↓
      Monitor Token Budget
             ↓
   70% Threshold? → Compact Working Context
             ↓
   Archive to Long-Term Memory (ChromaDB)
             ↓
   Keep Recent in Sliding Window
```

### 7.2 Implementation Priorities

**Phase 1: Foundation**
1. Implement context bucketing system (Core/Working/Transient)
2. Add token counting and budget monitoring
3. Create sliding window for recent context
4. Basic compaction: Remove transient, keep core

**Phase 2: Intelligent Management**
5. Integrate ChromaDB for long-term memory
6. Implement semantic retrieval for relevant context
7. Add structured summarization with templates
8. Context editing to remove stale tool outputs

**Phase 3: Optimization**
9. Prompt caching for stable system prompts
10. Background summarization processes
11. Semantic caching for repeated queries
12. Cost tracking and optimization

**Phase 4: Advanced Features**
13. Token-budget-aware response generation
14. Automatic context categorization (ML-based)
15. Hierarchical summarization (multi-tier)
16. Context health metrics and monitoring

### 7.3 Specific Strategies

**Reversible Compaction First**:
- Primary strategy: Store file contents externally, reference by path
- Tool outputs: Save to files, include path in context
- Code changes: Reference file + line numbers, not full content
- Only summarize when reversible compaction insufficient

**Structured Summarization Template**:
```markdown
## Session Summary - [Timestamp]

### Key Decisions
- Decision 1: Rationale
- Decision 2: Rationale

### Files Modified
- path/to/file1.py: Description of changes
- path/to/file2.py: Description of changes

### Unresolved Issues
- Issue 1: Details and context
- Issue 2: Details and context

### Technical Context
- Dependencies added/changed
- Configuration changes
- Important architectural notes

### Next Steps
- Action item 1
- Action item 2
```

**ChromaDB Integration**:
```python
# Store conversation turns
collection.add(
    documents=[message_content],
    metadatas=[{
        "role": role,
        "timestamp": timestamp,
        "session_id": session_id,
        "category": category,  # decision, code, discussion, etc.
        "importance": importance_score  # 1-10
    }],
    ids=[unique_message_id]
)

# Retrieve relevant context
def get_relevant_context(query: str, category: Optional[str] = None):
    where_filter = {"category": category} if category else None

    results = collection.query(
        query_texts=[query],
        n_results=5,
        where=where_filter
    )

    return results
```

**Monitoring and Metrics**:
- Track token usage per layer (Core/Working/Long-term/Transient)
- Monitor compaction frequency and effectiveness
- Measure cache hit rates
- Track cost per conversation turn
- Alert when approaching limits

### 7.4 Configuration

**Recommended Settings**:
```python
CONTEXT_CONFIG = {
    # Token budgets
    "max_total_tokens": 100000,  # Conservative for Claude Sonnet
    "compaction_threshold": 0.7,  # Compact at 70%

    # Sliding window
    "recent_window_size": 15,  # 15 message pairs

    # ChromaDB
    "vector_db_path": "./data/memory",
    "retrieval_top_k": 5,
    "similarity_threshold": 0.75,

    # Summarization
    "summarize_after_turns": 50,
    "keep_recent_raw": 10,  # Keep last 10 turns verbatim

    # Caching
    "cache_system_prompts": True,
    "cache_ttl_minutes": 5,

    # Categories
    "importance_threshold": 7,  # 7+ = core context
    "transient_lifetime_turns": 1,
}
```

### 7.5 Testing Strategy

**Unit Tests**:
- Token counting accuracy
- Sliding window behavior
- Compaction logic
- Categorization rules

**Integration Tests**:
- ChromaDB storage and retrieval
- End-to-end context flow
- Multi-turn conversations
- Cache behavior

**Performance Tests**:
- Context retrieval latency
- Summarization speed
- Token budget accuracy
- Memory usage under load

**Quality Tests**:
- Information preservation in summaries
- Relevance of retrieved context
- Cache hit rates
- Cost efficiency metrics

### 7.6 Migration Path

**From Current State**:
1. Audit existing context management (if any)
2. Identify current pain points and token waste
3. Implement basic bucketing and monitoring
4. Add ChromaDB for long-term memory
5. Gradually introduce compaction and summarization
6. Optimize based on metrics

**Backwards Compatibility**:
- Support legacy conversation format during transition
- Migrate historical conversations to ChromaDB
- Maintain old API while introducing new system
- Deprecate old approach after validation

---

## 8. References and Sources

### Anthropic Documentation
- [Context Windows - Claude Docs](https://platform.claude.com/docs/en/build-with-claude/context-windows)
- [Context Editing - Claude Docs](https://platform.claude.com/docs/en/build-with-claude/context-editing)
- [Managing Context on Claude Developer Platform](https://anthropic.com/news/context-management)
- [Effective Context Engineering for AI Agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- [Claude Code Best Practices](https://www.anthropic.com/engineering/claude-code-best-practices)
- [Prompt Caching - Claude Docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- [Prompt Caching with Claude](https://www.anthropic.com/news/prompt-caching)

### Technical Articles and Guides
- [Context Window Management Strategies (Maxim AI)](https://www.getmaxim.ai/articles/context-window-management-strategies-for-long-context-ai-agents-and-chatbots/)
- [Context Window Management Strategies (APXML)](https://apxml.com/courses/langchain-production-llm/chapter-3-advanced-memory-management/context-window-management)
- [LLM Context Windows: Why They Matter and 5 Solutions (Kolena)](https://www.kolena.com/guides/llm-context-windows-why-they-matter-and-5-solutions-for-context-limits/)
- [Context Engineering for AI Agents: Part 2](https://www.philschmid.de/context-engineering-part-2)
- [Cutting Through the Noise: Smarter Context Management (JetBrains)](https://blog.jetbrains.com/research/2025/12/efficient-context-management/)
- [How We Extended LLM Conversations by 10x](https://dev.to/amitksingh1490/how-we-extended-llm-conversations-by-10x-with-intelligent-context-compaction-4h0a)
- [Long Context Compaction for AI Agents - Design Principles](https://medium.com/@revoir07/long-context-compaction-for-ai-agents-part-1-design-principles-2bf4a5748154)
- [Claude Code Context Management (ClaudeCode.io)](https://claudecode.io/guides/context-management)
- [Managing Claude Code's Context (CometAPI)](https://www.cometapi.com/managing-claude-codes-context/)
- [Managing Claude Code Context (MCPcat)](https://mcpcat.io/guides/managing-claude-code-context/)

### Cost Optimization
- [LLM Cost Optimization: Stop Token Spend Waste](https://www.kosmoy.com/post/llm-cost-management-stop-burning-money-on-tokens)
- [Cost Optimization for LLM Applications (JetThoughts)](https://jetthoughts.com/blog/cost-optimization-llm-applications-token-management/)
- [The Technical Guide to Managing LLM Costs (Maxim AI)](https://www.getmaxim.ai/articles/the-technical-guide-to-managing-llm-costs-strategies-for-optimization-and-roi/)
- [10 Strategies to Reduce LLM Costs (Uptech)](https://www.uptech.team/blog/how-to-reduce-llm-costs)
- [LLM Cost Optimization Guide 2025](https://futureagi.com/blogs/llm-cost-optimization-2025)

### Research Papers
- [Token-Budget-Aware LLM Reasoning (ACL 2025)](https://aclanthology.org/2025.findings-acl.1274/)
- [Token-Budget-Aware LLM Reasoning (arXiv)](https://arxiv.org/abs/2412.18547)
- [Acon: Optimizing Context Compression for Long-horizon LLM Agents](https://arxiv.org/html/2510.00615v1)
- [Don't Break the Cache: Prompt Caching Evaluation](https://arxiv.org/html/2601.06007)

### Memory Frameworks
- [Letta (formerly MemGPT)](https://www.letta.com/)
- [MemGPT: Towards LLMs as Operating Systems](https://www.leoniemonigatti.com/papers/memgpt.html)
- [Letta Documentation](https://docs.letta.com/)
- [Agent Memory: How to Build Agents that Learn](https://www.letta.com/blog/agent-memory)
- [Letta GitHub Repository](https://github.com/letta-ai/letta)

### Vector Databases and RAG
- [Embeddings and Vector Databases with ChromaDB (Real Python)](https://realpython.com/chromadb-vector-database/)
- [Semantic Caching and Memory Patterns (Dataquest)](https://www.dataquest.io/blog/semantic-caching-and-memory-patterns-for-vector-databases/)
- [RAG with ChromaDB Tutorial](https://promptlyai.in/rag-made-simple/)
- [Building Memory-Augmented AI Agents with LangChain](https://medium.com/@saurabhzodex/building-memory-augmented-ai-agents-with-langchain-part-1-2c21cc8050da)
- [Conversational Memory for LLMs with Langchain (Pinecone)](https://www.pinecone.io/learn/series/langchain/langchain-conversational-memory/)

### Implementation Examples
- [GitHub - MemoryLLM (Hybrid Context Management)](https://github.com/maranone/MemoryLLM)
- [Infinite Chat Using Sliding Window (Microsoft)](https://devblogs.microsoft.com/surface-duo/android-openai-chatgpt-16/)
- [Implementing Memory in LLM Applications (Codecademy)](https://www.codecademy.com/article/implementing-memory-in-llm-applications-using-lang-chain)

### Best Practices and Pitfalls
- [LLM Context Management Guide (16x Engineer)](https://eval.16x.engineer/blog/llm-context-management-guide)
- [Doing Real Work With LLMs: Context Management](https://www.jonstokes.com/p/doing-real-work-with-llms-how-to)
- [Top Techniques to Manage Context Lengths](https://agenta.ai/blog/top-6-techniques-to-manage-context-length-in-llms)
- [How to Fix Your Context](https://www.dbreunig.com/2025/06/26/how-to-fix-your-context.html)
- [LLM Context Engineering: A Practical Guide](https://medium.com/the-low-end-disruptor/llm-context-engineering-a-practical-guide-248095d4bf71)

---

## Conclusion

Context window management is critical for building robust, cost-effective, long-running LLM applications. The key principles are:

1. **Treat context as a finite, valuable resource** - Quality over quantity
2. **Prefer reversible compaction over lossy summarization** - Minimize information loss
3. **Use tiered storage** - Core/Working/Long-term/Transient buckets
4. **Implement semantic retrieval** - ChromaDB for relevant context on-demand
5. **Monitor proactively** - Compact at 70% threshold, track metrics
6. **Optimize costs** - Caching, batching, appropriate model selection

For Elpis Phase 2, the recommended approach is a hybrid system combining:
- Sliding window for recent context
- ChromaDB for long-term semantic memory
- Structured summarization when needed
- File-based storage for reversible compaction
- Prompt caching for static content

This multi-layered architecture balances performance, cost, and information preservation while supporting the complex agentic workflows Elpis aims to enable.

---

**Report Status**: Complete
**Next Steps**: Share findings with research team and integrate into Phase 2 architecture design.
