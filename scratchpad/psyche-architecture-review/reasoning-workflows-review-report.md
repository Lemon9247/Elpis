# Reasoning Workflows Review Report

**Agent**: Reasoning Workflows Review Agent
**Date**: 2026-01-16
**Task**: Research LLM reasoning workflows and "thinking before responding" implementation patterns

---

## Executive Summary

This report examines how modern LLMs implement reasoning capabilities and how coding agents display/hide thinking processes. The goal is to inform Psyche's implementation of a reasoning workflow where the LLM "thinks" about prompts before responding, with the ability to show/hide this reasoning.

Key findings:
1. **Reasoning models** (o1, DeepSeek-R1, Claude's extended thinking) use internal chain-of-thought that is processed before the final response
2. **Coding agents** typically show reasoning in collapsible sections or separate panels
3. **Smaller models** require explicit prompting strategies to elicit reasoning
4. **Performance considerations** are critical for local models - reasoning adds latency

---

## 1. Reasoning Model Approaches

### 1.1 OpenAI o1/o3 Series

OpenAI's o1 and o3 models represent a paradigm shift in LLM reasoning. Key characteristics:

**Architecture:**
- Models are trained with reinforcement learning to generate internal "reasoning tokens"
- The model produces a hidden chain-of-thought before generating the visible response
- Reasoning tokens are not exposed to users but influence the final output
- The model learns to break down complex problems into steps

**Behavior:**
- Takes longer to respond (intentionally, as it "thinks")
- Shows "thinking" indicator during processing
- Produces more accurate responses on complex reasoning tasks
- Better at math, coding, and multi-step logical problems

**API Integration:**
- `reasoning_effort` parameter controls how much the model "thinks"
- Reasoning tokens count toward token limits but are not returned
- Some APIs expose summarized reasoning ("reasoning summary")

**Limitations:**
- Not available for local deployment
- Higher latency and cost
- Reasoning process is opaque

### 1.2 DeepSeek-R1

DeepSeek-R1 is an open-weights reasoning model that makes the thinking process transparent:

**Architecture:**
- Trained with reinforcement learning similar to o1
- Produces explicit `<think>...</think>` tags containing reasoning
- The thinking is visible in the output, not hidden
- Available in various sizes (1.5B to 70B+ parameters)

**Output Format:**
```
<think>
Let me analyze this step by step...
1. First, I need to understand the problem...
2. The key insight is...
3. Therefore, the approach should be...
</think>

Here is my response based on my analysis...
```

**Benefits for Local Deployment:**
- Open weights allow local inference
- Transparent reasoning enables debugging and trust
- Can be fine-tuned for specific domains
- Smaller distilled versions available (Qwen-based)

**Relevance to Psyche:**
- DeepSeek-R1's explicit `<think>` tags provide a natural pattern for implementing visible reasoning
- The distilled models (1.5B-7B) could potentially run locally
- The explicit thinking format makes UI display straightforward

### 1.3 Anthropic Claude Extended Thinking

Claude's "extended thinking" feature (available in Claude 3.5 and later):

**Implementation:**
- `thinking` block in API responses contains reasoning
- Controlled via `thinking` parameter in requests
- Thinking content uses a special budget separate from output tokens
- Thinking can be streamed progressively

**API Example:**
```json
{
  "model": "claude-3-5-sonnet-20241022",
  "max_tokens": 16000,
  "thinking": {
    "type": "enabled",
    "budget_tokens": 10000
  }
}
```

**Response Structure:**
```json
{
  "content": [
    {
      "type": "thinking",
      "thinking": "Let me work through this problem..."
    },
    {
      "type": "text",
      "text": "Based on my analysis..."
    }
  ]
}
```

**Key Features:**
- Separate token budget for thinking vs. response
- Thinking can be shown/hidden client-side
- Progressive streaming of both thinking and response
- Works with tool use (thinking occurs before tool calls)

---

## 2. Coding Agent Reasoning Display Patterns

### 2.1 Claude Code

Claude Code implements a sophisticated reasoning display:

**Approach:**
- Uses Claude's extended thinking feature
- Thinking is displayed in a collapsible section above the response
- Keybind (typically `Ctrl+T` or similar) toggles thinking visibility
- Thinking streams in real-time during generation

**UI Pattern:**
```
[Thinking] (collapsible)
  Analyzing the user's request...
  This appears to be a file editing task...
  I should first read the file to understand its structure...

[Response]
  I'll help you edit that file. Let me first read it...
```

**Implementation Notes:**
- Thinking panel uses distinct visual styling (dimmed, different color)
- Auto-collapses when generation completes (configurable)
- Thinking content is searchable/copyable

### 2.2 Cursor

Cursor's approach to reasoning:

**Features:**
- Shows "Agent thinking..." indicator during reasoning phase
- Reasoning appears in a separate panel/section
- Can be toggled via settings or keybind
- Tool use decisions shown as part of reasoning

**Pattern:**
- Inline thinking indicators (subtle, non-intrusive)
- Expandable sections for detailed reasoning
- Reasoning persists in conversation history (optional)

### 2.3 Aider

Aider's reasoning approach (for terminal-based interface):

**Implementation:**
- Uses prompting to elicit reasoning from non-reasoning models
- Shows "Thinking..." status during generation
- Reasoning often embedded in response with clear markers
- `--show-reasoning` flag for verbose output

**Prompting Strategy:**
```
Before making any changes, first explain your understanding of
the request and your planned approach. Then implement the changes.
```

### 2.4 OpenCode

Based on the existing research in the codebase:

**Pattern:**
- Backend orchestration separates reasoning from tool execution
- Custom system prompts per provider to elicit reasoning
- Reasoning logged but may not be directly displayed
- Focus on tool decision transparency

---

## 3. Chain-of-Thought Best Practices for Smaller Models

Since Psyche uses local models (likely 7B-14B parameters), explicit prompting strategies are essential.

### 3.1 Basic Chain-of-Thought Prompting

**Template:**
```
Think step by step before responding. First, analyze the request.
Then, consider what approach would be best. Finally, provide your response.
```

**Enhanced Template:**
```
Before responding, work through this systematically:
1. What is the user actually asking for?
2. What information or tools do I need?
3. What are potential approaches?
4. Which approach is best and why?

Then provide your response.
```

### 3.2 Structured Reasoning Prompts

**XML-Style Markers (Most Reliable):**
```
When given a task, first write your reasoning inside <thinking> tags,
then provide your response after the closing tag.

Example:
<thinking>
The user wants to... I should...
</thinking>

[Your actual response here]
```

This format is:
- Easy to parse programmatically
- Clear boundary between thinking and response
- Works well with smaller models
- Similar to DeepSeek-R1's approach

**JSON-Style (More Structured):**
```json
{
  "reasoning": "My step-by-step analysis...",
  "response": "The actual response to the user"
}
```

### 3.3 Few-Shot Examples

Including examples significantly improves smaller model reasoning:

```
User: What's the capital of France?

<thinking>
This is a straightforward factual question about geography.
France is a country in Western Europe.
Its capital city is well-known.
</thinking>

The capital of France is Paris.

---

User: [Actual user query]
```

### 3.4 Role-Based Reasoning

**Expert Role Prompt:**
```
You are a thoughtful assistant who always considers problems carefully.
Before responding, you think through:
- What is being asked
- What you know about the topic
- The best way to help

Show your thinking process, then give your response.
```

### 3.5 Self-Consistency Techniques

For complex tasks, generate multiple reasoning paths:

```
Consider this problem from multiple angles:
Approach 1: [First way to think about it]
Approach 2: [Alternative perspective]
Best approach: [Synthesis and decision]
```

---

## 4. Recommended Approach for Psyche

### 4.1 Architecture Overview

```
                    User Input
                         |
                         v
                 +---------------+
                 |   Psyche TUI  |
                 |  (User Input) |
                 +-------+-------+
                         |
                         v
                 +---------------+
                 | Reasoning     |
                 | Workflow      |
                 | Controller    |
                 +-------+-------+
                         |
          +--------------+---------------+
          |                              |
          v                              v
  +----------------+            +----------------+
  | Thinking Phase |            | Response Phase |
  | (Hidden/Shown) |  ------>   | (Always Shown) |
  +----------------+            +----------------+
          |                              |
          v                              v
  +----------------+            +----------------+
  | ThoughtPanel   |            | ChatView       |
  | (Collapsible)  |            | (Main Display) |
  +----------------+            +----------------+
```

### 4.2 Two-Phase Generation Approach

**Phase 1: Thinking**
- Generate reasoning with explicit `<thinking>` markers
- Stream to ThoughtPanel (if visible) or buffer (if hidden)
- Use lower temperature for consistent reasoning

**Phase 2: Response**
- Parse thinking output to extract response portion
- If no response portion, continue generation
- Stream to ChatView

**Prompt Structure:**
```
[System Prompt - includes reasoning instruction]

You are Psyche, a thoughtful AI assistant.

When responding to a request, first think through it inside <thinking> tags.
Consider:
- What the user is asking for
- What information or tools you need
- Your approach to solving it

After thinking, provide your response outside the tags.

Example:
<thinking>
The user wants help with X. I should consider Y and Z.
The best approach is to...
</thinking>

Here is my response based on my analysis...

---

[Tool descriptions and existing context]
```

### 4.3 UI Implementation for Textual TUI

**Existing Infrastructure:**
Psyche already has a `ThoughtPanel` widget that can be repurposed:

```python
# Current: Used for idle reflection thoughts
# Proposed: Also used for active reasoning display

class ThoughtPanel(RichLog):
    """Panel for displaying reasoning/thinking."""

    def add_thought(self, content: str, thought_type: str = "reflection"):
        # Existing method - add "reasoning" type
        type_colors = {
            "reflection": "cyan",
            "planning": "yellow",
            "reasoning": "green",  # NEW: Active reasoning
            ...
        }
```

**Toggle Mechanism:**
The app already has `ctrl+t` binding and `/thoughts` command:

```python
BINDINGS = [
    Binding("ctrl+t", "toggle_thoughts", "Thoughts"),
]

# Commands already support:
# /thoughts on
# /thoughts off
# /thoughts toggle
```

**Proposed Enhancement - Reasoning-Specific Commands:**
```
/thinking on     - Show reasoning for new messages
/thinking off    - Hide reasoning (still computed)
/thinking toggle - Toggle visibility
ctrl+r           - Quick toggle for reasoning display
```

### 4.4 Parsing Reasoning from Generation

**Parser Implementation:**
```python
import re

def parse_reasoning_response(text: str) -> tuple[str, str]:
    """
    Parse LLM output to separate thinking from response.

    Returns:
        (thinking_content, response_content)
    """
    # Pattern for <thinking>...</thinking> blocks
    thinking_pattern = r'<thinking>(.*?)</thinking>'
    match = re.search(thinking_pattern, text, re.DOTALL)

    if match:
        thinking = match.group(1).strip()
        # Response is everything after the thinking block
        response = text[match.end():].strip()
        return thinking, response
    else:
        # No thinking block - entire text is response
        return "", text
```

**Streaming Consideration:**
For streaming, need a state machine to track whether we're inside thinking tags:

```python
class ReasoningStreamParser:
    def __init__(self):
        self.in_thinking = False
        self.thinking_buffer = []
        self.response_buffer = []

    def process_token(self, token: str) -> tuple[str, str, str]:
        """
        Process a streaming token.

        Returns:
            (thinking_token, response_token, state)
            where state is "thinking", "response", or "transition"
        """
        # Check for tag transitions
        if "<thinking>" in token:
            self.in_thinking = True
            return ("", "", "transition")
        elif "</thinking>" in token:
            self.in_thinking = False
            return ("", "", "transition")

        if self.in_thinking:
            return (token, "", "thinking")
        else:
            return ("", token, "response")
```

### 4.5 Server-Side Changes (MemoryServer)

The `MemoryServer._process_user_input` method should be enhanced:

```python
async def _process_user_input_with_reasoning(self, text: str) -> None:
    """Process user input with explicit reasoning phase."""

    # 1. Add user message to context
    self._compactor.add_message(create_message("user", text))

    # 2. Generate with reasoning prompt
    messages = self._compactor.get_api_messages()

    # Inject reasoning instruction if not in system prompt
    if self.config.enable_reasoning:
        messages = self._inject_reasoning_instruction(messages)

    # 3. Stream generation with reasoning parser
    parser = ReasoningStreamParser()

    async for token in self.client.generate_stream(...):
        thinking_tok, response_tok, state = parser.process_token(token)

        if thinking_tok and self.on_thinking:
            self.on_thinking(thinking_tok)  # New callback

        if response_tok and self.on_token:
            self.on_token(response_tok)

    # 4. Store complete thinking for context (optional)
    if self.config.store_reasoning:
        full_thinking = parser.get_full_thinking()
        # Could store as a thought event or separate context entry
```

### 4.6 Performance Considerations for Local Models

**Latency Impact:**
- Reasoning adds ~30-100% to generation time
- For a 7B model: expect 2-5 seconds of thinking before response starts
- Consider progress indicators during thinking phase

**Token Budget:**
- Thinking tokens compete with response tokens
- Recommend: 512-1024 tokens for thinking, rest for response
- Implement hard limits to prevent runaway reasoning

**Quality vs. Speed Tradeoffs:**

| Configuration | Thinking Tokens | Response Tokens | Use Case |
|--------------|-----------------|-----------------|----------|
| Quick        | 256             | 2048            | Simple questions |
| Balanced     | 512             | 1536            | Default |
| Deep         | 1024            | 1024            | Complex tasks |

**Caching/Optimization:**
- Cache reasoning for similar queries (semantic similarity)
- Skip reasoning for follow-up questions in same context
- Allow user to disable reasoning per-message (`/quick [message]`)

### 4.7 Configuration Options

```python
@dataclass
class ReasoningConfig:
    """Configuration for reasoning workflow."""

    # Feature toggles
    enabled: bool = True           # Master toggle
    display_by_default: bool = False  # Show thinking by default

    # Token budgets
    thinking_tokens: int = 512     # Max tokens for thinking
    min_thinking_tokens: int = 50  # Minimum thinking length

    # Prompting
    use_xml_markers: bool = True   # Use <thinking> tags
    include_examples: bool = False # Include few-shot examples

    # Behavior
    skip_for_simple: bool = True   # Skip reasoning for short queries
    simple_query_threshold: int = 20  # Words below this = simple
    store_reasoning: bool = True   # Save reasoning to context

    # Performance
    reasoning_temperature: float = 0.3  # Lower for consistent reasoning
    response_temperature: float = 0.7   # Normal for response
```

---

## 5. Implementation Recommendations

### 5.1 Phase 1: Basic Reasoning (Minimal Changes)

1. **Update System Prompt**
   - Add reasoning instruction to `MemoryServer._build_system_prompt()`
   - Use `<thinking>` tag format

2. **Add Parsing**
   - Implement `parse_reasoning_response()` function
   - Extract thinking from streamed output

3. **Route to ThoughtPanel**
   - Send thinking content to existing ThoughtPanel
   - Add new thought type "reasoning"

4. **Add Toggle Command**
   - `/reasoning on|off` command
   - `ctrl+r` keybind for quick toggle

**Estimated Effort:** 2-3 hours

### 5.2 Phase 2: Streaming Reasoning (Better UX)

1. **Streaming Parser**
   - Implement `ReasoningStreamParser` class
   - Handle tag transitions in token stream

2. **Dual-Stream UI**
   - Stream thinking to ThoughtPanel in real-time
   - Stream response to ChatView in real-time

3. **Progress Indication**
   - Show "Thinking..." status during reasoning phase
   - Transition indicator when moving to response

**Estimated Effort:** 4-6 hours

### 5.3 Phase 3: Advanced Features (Polish)

1. **Adaptive Reasoning**
   - Skip reasoning for simple queries
   - Deeper reasoning for complex tasks

2. **Reasoning Quality Metrics**
   - Track reasoning length, coherence
   - A/B test different prompts

3. **Reasoning Cache**
   - Cache and reuse reasoning for similar queries
   - Semantic similarity matching

4. **User Preferences**
   - Per-user reasoning preferences
   - Saved configurations

**Estimated Effort:** 8+ hours

---

## 6. Alternative Approaches Considered

### 6.1 Hidden Reasoning (o1-style)

**Approach:** Generate reasoning but never show it to user.

**Pros:**
- Simpler UI (no need for toggle)
- User sees faster perceived response

**Cons:**
- Loss of transparency/debugging
- Cannot leverage reasoning for user education
- Wastes compute if reasoning is poor

**Verdict:** Not recommended for Psyche, which values transparency.

### 6.2 Separate Reasoning Model

**Approach:** Use a different (smaller) model for reasoning, main model for response.

**Pros:**
- Can use specialized reasoning model
- Main model context not polluted with reasoning

**Cons:**
- Two model loads (memory/compute)
- Coordination complexity
- Reasoning context may not transfer well

**Verdict:** Not practical for local deployment with limited resources.

### 6.3 Post-hoc Reasoning Extraction

**Approach:** Generate response first, then extract/generate reasoning.

**Pros:**
- Faster time-to-first-token
- Reasoning doesn't affect response quality

**Cons:**
- Reasoning may not reflect actual process
- Defeats purpose of "thinking before responding"
- Could be misleading to users

**Verdict:** Not recommended - conflicts with stated goals.

---

## 7. References and Further Reading

### Academic Papers

1. **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models** (Wei et al., 2022)
   - Foundational paper on CoT prompting
   - Shows dramatic improvements on reasoning tasks

2. **Self-Consistency Improves Chain of Thought Reasoning** (Wang et al., 2023)
   - Multiple reasoning paths improve accuracy
   - Applicable to local models

3. **Tree of Thoughts: Deliberate Problem Solving with Large Language Models** (Yao et al., 2023)
   - Structured exploration of reasoning paths
   - More sophisticated than linear CoT

4. **Large Language Models are Zero-Shot Reasoners** (Kojima et al., 2022)
   - "Let's think step by step" effectiveness
   - Simple prompting unlocks reasoning

### Industry Documentation

1. **OpenAI o1 Guide**
   - https://platform.openai.com/docs/guides/reasoning
   - Architecture and best practices

2. **Anthropic Extended Thinking**
   - https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking
   - API integration details

3. **DeepSeek-R1 Technical Report**
   - https://github.com/deepseek-ai/DeepSeek-R1
   - Open reasoning model approach

### Coding Agent Implementations

1. **Claude Code** - https://claude.ai/code
2. **Cursor** - https://cursor.sh
3. **Aider** - https://aider.chat
4. **OpenCode** - https://opencode.ai

---

## 8. Conclusion

Implementing reasoning workflows in Psyche is both feasible and valuable. The recommended approach:

1. **Use explicit `<thinking>` tags** in the system prompt to elicit structured reasoning
2. **Leverage the existing ThoughtPanel** infrastructure for display
3. **Start simple** with Phase 1 implementation, iterate based on user feedback
4. **Consider performance** carefully - provide options to skip reasoning for simple queries

The key insight from modern reasoning models (o1, DeepSeek-R1, Claude extended thinking) is that **structured thinking improves output quality**, but the implementation can be as simple as prompt engineering for smaller local models.

For Psyche specifically, the transparent reasoning approach (visible `<thinking>` blocks) aligns with the project's values and provides a natural UI integration point through the existing thought panel system.

---

*Report completed by Reasoning Workflows Review Agent*
