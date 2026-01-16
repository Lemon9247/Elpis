# Phase 3: Memory & Reasoning Implementation Plan

**Branch**: `phase3/memory-reasoning`
**Sessions**: 4 (Sessions 10-13 of master workplan)
**Depends on**: Phase 2 complete (merged)

## Summary

| Session | Track | Focus | Key Changes |
|---------|-------|-------|-------------|
| 10 | B2.3 | Structured Summarization | Verification testing + format improvements |
| 11 | B2.4 | Importance Scoring | Heuristic scoring + auto-storage |
| 12 | D1 | Reasoning Workflow | `<thinking>` tags + ThoughtPanel routing |
| 13 | - | Integration Testing | End-to-end memory + reasoning verification |

---

## Current State Analysis

### What Already Exists

| Component | File | Status |
|-----------|------|--------|
| `_summarize_conversation()` | `server.py:1303-1356` | Implemented, needs testing |
| Auto memory retrieval | `server.py:541-590` | Working |
| Periodic checkpoints | `server.py:510-539` | Working |
| Memory consolidation | `server.py:1240-1301` | Working |
| ThoughtPanel widget | `thought_panel.py` | Has 4 types: reflection, planning, idle, memory |
| Command registry | `commands.py` | 6 commands: help, quit, clear, status, thoughts, emotion |
| Staged message buffer | `server.py:1551` | Fixed (populated during compaction) |

### What Needs Implementation

| Component | Status |
|-----------|--------|
| Heuristic importance scoring | **Not implemented** |
| Auto-storage of important exchanges | **Not implemented** |
| `<thinking>` tag system prompt | **Not implemented** |
| Reasoning parser | **Not implemented** |
| "reasoning" thought type | **Not implemented** |
| `/thinking` command | **Not implemented** |

---

## Session 10: B2.3 - Structured Summarization Verification

### Objective
Verify the existing summarization system works correctly and enhance the summary format.

### Tasks

#### 10.1 Verification Testing
Test the existing `_summarize_conversation()` method end-to-end:

```python
# Test scenarios:
1. Create multi-turn conversation with code, explanations, and tool use
2. Trigger compaction (exceed token limit)
3. Verify summary stored to Mnemosyne with correct metadata
4. Verify emotional context captured
5. Test fallback when Mnemosyne unavailable
```

**Test file**: `tests/psyche/integration/test_summarization.py`

#### 10.2 Enhanced Summary Format (Optional)
If verification reveals issues, improve the summary prompt:

**File**: `src/psyche/memory/server.py` (line ~1310)

Current prompt requests basic summary. Enhanced version should request structured format:

```python
SUMMARY_PROMPT = """Summarize this conversation, extracting:
1. **Topics discussed**: Key subjects covered
2. **Decisions made**: Any choices or conclusions reached
3. **Facts learned**: New information discovered
4. **Code written**: Key functions/files created or modified
5. **Errors encountered**: Problems and their resolutions

Be concise but comprehensive. Format as bullet points."""
```

### Files Modified
- `src/psyche/memory/server.py` - Enhanced summary prompt (if needed)
- `tests/psyche/integration/test_summarization.py` - New test file

### Testing Checklist
- [ ] Conversation summary stored to Mnemosyne on compaction
- [ ] Summary includes key topics and decisions
- [ ] Emotional context attached to stored memory
- [ ] Fallback works when Mnemosyne unavailable
- [ ] Retrieved memories include summaries

---

## Session 11: B2.4 - Heuristic Importance Scoring

### Objective
Implement automatic importance calculation for exchanges and auto-store high-importance ones.

### Problem
Currently only stores messages dropped during compaction. Important exchanges within the working memory window are lost.

### Solution

#### 11.1 Create Importance Scoring Module

**New file**: `src/psyche/memory/importance.py`

```python
"""Heuristic importance scoring for automatic memory storage."""

from dataclasses import dataclass
from typing import Any


@dataclass
class ImportanceScore:
    """Breakdown of importance score factors."""
    total: float
    length_score: float
    code_score: float
    tool_score: float
    error_score: float
    explicit_score: float
    emotion_score: float


def calculate_importance(
    message: str,
    response: str,
    tool_results: list[dict[str, Any]] | None = None,
    emotion: dict[str, float] | None = None,
) -> ImportanceScore:
    """
    Calculate importance score (0.0 to 1.0) for an exchange.

    Factors:
    - Response length (longer = more effort)
    - Contains code blocks (likely a solution)
    - Tool execution occurred (concrete actions)
    - Error messages present (learn from mistakes)
    - User said "remember this" (explicit request)
    - Emotional intensity (significant moments)
    """
    scores = {
        "length_score": 0.0,
        "code_score": 0.0,
        "tool_score": 0.0,
        "error_score": 0.0,
        "explicit_score": 0.0,
        "emotion_score": 0.0,
    }

    # Length-based scoring
    if len(response) > 500:
        scores["length_score"] = 0.3
    elif len(response) > 200:
        scores["length_score"] = 0.15

    # Code blocks (likely a solution)
    if "```" in response:
        scores["code_score"] = 0.25

    # Tool execution (concrete actions)
    if tool_results:
        scores["tool_score"] = 0.2
        # Failures are more important (learn from mistakes)
        if any("error" in str(r).lower() for r in tool_results):
            scores["error_score"] = 0.15

    # Explicit user request
    explicit_phrases = ["remember", "important", "note that", "don't forget", "keep in mind"]
    if any(phrase in message.lower() for phrase in explicit_phrases):
        scores["explicit_score"] = 0.3

    # Emotional intensity from Elpis
    if emotion:
        valence = abs(emotion.get("valence", 0))
        arousal = abs(emotion.get("arousal", 0))
        if valence > 0.5 or arousal > 0.5:
            scores["emotion_score"] = 0.15

    total = min(1.0, sum(scores.values()))

    return ImportanceScore(total=total, **scores)
```

#### 11.2 Add Configuration

**File**: `src/psyche/memory/config.py`

Add to `ServerConfig`:

```python
# Auto-storage settings
auto_storage: bool = True
auto_storage_threshold: float = 0.6  # Min importance to auto-store
```

#### 11.3 Integrate Into Response Flow

**File**: `src/psyche/memory/server.py`

Add after response generation (around line 730, after `_maybe_checkpoint()`):

```python
async def _after_response(
    self,
    user_message: str,
    response: str,
    tool_results: list[dict] | None = None,
) -> None:
    """Called after generating response. Handles auto-storage of important exchanges."""
    if not self.config.auto_storage:
        return

    if not self.mnemosyne_client or not self.mnemosyne_client.is_connected:
        return

    # Get current emotional state
    emotion = None
    if self.client:
        try:
            emotion_state = await self.client.get_emotion()
            emotion = emotion_state.to_dict() if emotion_state else None
        except Exception:
            pass

    # Calculate importance
    from psyche.memory.importance import calculate_importance
    score = calculate_importance(user_message, response, tool_results, emotion)

    # Auto-store if above threshold
    if score.total >= self.config.auto_storage_threshold:
        try:
            # Create memory content with context
            memory_content = f"User: {user_message[:200]}...\n\nResponse: {response[:500]}..."

            await self.mnemosyne_client.store_memory(
                content=memory_content,
                summary=response[:100],
                memory_type="episodic",
                emotional_context=emotion,
                importance=score.total,
                tags=["auto-stored", "important"],
            )
            logger.debug(f"Auto-stored exchange (importance={score.total:.2f})")
        except Exception as e:
            logger.warning(f"Failed to auto-store exchange: {e}")
```

Call this from the response handling code (after tool execution completes in ReAct loop).

### Files Modified
- `src/psyche/memory/importance.py` - **New file**: Importance scoring
- `src/psyche/memory/config.py` - Add auto_storage settings
- `src/psyche/memory/server.py` - Integrate `_after_response()` call

### Testing Checklist
- [ ] High-importance exchanges auto-stored (code blocks, tool use)
- [ ] Low-importance exchanges not stored (simple questions)
- [ ] Explicit "remember this" triggers storage
- [ ] Emotional intensity affects scoring
- [ ] Threshold is configurable
- [ ] Graceful handling when Mnemosyne unavailable

---

## Session 12: D1 - Reasoning Workflow

### Objective
Add explicit reasoning/thinking step that displays in ThoughtPanel.

### Tasks

#### 12.1 Update System Prompt

**File**: `src/psyche/memory/server.py` (line ~268-308)

Add reasoning instruction to system prompt:

```python
REASONING_PROMPT = """
When responding to complex questions or tasks, first think through your approach
inside <thinking> tags. Consider:
- What is being asked?
- What information do I need?
- What tools might help?
- What's my approach?

After thinking, provide your response outside the tags.

Example:
<thinking>
The user wants to fix a bug in the login function. I should:
1. First read the current implementation
2. Identify the issue
3. Propose a fix
</thinking>

I'll help you fix that bug. Let me start by reading the login function...
"""
```

Add this to the system prompt builder when reasoning mode is enabled.

#### 12.2 Implement Reasoning Parser

**New file**: `src/psyche/memory/reasoning.py`

```python
"""Parser for extracting reasoning from model responses."""

import re
from dataclasses import dataclass


@dataclass
class ParsedResponse:
    """Response with extracted reasoning."""
    thinking: str  # Content from <thinking> tags
    response: str  # Content outside tags
    has_thinking: bool


THINKING_PATTERN = re.compile(
    r"<thinking>(.*?)</thinking>",
    re.DOTALL | re.IGNORECASE
)


def parse_reasoning(text: str) -> ParsedResponse:
    """
    Extract reasoning from model response.

    Args:
        text: Raw model response that may contain <thinking> tags

    Returns:
        ParsedResponse with separated thinking and response
    """
    match = THINKING_PATTERN.search(text)

    if match:
        thinking = match.group(1).strip()
        # Remove all thinking blocks from response
        response = THINKING_PATTERN.sub("", text).strip()
        return ParsedResponse(
            thinking=thinking,
            response=response,
            has_thinking=True,
        )

    return ParsedResponse(
        thinking="",
        response=text,
        has_thinking=False,
    )
```

#### 12.3 Add "reasoning" Thought Type

**File**: `src/psyche/client/widgets/thought_panel.py`

Add to `TYPE_COLORS` dict (around line 30):

```python
TYPE_COLORS = {
    "reflection": "cyan",
    "planning": "yellow",
    "idle": "dim",
    "memory": "magenta",
    "reasoning": "green",  # NEW
}
```

#### 12.4 Route Reasoning to ThoughtPanel

**File**: `src/psyche/memory/server.py`

In the streaming response handling (around line 650-700), after collecting full response:

```python
# After response complete, check for reasoning
from psyche.memory.reasoning import parse_reasoning

parsed = parse_reasoning(full_response)

if parsed.has_thinking and self.on_thought:
    # Send reasoning to thought panel
    self.on_thought(ThoughtEvent(
        thought=parsed.thinking,
        thought_type="reasoning",
        emotion=current_emotion,
    ))

# Continue with parsed.response (without thinking tags)
final_response = parsed.response
```

#### 12.5 Add `/thinking` Command

**File**: `src/psyche/client/commands.py`

Add to `COMMANDS` dict:

```python
"thinking": Command(
    name="thinking",
    aliases=["r", "reason"],
    description="Toggle reasoning display (on/off)",
    shortcut="Ctrl+R",
),
```

**File**: `src/psyche/client/app.py`

Add to BINDINGS:

```python
Binding("ctrl+r", "toggle_reasoning", "Reasoning", show=False),
```

Add handler in `_handle_command()`:

```python
elif cmd.name == "thinking":
    # Toggle or set reasoning mode
    if args:
        self._reasoning_enabled = args.lower() in ("on", "true", "1")
    else:
        self._reasoning_enabled = not self._reasoning_enabled

    status = "enabled" if self._reasoning_enabled else "disabled"
    chat.add_system_message(f"[dim]Reasoning display {status}[/]")

    # Notify server of reasoning mode change
    self.memory_server.set_reasoning_mode(self._reasoning_enabled)
```

Add action method:

```python
def action_toggle_reasoning(self) -> None:
    """Toggle reasoning display mode."""
    self._handle_command("/thinking")
```

#### 12.6 Server-Side Reasoning Mode

**File**: `src/psyche/memory/server.py`

Add instance variable and method:

```python
def __init__(self, ...):
    ...
    self._reasoning_enabled: bool = False

def set_reasoning_mode(self, enabled: bool) -> None:
    """Enable or disable reasoning mode."""
    self._reasoning_enabled = enabled
    logger.debug(f"Reasoning mode: {'enabled' if enabled else 'disabled'}")
```

Modify system prompt builder to include reasoning prompt when enabled.

### Files Modified
- `src/psyche/memory/server.py` - Reasoning mode, system prompt, response parsing
- `src/psyche/memory/reasoning.py` - **New file**: Reasoning parser
- `src/psyche/client/widgets/thought_panel.py` - Add "reasoning" type color
- `src/psyche/client/commands.py` - Add `/thinking` command
- `src/psyche/client/app.py` - Add keybinding and handler

### Testing Checklist
- [ ] `/thinking on` enables reasoning in system prompt
- [ ] Model outputs `<thinking>` tags when enabled
- [ ] Reasoning extracted and shown in ThoughtPanel
- [ ] Response shown without thinking tags
- [ ] `/thinking off` disables reasoning prompt
- [ ] `Ctrl+R` toggles reasoning mode
- [ ] Reasoning type shows in green color

---

## Session 13: Integration Testing

### Objective
Verify all Phase 3 components work together end-to-end.

### Test Scenarios

#### 13.1 Memory System End-to-End
```
1. Start psyche with fresh Mnemosyne
2. Have multi-turn conversation with code and tool use
3. Verify auto-storage triggers for important exchanges
4. Let context fill up, trigger compaction
5. Verify summaries stored correctly
6. Start new session, verify memories retrieved
7. Check emotional context preserved across sessions
```

#### 13.2 Reasoning Workflow
```
1. Enable reasoning mode (/thinking on)
2. Ask complex question requiring planning
3. Verify thinking appears in ThoughtPanel
4. Verify response doesn't include <thinking> tags
5. Toggle off, verify no thinking output
6. Test Ctrl+R keybinding
```

#### 13.3 Combined Features
```
1. Enable reasoning, have conversation
2. Ask to "remember this" - verify explicit storage
3. Reasoning thoughts should NOT be stored (only actual memories)
4. Continue until compaction
5. Verify structured summaries include reasoning outcomes
6. New session: memories + reasoning work together
```

#### 13.4 Edge Cases
```
- Mnemosyne disconnected: fallback storage works
- Rapid messages: no race conditions
- Long reasoning blocks: properly truncated
- Interrupt during reasoning: handled gracefully
- Empty reasoning tags: handled
```

### Test Files
- `tests/psyche/integration/test_phase3_memory.py`
- `tests/psyche/integration/test_phase3_reasoning.py`
- `tests/psyche/integration/test_phase3_integration.py`

### Success Criteria
From master workplan:
- [ ] Relevant memories auto-retrieved on each message
- [ ] Checkpoints saved every 20 messages
- [ ] Heuristic importance scoring working (auto-stores important exchanges)
- [ ] Reasoning displayed in ThoughtPanel (toggleable)
- [ ] Dream state streams tokens (already done in Phase 2)

---

## Critical Files

| File | Sessions | Changes |
|------|----------|---------|
| `src/psyche/memory/server.py` | 10,11,12 | Summarization, auto-storage, reasoning mode |
| `src/psyche/memory/importance.py` | 11 | **New**: Importance scoring module |
| `src/psyche/memory/reasoning.py` | 12 | **New**: Reasoning parser |
| `src/psyche/memory/config.py` | 11 | Auto-storage settings |
| `src/psyche/client/widgets/thought_panel.py` | 12 | Add "reasoning" type |
| `src/psyche/client/commands.py` | 12 | Add `/thinking` command |
| `src/psyche/client/app.py` | 12 | Reasoning toggle handler |

---

## Dependencies

```
Session 10 (B2.3) ────────────────────────────────────────┐
                                                          │
Session 11 (B2.4) ─────────────────────────────┬──────────┤
                                               │          │
Session 12 (D1) ───────────────────────────────┴──────────┤
                                                          │
                                Session 13 (Integration) ─┘
```

- Session 10 can run standalone (verification)
- Session 11 builds on memory infrastructure (no direct dependencies)
- Session 12 is independent (reasoning workflow)
- Session 13 requires all previous sessions complete

---

## Phase 3 Gate Criteria

From master workplan - Phase 3 complete when:
- [ ] Memories structured (verified in Session 10)
- [ ] Importance scoring working (Session 11)
- [ ] Auto-stores important exchanges (Session 11)
- [ ] Thinking visible in ThoughtPanel (Session 12)
- [ ] Toggleable via `/thinking` or `Ctrl+R` (Session 12)
- [ ] All integration tests pass (Session 13)

---

## Notes

1. **Summarization already works**: Session 10 is primarily verification, not implementation
2. **Importance scoring is NEW**: Session 11 is full implementation
3. **Reasoning is NEW**: Session 12 is full implementation
4. **No changes to Mnemosyne**: All changes are in Psyche
5. **Backward compatible**: Reasoning mode off by default, auto-storage configurable

---

## Post-Implementation

After Phase 3 completion:
1. Merge `phase3/memory-reasoning` to main
2. Update master workplan with completion status
3. Write Phase 3 completion report to `scratchpad/reports/phase3/`
4. Begin Phase 4: Architecture Refactor
