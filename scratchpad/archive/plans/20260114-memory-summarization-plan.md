# Plan: Fix Memory Storage Flow + Add LLM Summarization

## Problems

1. **Connection bugs**: No check if Mnemosyne is connected before storing
2. **Data loss**: Staged messages cleared even if storage fails
3. **No summarization**: Just `msg.content[:100]` truncation, not real summaries

## Solution Overview

### Part 1: Bug Fixes

#### Fix 1: Add connection check in `_store_messages_to_mnemosyne`
```python
if not self.mnemosyne_client or not self.mnemosyne_client.is_connected:
    logger.warning("Mnemosyne not connected, cannot store messages")
    return False  # Return success status
```

#### Fix 2: Don't clear staged messages on failure
```python
# In shutdown_with_consolidation:
if self._staged_messages:
    success = await self._store_messages_to_mnemosyne(self._staged_messages)
    if success:
        self._staged_messages = []
    else:
        logger.error(f"Failed to store {len(self._staged_messages)} staged messages")
```

#### Fix 3: Track which messages were stored successfully
Return a boolean from `_store_messages_to_mnemosyne` indicating success/failure.

### Part 2: LLM Summarization

On shutdown, generate a conversation summary using Elpis before storing.

#### Approach: Summarize conversation on shutdown

```python
async def _summarize_conversation(self, messages: List[Message]) -> str:
    """Use Elpis to summarize the conversation."""
    # Build a summary prompt
    conversation_text = "\n".join([
        f"{m.role}: {m.content[:500]}" for m in messages if m.role != "system"
    ])

    summary_prompt = [
        {"role": "system", "content": "Summarize this conversation. Extract key facts, decisions, and important details. Be concise."},
        {"role": "user", "content": conversation_text}
    ]

    result = await self.client.generate(
        messages=summary_prompt,
        max_tokens=500,
        temperature=0.3,
    )
    return result.content
```

#### Storage Flow Change

Instead of storing raw messages individually:
1. **On compaction**: Store individual messages (with truncated summaries - OK for episodic)
2. **On shutdown**: Generate conversation summary, store as semantic memory

```python
async def shutdown_with_consolidation(self) -> None:
    # ... existing code ...

    # NEW: Generate and store conversation summary
    all_messages = self._staged_messages + [
        m for m in self._compactor.messages if m.role != "system"
    ]
    if all_messages:
        summary = await self._summarize_conversation(all_messages)
        await self.mnemosyne_client.store_memory(
            content=summary,
            summary=summary[:100],
            memory_type="semantic",  # Semantic memory, not episodic
            tags=["conversation_summary", "shutdown"],
            emotional_context=...
        )
```

## Files to Modify

### 1. `src/psyche/memory/server.py`

**`_store_messages_to_mnemosyne` (lines 985-1016):**
- Add connection check
- Return bool success status
- Track individual failures

**`_handle_compaction_result` (lines 1018-1036):**
- Check return value from store function
- Don't clear staged if failed

**`shutdown_with_consolidation` (lines 1038-1083):**
- Add `_summarize_conversation` call
- Store conversation summary as semantic memory
- Only clear staged on success

**New method: `_summarize_conversation`:**
- Call Elpis to generate summary
- Handle errors gracefully

### 2. `src/psyche/mcp/client.py`

**`MnemosyneClient`:**
- Add `is_connected` property if missing (check if exists)

## Implementation Order

1. Add `is_connected` check to MnemosyneClient (if needed)
2. Fix `_store_messages_to_mnemosyne` - connection check + return status
3. Fix `_handle_compaction_result` - don't clear on failure
4. Fix `shutdown_with_consolidation` - don't clear on failure
5. Add `_summarize_conversation` method
6. Update `shutdown_with_consolidation` to generate + store summary
7. Test end-to-end

## Testing

1. Verify connection check prevents silent failures
2. Verify staged messages retained on failure
3. Verify shutdown generates proper summary
4. Verify summary stored as semantic memory in Mnemosyne
