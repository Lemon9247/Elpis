# Session Report: Memory Storage Flow Fixes + LLM Summarization

**Date**: 2026-01-14

## Summary

Fixed critical bugs in Psyche's memory storage flow and added LLM-based conversation summarization on shutdown. Previously, staged messages could be lost silently if Mnemosyne storage failed, and conversations were stored with simple truncation (`content[:100]`) instead of real summaries.

## Changes Made

### 1. Bug Fixes in `src/psyche/memory/server.py`

#### `_store_messages_to_mnemosyne` (lines 985-1034)
- Added connection check: `if not self.mnemosyne_client.is_connected`
- Changed return type from `None` to `bool` to indicate success/failure
- Added success counting to track partial failures
- Logs warning instead of silently failing when not connected

#### `_handle_compaction_result` (lines 1036-1062)
- Now checks return value from `_store_messages_to_mnemosyne`
- Only clears staged messages if storage succeeded
- Failed messages are retained for retry on next compaction cycle
- New messages are appended to any failed messages (accumulation instead of replacement)

#### `shutdown_with_consolidation` (lines 1064-1119)
- Added connection check before attempting shutdown consolidation
- Only clears staged messages if storage succeeded
- Now generates and stores conversation summary before running consolidation

### 2. New Summarization Features

#### `_summarize_conversation` (lines 985-1038)
New method that uses Elpis to generate conversation summaries:
- Builds conversation text from messages (truncates to 500 chars each)
- Uses a summarization prompt focusing on key facts, decisions, and important details
- Returns empty string on failure (graceful degradation)

#### `_store_conversation_summary` (lines 1040-1078)
New method that stores the summary as semantic memory:
- Generates summary using `_summarize_conversation`
- Stores as `memory_type="semantic"` (distilled knowledge, not raw episodic)
- Includes emotional context from current state
- Tags with `["conversation_summary", "shutdown"]`

### 3. Storage Flow Changes

**Before:**
```
shutdown -> store messages -> clear staged -> store remaining -> consolidate
                             (regardless of success)
```

**After:**
```
shutdown -> check connection -> store staged (only clear on success)
         -> store remaining -> generate summary -> store as semantic memory
         -> consolidate
```

## Files Modified

- `src/psyche/memory/server.py`: Main changes (+96 lines, ~4 new methods/modifications)

## Testing

- All 205 Psyche tests pass
- All 36 memory-related tests pass
- No regressions detected

## Related Plan

Plan saved at: `scratchpad/plans/20260114-memory-summarization-plan.md`

## Next Steps (optional future work)

1. Add tests specifically for the new summarization methods
2. Consider adding summarization on compaction (not just shutdown) for very long sessions
3. Monitor memory usage - accumulated failed messages could grow if Mnemosyne is down for extended periods
