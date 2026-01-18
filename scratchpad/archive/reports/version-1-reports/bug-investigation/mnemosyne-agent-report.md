# Mnemosyne Agent Bug Investigation Report

**Date**: 2026-01-14
**Agent**: Mnemosyne Agent
**Scope**: `src/mnemosyne/` - Memory MCP Server

---

## Executive Summary

I have identified **6 critical/high severity issues** and **4 medium/low severity issues** in the Mnemosyne memory MCP server. The most concerning bugs involve:

1. Missing `await` on synchronous ChromaDB calls inside async handlers
2. Bare `except:` clauses swallowing exceptions silently
3. Potential KeyError crashes when accessing optional dictionary keys
4. Resource cleanup issues (no `close()` method for ChromaDB client)

---

## Critical Issues

### Issue 1: Missing `await` on Synchronous ChromaDB Operations

**File**: `/home/lemoneater/Projects/Elpis/src/mnemosyne/server.py`
**Lines**: 221, 235, 259-261, 275, 282-283, 299, 336, 352
**Severity**: Critical

**Description**: The server's async handlers (`_handle_store_memory`, `_handle_search_memories`, etc.) call synchronous ChromaDB methods without proper async handling. While this doesn't cause immediate crashes because the methods are synchronous, it **blocks the event loop** during database operations.

In an MCP server context, this can cause:
- Task group timeouts when multiple requests arrive
- Unresponsive behavior during long-running ChromaDB operations
- Potential "unhandled exception in task group" errors under load

**Affected Code Examples**:
```python
# Line 221 - server.py
memory_store.add_memory(memory)  # Blocks event loop

# Line 235 - server.py
memories = memory_store.search_memories(query, n_results)  # Blocks event loop

# Lines 259-261 - server.py
memory_store.count_memories()  # Blocks event loop (called 3 times)
```

**Suggested Fix**: Either:
1. Run ChromaDB operations in a thread pool using `asyncio.to_thread()`
2. Or ensure ChromaDB operations are properly non-blocking

---

### Issue 2: Bare `except:` Clauses Swallowing Exceptions

**File**: `/home/lemoneater/Projects/Elpis/src/mnemosyne/storage/chroma_store.py`
**Lines**: 139-140, 147-148
**Severity**: Critical

**Description**: The `get_memory()` method uses bare `except:` clauses that silently swallow ALL exceptions, including `KeyboardInterrupt`, `SystemExit`, and potentially useful debugging information.

**Affected Code**:
```python
# Lines 135-148 - chroma_store.py
def get_memory(self, memory_id: str) -> Optional[Memory]:
    # Try short-term first
    try:
        result = self.short_term.get(ids=[memory_id])
        if result["ids"]:
            return self._result_to_memory(result, 0)
    except:  # <-- Bare except! Swallows ALL exceptions
        pass

    # Try long-term
    try:
        result = self.long_term.get(ids=[memory_id])
        if result["ids"]:
            return self._result_to_memory(result, 0)
    except:  # <-- Bare except! Swallows ALL exceptions
        pass
```

**Impact**:
- Database corruption or connection issues will be silently ignored
- Memory parsing errors will be hidden
- Makes debugging extremely difficult

**Suggested Fix**: Replace with `except Exception as e:` and log the error, or catch specific ChromaDB exceptions.

---

## High Severity Issues

### Issue 3: Potential KeyError in EmotionalContext Parsing

**File**: `/home/lemoneater/Projects/Elpis/src/mnemosyne/server.py`
**Lines**: 202-206
**Severity**: High

**Description**: When parsing `emotional_context` from tool arguments, the code accesses dictionary keys without checking if they exist. If a partial emotional context is provided (e.g., missing `arousal`), it will raise a `KeyError`.

**Affected Code**:
```python
# Lines 200-206 - server.py
if args.get("emotional_context"):
    ec = args["emotional_context"]
    emotional_ctx = EmotionalContext(
        valence=ec["valence"],    # KeyError if missing
        arousal=ec["arousal"],    # KeyError if missing
        quadrant=ec["quadrant"],  # KeyError if missing
    )
```

**Impact**: The exception is caught by the outer try/except in `call_tool()`, but:
- The error message won't be helpful ("'valence'")
- Partial data causes full tool failure instead of graceful handling

**Suggested Fix**: Use `ec.get("valence", 0.0)` with defaults, or validate required fields explicitly.

---

### Issue 4: No Resource Cleanup / Graceful Shutdown

**File**: `/home/lemoneater/Projects/Elpis/src/mnemosyne/server.py`
**Severity**: High

**Description**: The server has no cleanup mechanism for the ChromaDB client. When the server shuts down:
- ChromaDB's `PersistentClient` may not flush pending writes
- No explicit connection closing
- Potential for database corruption on hard shutdown

**Additional Context**: The `run_server()` function (lines 391-400) doesn't have any cleanup in a `finally` block.

**Suggested Fix**: Add a shutdown handler that calls appropriate cleanup methods on ChromaDB client.

---

### Issue 5: Synchronous `consolidate()` Called in Async Context

**File**: `/home/lemoneater/Projects/Elpis/src/mnemosyne/server.py`
**Lines**: 274-275
**Severity**: High

**Description**: The `_handle_consolidate_memories()` async handler calls `consolidator.consolidate()` which is a synchronous method that performs multiple database operations (get_all_short_term, promote_memory, delete_memory, etc.).

**Affected Code**:
```python
# Lines 273-277 - server.py
async def _handle_consolidate_memories(args: Dict[str, Any]) -> Dict[str, Any]:
    # ...
    consolidator = MemoryConsolidator(store=memory_store, config=config)
    report = consolidator.consolidate()  # Long-running sync operation!
    return report.to_dict()
```

**Impact**:
- Consolidation can take a significant amount of time
- Blocks the entire event loop during consolidation
- Other MCP requests will be unresponsive
- Could cause task group timeouts/errors

**Suggested Fix**: Run consolidation in a thread pool: `await asyncio.to_thread(consolidator.consolidate)`

---

## Medium Severity Issues

### Issue 6: ChromaDB Exception Types Not Specifically Caught

**File**: `/home/lemoneater/Projects/Elpis/src/mnemosyne/storage/chroma_store.py`
**Lines**: 298-300, 311-313, 345-346, 392-393, 415-416, 425-426
**Severity**: Medium

**Description**: All exception handling uses generic `except Exception as e:` instead of catching specific ChromaDB exceptions. This makes it impossible to distinguish between:
- Connection failures
- Query errors
- Invalid ID errors
- Rate limiting

**Suggested Fix**: Import and catch specific ChromaDB exceptions for better error handling.

---

### Issue 7: Missing Validation for `n_results` Parameter

**File**: `/home/lemoneater/Projects/Elpis/src/mnemosyne/storage/chroma_store.py`
**Lines**: 188-189
**Severity**: Medium

**Description**: The code already handles empty collections (line 184-186), but if `n_results` is 0 or negative, ChromaDB may behave unexpectedly.

**Affected Code**:
```python
# Lines 187-189
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=min(n_results, count),  # What if n_results <= 0?
```

**Suggested Fix**: Add validation: `n_results = max(1, n_results or 10)`

---

## Low Severity Issues

### Issue 8: Mutable Default Argument

**File**: `/home/lemoneater/Projects/Elpis/src/mnemosyne/core/consolidator.py`
**Line**: 51-52
**Severity**: Low

**Description**: The `config` parameter defaults to `None` which is fine, but if it were changed to a mutable default like `ConsolidationConfig()` in the future, it would cause a well-known Python gotcha.

Current code is OK but worth noting for maintenance.

---

### Issue 9: Rough Token Estimation May Be Inaccurate

**File**: `/home/lemoneater/Projects/Elpis/src/mnemosyne/server.py`
**Lines**: 309-310
**Severity**: Low

**Description**: Token estimation uses `len(text) // 4` which is a rough approximation. For non-ASCII content or special characters, this could significantly undercount tokens.

**Impact**: Could return more content than expected, potentially hitting context limits.

**Suggested Fix**: Consider using a proper tokenizer or being more conservative (e.g., `// 3`).

---

### Issue 10: `get_recent_memories` Only Searches Short-Term

**File**: `/home/lemoneater/Projects/Elpis/src/mnemosyne/server.py`
**Lines**: 344-364
**Severity**: Low

**Description**: Despite the comment on line 355 saying "Also search long-term for recent additions", the code only retrieves short-term memories. Recent memories that were promoted to long-term storage won't be returned.

**Affected Code**:
```python
# Line 352 - Only gets short-term
short_term_memories = memory_store.get_all_short_term(limit=limit * 2)

# Lines 355-356 - Comment says it will search long-term but doesn't
# Also search long-term for recent additions
# Note: We'll combine and filter both collections
all_memories = short_term_memories  # <-- Long-term not included!
```

**Suggested Fix**: Actually implement the long-term memory search as the comment suggests.

---

## Summary Table

| # | Severity | File | Issue |
|---|----------|------|-------|
| 1 | Critical | server.py | Blocking sync calls in async handlers |
| 2 | Critical | chroma_store.py | Bare `except:` swallowing exceptions |
| 3 | High | server.py | KeyError in EmotionalContext parsing |
| 4 | High | server.py | No resource cleanup on shutdown |
| 5 | High | server.py | Sync consolidate() blocks event loop |
| 6 | Medium | chroma_store.py | Generic exception handling |
| 7 | Medium | chroma_store.py | Missing n_results validation |
| 8 | Low | consolidator.py | Mutable default argument pattern |
| 9 | Low | server.py | Rough token estimation |
| 10 | Low | server.py | get_recent_memories ignores long-term |

---

## Recommendations

1. **Immediate**: Fix the bare `except:` clauses (Issue 2) - this is masking real problems
2. **High Priority**: Wrap all ChromaDB operations in `asyncio.to_thread()` (Issues 1, 5)
3. **Important**: Add proper KeyError handling for emotional context (Issue 3)
4. **Improve**: Add graceful shutdown handling (Issue 4)

---

*Report generated by Mnemosyne Agent*
