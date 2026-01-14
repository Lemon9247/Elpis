# Bug Investigation Synthesis Report

**Date**: 2026-01-14

## Overview

Three agents investigated the Elpis project packages for critical bugs that could cause task group errors and runtime failures.

## Critical Bugs Summary (Most Likely to Cause Task Group Errors)

### 1. Fire-and-Forget Async Tasks (Elpis)
**File**: `src/elpis/server.py:419`
- Stream producer task created without storing reference
- Unhandled exceptions lost, no cancellation possible
- **This is a prime candidate for "unhandled exception in task group" errors**

### 2. Race Condition in Stream State (Elpis)
**File**: `src/elpis/server.py:456-479`
- Stream deleted while background producer still running
- Can cause data loss and concurrent access issues

### 3. Blocking Sync Calls in Async Handlers (Mnemosyne)
**File**: `src/mnemosyne/server.py` (multiple lines)
- All ChromaDB operations block the event loop
- Can cause task group timeouts under load

### 4. Bare `except:` Swallowing All Exceptions (Mnemosyne)
**File**: `src/mnemosyne/storage/chroma_store.py:139-140, 147-148`
- Silently swallows ALL exceptions including SystemExit
- Makes debugging impossible

### 5. Server Task Exception Silently Ignored (Psyche)
**File**: `src/psyche/client/app.py:87-93`
- Background server task dies silently
- App continues with dead server

## Issue Count by Package

| Package | Critical | High | Medium | Low |
|---------|----------|------|--------|-----|
| Elpis | 2 | 4 | 5 | 3 |
| Mnemosyne | 2 | 3 | 2 | 3 |
| Psyche | 1 | 4 | 4 | 6 |
| **Total** | **5** | **11** | **11** | **12** |

## Immediate Fix Recommendations

1. **Fix fire-and-forget tasks** - Store task references, implement proper cancellation
2. **Fix bare except clauses** - Replace with `except Exception as e:` and log
3. **Wrap sync DB operations** - Use `asyncio.to_thread()` for ChromaDB calls
4. **Add connection checks** - Verify `is_connected` before MCP operations
5. **Add server health monitoring** - Watch for task completion in TUI app

## Reports Location

- `scratchpad/bug-investigation/elpis-agent-report.md`
- `scratchpad/bug-investigation/mnemosyne-agent-report.md`
- `scratchpad/bug-investigation/psyche-agent-report.md`
