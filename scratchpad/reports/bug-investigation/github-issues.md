# GitHub Issues for Non-Critical Bugs

Run `gh auth login` first, then use these commands to create the issues.

---

## Issue 1: Elpis Thread Management and Cleanup

```bash
gh issue create --title "Elpis: Thread join timeout and cleanup improvements" --body "## Description
Several high/medium severity issues related to thread management and cleanup in the Elpis inference backends.

## Issues

### High: Thread join timeout without alive check
**Files**: \`src/elpis/llm/backends/llama_cpp/inference.py:285\`, \`src/elpis/llm/backends/transformers/inference.py:310,348\`

The streaming implementations use \`thread.join(timeout=1.0)\` but don't check if the thread is still alive after timeout.

**Suggested fix**: Check \`thread.is_alive()\` after join and log warning.

### High: Steering hook not removed on exception
**File**: \`src/elpis/llm/backends/transformers/inference.py:279-310\`

If an exception occurs while iterating the streamer, the steering hook remains attached.

### High: Unreliable __del__ cleanup
**File**: \`src/elpis/llm/backends/transformers/inference.py:443-446\`

The \`__del__\` method is not guaranteed to run.

**Suggested fix**: Implement explicit async cleanup method.

### High: Type mismatch in hardware_backend
**Files**: \`src/elpis/config/settings.py:36-38\`, \`src/elpis/llm/backends/llama_cpp/config.py:66\`

Should use \`Literal\` type consistently.

### Medium: Silent failure when emotion vectors missing
**File**: \`src/elpis/llm/backends/transformers/steering.py:63-66\`

### Medium: torch.load compatibility
**File**: \`src/elpis/llm/backends/transformers/steering.py:71-72\`
"
```

---

## Issue 2: Mnemosyne Resource Management

```bash
gh issue create --title "Mnemosyne: Resource cleanup and error handling improvements" --body "## Description
Issues related to resource management and error handling in the Mnemosyne memory server.

## Issues

### High: No resource cleanup on shutdown
**File**: \`src/mnemosyne/server.py\`

The server has no cleanup mechanism for the ChromaDB client. No \`finally\` block in \`run_server()\`.

**Suggested fix**: Add shutdown handler for ChromaDB cleanup.

### Medium: Generic exception handling
**File**: \`src/mnemosyne/storage/chroma_store.py\` (multiple locations)

All exception handling uses generic \`except Exception\` instead of specific ChromaDB exceptions.

### Medium: Missing n_results validation
**File**: \`src/mnemosyne/storage/chroma_store.py:188-189\`

If \`n_results\` is 0 or negative, ChromaDB may behave unexpectedly.

### Low: get_recent_memories ignores long-term
**File**: \`src/mnemosyne/server.py:344-364\`

Despite comment saying it will search long-term, it only searches short-term.

### Low: Rough token estimation
**File**: \`src/mnemosyne/server.py:309-310\`

Token estimation uses \`len(text) // 4\` which is inaccurate for non-ASCII.
"
```

---

## Issue 3: Psyche Connection Handling and Tools

```bash
gh issue create --title "Psyche: Connection guards and tool improvements" --body "## Description
Issues related to connection handling and tool implementations in Psyche.

## Issues

### High: Memory tools don't check connection
**File**: \`src/psyche/tools/implementations/memory_tools.py:43-76\`

\`recall_memory\` and \`store_memory\` don't verify \`is_connected\` before calling client.

### Medium: Potential race in staged messages
**File**: \`src/psyche/memory/server.py:1131-1157\`

Concurrent modifications to \`_staged_messages\` possible under certain conditions.

### Medium: Callbacks modify widgets directly
**File**: \`src/psyche/client/app.py:95-123\`

Token/thought callbacks directly modify Textual widgets. Should use \`post_message()\`.

### Low: File opened twice
**File**: \`src/psyche/tools/implementations/file_tools.py:109-117\`

File opened once to read, once to count lines. Should be single pass.

### Low: Type annotation error
**File**: \`src/psyche/tools/implementations/memory_tools.py:16\`

\`callable\` should be \`Callable\` from typing module.

### Low: Incomplete dangerous commands list
**File**: \`src/psyche/tools/implementations/bash_tool.py:20-32\`

Missing: \`sudo\`, \`chmod 777\`, \`chown\`, etc.

### Low: Hardcoded poll interval
**File**: \`src/psyche/mcp/client.py:175-235\`

Stream polling at 0.05s with no backoff.
"
```

---

## Quick Create All (after gh auth login)

```bash
# Issue 1
gh issue create --title "Elpis: Thread join timeout and cleanup improvements" --body-file /dev/stdin << 'ISSUE1'
High/medium severity issues in Elpis inference backends:
- Thread join timeout without alive check
- Steering hook not removed on exception
- Unreliable __del__ cleanup
- Type mismatch in hardware_backend
- Silent failure when vectors missing
See scratchpad/bug-investigation/elpis-agent-report.md for details.
ISSUE1

# Issue 2
gh issue create --title "Mnemosyne: Resource cleanup improvements" --body-file /dev/stdin << 'ISSUE2'
High/medium severity issues in Mnemosyne:
- No resource cleanup on shutdown
- Generic exception handling (should catch specific ChromaDB errors)
- Missing n_results validation
- get_recent_memories ignores long-term storage
See scratchpad/bug-investigation/mnemosyne-agent-report.md for details.
ISSUE2

# Issue 3
gh issue create --title "Psyche: Connection guards and tool improvements" --body-file /dev/stdin << 'ISSUE3'
High/medium severity issues in Psyche:
- Memory tools don't check connection state
- Potential race in staged messages
- Callbacks modify widgets directly (thread safety)
- File opened twice in read_file
- Incomplete dangerous commands blocklist
See scratchpad/bug-investigation/psyche-agent-report.md for details.
ISSUE3
```
