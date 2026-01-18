# Fix: Remote Mode Tool Architecture (Phase 6)

## Problem
In remote mode (`hermes --server`), Psyche generates tool_call blocks but nothing happens. Tools are never executed.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     PSYCHE SERVER                           │
│  - Memory tools (recall_memory, store_memory)               │
│  - Executes memory operations INTERNALLY                    │
│  - Returns OTHER tool_calls to client                       │
└─────────────────────────────────────────────────────────────┘
                            ↕ HTTP API
┌─────────────────────────────────────────────────────────────┐
│                     HERMES CLIENT                           │
│  - File tools (read_file, create_file, edit_file)           │
│  - Bash tool (execute_bash)                                 │
│  - Search tools (search_codebase, list_directory)           │
│  - Executes these LOCALLY when server returns tool_calls    │
└─────────────────────────────────────────────────────────────┘
```

## Current State (Verified)

| Component | File | Status | Line Numbers |
|-----------|------|--------|--------------|
| HTTP Server tool parsing | `http.py` | ✅ Works | `_parse_tool_calls()` at 434-466 |
| HTTP Server streaming | `http.py` | ✅ Works | `_stream_response()` at 368-432 |
| Client `generate_stream()` | `psyche_client.py` | ❌ Broken | Lines 515-561 - only extracts content |
| Hermes `_process_via_client()` | `app.py` | ❌ Broken | Lines 394-418 - never checks tool_calls |
| ToolEngine in remote mode | `cli.py` | ⚠️ Partial | Lines 377-386 - created but not passed |

---

## Implementation Plan

### Step 1: Client Tool Call Capture

**File:** `src/psyche/handlers/psyche_client.py`

#### 1.1 Add instance variables to `__init__()` (around line 389)

After the existing instance variables, add:
```python
# Tool call state from last response
self._last_tool_calls: Optional[List[Dict[str, Any]]] = None
self._last_finish_reason: Optional[str] = None
```

#### 1.2 Modify `generate_stream()` (lines 515-561)

**Current code** (lines 545-557):
```python
try:
    chunk = json.loads(data_str)
    delta = chunk.get("choices", [{}])[0].get("delta", {})
    token = delta.get("content", "")

    if token:
        full_content += token
        if on_token:
            on_token(token)
        yield token

except json.JSONDecodeError:
    continue
```

**New code:**
```python
try:
    chunk = json.loads(data_str)
    choice = chunk.get("choices", [{}])[0]
    delta = choice.get("delta", {})

    # Check for content token
    token = delta.get("content", "")
    if token:
        full_content += token
        if on_token:
            on_token(token)
        yield token

    # Check for finish reason and tool_calls
    finish_reason = choice.get("finish_reason")
    if finish_reason:
        self._last_finish_reason = finish_reason
        # Tool calls come in the delta on the finish chunk
        if "tool_calls" in delta:
            self._last_tool_calls = delta["tool_calls"]

except json.JSONDecodeError:
    continue
```

Also add at the START of the method (after the docstring):
```python
# Reset tool call state
self._last_tool_calls = None
self._last_finish_reason = None
```

#### 1.3 Add new method after `generate_stream()` (around line 562)

```python
def get_pending_tool_calls(self) -> Optional[List[Dict[str, Any]]]:
    """
    Get tool_calls from the last streamed response, if any.

    Returns:
        List of tool call dictionaries if finish_reason was "tool_calls",
        None otherwise.
    """
    if self._last_finish_reason == "tool_calls" and self._last_tool_calls:
        return self._last_tool_calls
    return None
```

---

### Step 2: Hermes App Tool Engine Support

**File:** `src/hermes/app.py`

#### 2.1 Add import at top of file (around line 20)

```python
from psyche.tools.tool_engine import ToolEngine
```

#### 2.2 Modify `__init__()` signature (lines 68-77)

**Current:**
```python
def __init__(
    self,
    client: Optional[PsycheClient] = None,
    react_handler: Optional[ReactHandler] = None,
    idle_handler: Optional[IdleHandler] = None,
    elpis_client: Optional["ElpisClient"] = None,
    mnemosyne_client: Optional["MnemosyneClient"] = None,
    *args,
    **kwargs,
):
```

**New:**
```python
def __init__(
    self,
    client: Optional[PsycheClient] = None,
    react_handler: Optional[ReactHandler] = None,
    idle_handler: Optional[IdleHandler] = None,
    elpis_client: Optional["ElpisClient"] = None,
    mnemosyne_client: Optional["MnemosyneClient"] = None,
    tool_engine: Optional[ToolEngine] = None,
    *args,
    **kwargs,
):
```

#### 2.3 Store tool_engine (around line 94)

Add after `self._mnemosyne_client = mnemosyne_client`:
```python
self._tool_engine = tool_engine
```

---

### Step 3: Hermes Tool Execution Loop

**File:** `src/hermes/app.py`

#### 3.1 Rewrite `_process_via_client()` (lines 394-418)

Replace the entire method with:

```python
async def _process_via_client(self, text: str) -> None:
    """Process input via remote client with tool execution loop."""
    import json

    MAX_ITERATIONS = 10
    chat = self.query_one("#chat", ChatView)

    # Add user message to client history
    await self._client.add_user_message(text)

    for iteration in range(MAX_ITERATIONS):
        # Stream response and display
        chat.start_stream()
        full_response = ""

        try:
            async for token in self._client.generate_stream():
                full_response += token
                chat.append_token(token)
        except Exception as e:
            chat.end_stream()
            raise

        chat.end_stream()

        # Check for tool calls
        tool_calls = self._client.get_pending_tool_calls()

        if not tool_calls or not self._tool_engine:
            # No tools to execute - add response to history and we're done
            if full_response:
                await self._client.add_assistant_message(full_response, user_message=text)
            break

        # Add assistant message with tool calls to history
        # (The server expects the full conversation history including the tool-calling message)
        if full_response:
            await self._client.add_assistant_message(full_response, user_message=text)

        # Execute each tool locally
        for tc in tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "unknown")
            args_str = func.get("arguments", "{}")

            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                args = {}

            # UI: show tool start
            self._on_tool_call(name, args, None)

            # Execute tool
            try:
                result = await self._tool_engine.execute_tool_call(tc)
            except Exception as e:
                result = {"success": False, "error": str(e)}

            # UI: show tool complete
            self._on_tool_call(name, args, result)

            # Send result back to server
            self._client.add_tool_result(name, json.dumps(result))

        # Loop continues - server will generate next response with tool results
```

---

### Step 4: CLI Integration

**File:** `src/hermes/cli.py`

#### 4.1 Update remote mode app creation (lines 405-411)

**Current:**
```python
app = Hermes(
    client=client,
    react_handler=None,  # Will use client directly
    idle_handler=None,  # No idle in remote mode
    elpis_client=None,  # Not available in remote mode
    mnemosyne_client=None,
)
```

**New:**
```python
app = Hermes(
    client=client,
    react_handler=None,  # Will use client directly
    idle_handler=None,  # No idle in remote mode
    elpis_client=None,  # Not available in remote mode
    mnemosyne_client=None,
    tool_engine=tool_engine,  # For local tool execution
)
```

---

### Step 5: Server Memory Tool Execution (Required)

**File:** `src/psyche/server/http.py`

Memory tools must execute server-side because Psyche's memory is part of her "self" - the server has the Mnemosyne connection.

#### 5.1 Add constant after imports (around line 30)

```python
# Memory tools that should be executed server-side
MEMORY_TOOLS = {"recall_memory", "store_memory"}
```

#### 5.2 Add helper method (after `_strip_tool_calls`, around line 475)

```python
async def _execute_memory_tool(self, tool_call: Dict[str, Any]) -> str:
    """Execute a memory tool internally and return the result as JSON."""
    func = tool_call.get("function", {})
    name = func.get("name")
    args_str = func.get("arguments", "{}")

    try:
        args = json.loads(args_str)
    except json.JSONDecodeError:
        return json.dumps({"success": False, "error": "Invalid arguments"})

    if name == "recall_memory":
        query = args.get("query", "")
        n_results = args.get("n_results", 5)
        memories = await self.core.retrieve_memories(query, n_results)
        return json.dumps({"memories": memories})

    elif name == "store_memory":
        content = args.get("content", "")
        summary = args.get("summary")
        memory_type = args.get("memory_type", "episodic")
        tags = args.get("tags", [])

        success = await self.core.store_memory(
            content=content,
            summary=summary,
            memory_type=memory_type,
            tags=tags,
        )
        return json.dumps({"success": success})

    return json.dumps({"success": False, "error": f"Unknown memory tool: {name}"})

def _separate_tool_calls(
    self, tool_calls: List[Dict[str, Any]]
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Separate memory tools from client tools."""
    memory_calls = []
    client_calls = []
    for tc in tool_calls:
        name = tc.get("function", {}).get("name", "")
        if name in MEMORY_TOOLS:
            memory_calls.append(tc)
        else:
            client_calls.append(tc)
    return memory_calls, client_calls
```

#### 5.3 Rewrite `_generate_response()` (lines 316-366)

Replace the entire method:

```python
async def _generate_response(
    self,
    request: ChatCompletionRequest,
    connection_id: str,
) -> ChatCompletionResponse:
    """Generate non-streaming response with internal memory tool execution."""
    MAX_MEMORY_ITERATIONS = 5
    accumulated_content = ""

    try:
        for iteration in range(MAX_MEMORY_ITERATIONS):
            result = await self.core.generate(
                max_tokens=request.max_tokens or 2048,
                temperature=request.temperature,
            )

            content = result["content"]

            # Parse for tool calls if tools were provided
            tool_calls = None
            if request.tools:
                tool_calls = self._parse_tool_calls(content)
                if tool_calls:
                    content = self._strip_tool_calls(content)

            accumulated_content += content

            if not tool_calls:
                # No tools - return final response
                message = ChatMessage(
                    role="assistant",
                    content=accumulated_content if accumulated_content.strip() else None,
                )
                return ChatCompletionResponse(
                    id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
                    created=int(time.time()),
                    model=self.config.model_name,
                    choices=[
                        ChatCompletionChoice(
                            index=0, message=message, finish_reason="stop"
                        )
                    ],
                    usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                )

            # Separate memory vs client tools
            memory_calls, client_calls = self._separate_tool_calls(tool_calls)

            if memory_calls:
                # Execute memory tools internally
                for tc in memory_calls:
                    result_str = await self._execute_memory_tool(tc)
                    self.core.add_tool_result(tc["function"]["name"], result_str)
                # If there are also client tools, return them
                # Otherwise continue loop to generate follow-up

            if client_calls:
                # Return client tools for execution
                message = ChatMessage(
                    role="assistant",
                    content=accumulated_content if accumulated_content.strip() else None,
                    tool_calls=client_calls,
                )
                return ChatCompletionResponse(
                    id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
                    created=int(time.time()),
                    model=self.config.model_name,
                    choices=[
                        ChatCompletionChoice(
                            index=0, message=message, finish_reason="tool_calls"
                        )
                    ],
                    usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                )

            # Only memory tools - continue loop

        # Max iterations reached
        message = ChatMessage(role="assistant", content=accumulated_content)
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=self.config.model_name,
            choices=[
                ChatCompletionChoice(index=0, message=message, finish_reason="stop")
            ],
            usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )

    finally:
        self._on_disconnect(connection_id)
```

#### 5.4 Rewrite `_stream_response()` (lines 368-432)

This is more complex because we stream tokens, then may need to loop for memory tools:

```python
async def _stream_response(
    self,
    request: ChatCompletionRequest,
    connection_id: str,
) -> AsyncIterator[str]:
    """Generate streaming response with internal memory tool execution."""
    MAX_MEMORY_ITERATIONS = 5
    response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    try:
        for iteration in range(MAX_MEMORY_ITERATIONS):
            full_content = ""

            # Stream tokens
            async for token in self.core.generate_stream(
                max_tokens=request.max_tokens or 2048,
                temperature=request.temperature,
            ):
                full_content += token

                chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": self.config.model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": token},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0)

            # Check for tool calls
            tool_calls = None
            if request.tools:
                tool_calls = self._parse_tool_calls(full_content)

            if not tool_calls:
                # No tools - send finish and done
                finish_chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": self.config.model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(finish_chunk)}\n\n"
                yield "data: [DONE]\n\n"
                return

            # Separate memory vs client tools
            memory_calls, client_calls = self._separate_tool_calls(tool_calls)

            if memory_calls:
                # Execute memory tools internally (silent - no streaming for this)
                for tc in memory_calls:
                    result_str = await self._execute_memory_tool(tc)
                    self.core.add_tool_result(tc["function"]["name"], result_str)

            if client_calls:
                # Return client tools to client
                finish_chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": self.config.model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"tool_calls": client_calls},
                            "finish_reason": "tool_calls",
                        }
                    ],
                }
                yield f"data: {json.dumps(finish_chunk)}\n\n"
                yield "data: [DONE]\n\n"
                return

            # Only memory tools - loop continues, will generate follow-up response

        # Max iterations - finish normally
        finish_chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": self.config.model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(finish_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    finally:
        self._on_disconnect(connection_id)
```

---

## File Summary

| File | Changes |
|------|---------|
| `src/psyche/handlers/psyche_client.py` | Add tool call capture to `generate_stream()`, add `get_pending_tool_calls()` |
| `src/hermes/app.py` | Add `tool_engine` parameter, rewrite `_process_via_client()` with tool loop |
| `src/hermes/cli.py` | Pass `tool_engine` to Hermes app |
| `src/psyche/server/http.py` | Add server-side memory tool execution in both `_generate_response()` and `_stream_response()` |

---

## Verification

### Test 1: Client-side file tools
```bash
# Terminal 1: Start server
psyche-server

# Terminal 2: Connect client
hermes --server http://localhost:8741 --workspace .

# In hermes, ask:
> Read the contents of pyproject.toml
```

**Expected:**
1. Psyche returns tool_call for `read_file`
2. Hermes ToolActivity widget shows "read_file" in progress
3. Hermes executes tool locally, shows completion
4. Psyche receives result and responds with file summary

### Test 2: Multi-tool sequence
```bash
> List the files in src/hermes and then read the README.md
```

**Expected:**
1. First `list_directory` tool_call returned and executed
2. Then `read_file` tool_call returned and executed
3. Psyche summarizes both results

### Test 3: Server-side memory tools
```bash
> Remember that my favorite color is blue
```

**Expected:**
1. Psyche uses `store_memory` internally (NO tool_call returned to client)
2. Response confirms memory stored
3. ToolActivity widget shows NO activity (memory is server-side)

### Test 4: Memory recall
```bash
# New session or same session
> What's my favorite color?
```

**Expected:**
1. Psyche uses `recall_memory` internally (no tool_call to client)
2. Response correctly states "blue"

### Test 5: Mixed memory + file tools
```bash
> Read README.md and remember the project name
```

**Expected:**
1. `read_file` tool_call returned to client, executed locally
2. Client sends result back to server
3. Server generates follow-up with `store_memory` executed internally
4. Final response confirms both operations

---

## Dependencies

- No new packages required
- Uses existing `ToolEngine` from `src/psyche/tools/tool_engine.py`
- Uses existing `_on_tool_call` callback in Hermes app
- Uses existing `core.retrieve_memories()` and `core.store_memory()` in PsycheCore

## Estimated Sessions

- Steps 1-4 (Client-side tool execution): 1 session
- Step 5 (Server-side memory tools): 1 session
- Testing and fixes: 0.5 session

Total: ~2.5 sessions
