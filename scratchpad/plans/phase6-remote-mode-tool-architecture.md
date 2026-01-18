# Fix: Remote Mode Tool Architecture

## Problem
In remote mode (`hermes --server`), Psyche generates tool_call blocks but nothing happens. Tools are never executed. Additionally, there's confusion about which tools belong where.

## Desired Architecture

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

## Current State (Broken)

1. Psyche's system prompt mentions memory tools (lines 87-92 in server.py)
2. Psyche has internal memory methods (`retrieve_memories`, `store_memory`)
3. BUT: ALL tool_calls are returned to client (including memory)
4. Client never executes ANY tool_calls
5. Memory tools are also sent FROM client (duplication)

## Expected Flow

```
1. User sends message to Hermes
2. Hermes sends request to Psyche WITH file/bash/search tool definitions
   (NO memory tools - those are Psyche's internal tools)
3. Psyche generates response, may include tool_calls:
   - Memory tool_call → Psyche executes INTERNALLY, continues generating
   - Other tool_call → Return to client with finish_reason="tool_calls"
4. Hermes receives tool_calls, executes locally via ToolEngine
5. Hermes sends tool results back to Psyche
6. Psyche generates next response
7. Repeat until finish_reason="stop"
```

### Code Evidence

**`src/hermes/app.py:394-418` - `_process_via_client()`:**
- Only streams tokens and displays them
- Never checks for tool_calls
- Never executes tools
- Never sends results back

**`src/psyche/handlers/psyche_client.py:515-561` - `generate_stream()`:**
- Yields tokens but ignores finish chunk
- Does not extract tool_calls from final SSE chunk
- Non-streaming `generate()` DOES extract tool_calls (line 512)

**`src/psyche/server/http.py:403-426`:**
- Server correctly parses tool_calls from response
- Returns them in final SSE chunk with `finish_reason="tool_calls"`
- But client never processes them

## Fix Implementation

### Files to Modify

#### Server-Side (Psyche executes memory tools internally)

1. **`src/psyche/server/http.py`**
   - Add internal tool execution loop for memory tools
   - After generating response, check for tool_calls
   - If `recall_memory` or `store_memory`: execute via `core.retrieve_memories()` / `core.store_memory()`
   - Add result to context, regenerate response
   - Only return NON-memory tool_calls to client

#### Client-Side (Hermes executes file/bash/search tools)

2. **`src/psyche/handlers/psyche_client.py`**
   - Update `generate_stream()` to capture finish_reason and tool_calls from final SSE chunk
   - Return/store tool_calls so caller can access them after streaming

3. **`src/hermes/app.py`**
   - Add `tool_engine` parameter to `__init__`
   - Update `_process_via_client()` to implement tool execution loop:
     - Check for tool_calls after streaming
     - Execute tools via ToolEngine
     - Send tool results back via client
     - Loop until no more tool_calls
   - Wire up `_on_tool_call` callback for UI updates

4. **`src/hermes/cli.py`**
   - Pass `tool_engine` to Hermes app in remote mode (line 405)
   - Remove memory tool registration in remote mode (server handles those)
   - Only register file/bash/search tools for client-side execution

### Implementation Details

#### 0. HTTP Server - Internal Memory Tool Execution

Add loop in `_generate_response()` and `_stream_response()` to handle memory tools:

```python
MEMORY_TOOLS = {"recall_memory", "store_memory"}

async def _execute_memory_tool(self, tool_call: Dict) -> str:
    """Execute memory tool internally and return result."""
    name = tool_call["function"]["name"]
    args = json.loads(tool_call["function"]["arguments"])

    if name == "recall_memory":
        memories = await self.core.retrieve_memories(args["query"], args.get("n_results", 5))
        return json.dumps({"memories": memories})
    elif name == "store_memory":
        success = await self.core.store_memory(args["content"], ...)
        return json.dumps({"success": success})

async def _generate_response(self, request, connection_id):
    MAX_ITERATIONS = 5

    for _ in range(MAX_ITERATIONS):
        result = await self.core.generate(...)
        tool_calls = self._parse_tool_calls(result["content"])

        if not tool_calls:
            return response  # No tools, done

        # Separate memory vs client tools
        memory_calls = [tc for tc in tool_calls if tc["function"]["name"] in MEMORY_TOOLS]
        client_calls = [tc for tc in tool_calls if tc["function"]["name"] not in MEMORY_TOOLS]

        if memory_calls:
            # Execute memory tools internally
            for tc in memory_calls:
                result = await self._execute_memory_tool(tc)
                self.core.add_tool_result(tc["function"]["name"], result)
            # Continue loop to generate next response
            continue

        if client_calls:
            # Return to client for execution
            return response_with_tool_calls(client_calls)

        return response  # Done
```

#### 1. RemotePsycheClient.generate_stream() - Capture tool_calls

Current code ignores the finish chunk. Need to capture tool_calls:

```python
async def generate_stream(self, ...) -> AsyncIterator[str]:
    # Add instance variable to store tool_calls
    self._last_tool_calls = None
    self._last_finish_reason = None

    async for line in resp.content:
        # ... existing token handling ...

        # NEW: Check finish chunk for tool_calls
        finish_reason = delta.get("finish_reason")
        if finish_reason:
            self._last_finish_reason = finish_reason
            if "tool_calls" in delta:
                self._last_tool_calls = delta["tool_calls"]

def get_pending_tool_calls(self) -> Optional[List[Dict]]:
    """Get tool_calls from last response, if any."""
    if self._last_finish_reason == "tool_calls":
        return self._last_tool_calls
    return None
```

#### 2. Hermes.__init__() - Add tool_engine parameter

```python
def __init__(
    self,
    client: Optional[PsycheClient] = None,
    react_handler: Optional[ReactHandler] = None,
    idle_handler: Optional[IdleHandler] = None,
    elpis_client: Optional["ElpisClient"] = None,
    mnemosyne_client: Optional["MnemosyneClient"] = None,
    tool_engine: Optional["ToolEngine"] = None,  # NEW
    ...
):
    self._tool_engine = tool_engine  # NEW
```

#### 3. Hermes._process_via_client() - Add tool loop

```python
async def _process_via_client(self, text: str) -> None:
    """Process input via remote client with tool execution loop."""
    MAX_ITERATIONS = 10

    for iteration in range(MAX_ITERATIONS):
        # Stream response and display
        chat.start_stream()
        async for token in self._client.generate_stream():
            chat.append_token(token)
        chat.end_stream()

        # Check for tool calls
        tool_calls = self._client.get_pending_tool_calls()
        if not tool_calls or not self._tool_engine:
            break  # Done - no tools or can't execute

        # Execute each tool locally
        for tc in tool_calls:
            name = tc["function"]["name"]
            args = json.loads(tc["function"]["arguments"])

            self._on_tool_call(name, args, None)  # UI: start
            result = await self._tool_engine.execute_tool_call(tc)
            self._on_tool_call(name, args, result)  # UI: complete

            # Send result back to server
            self._client.add_tool_result(name, json.dumps(result))
```

#### 4. hermes/cli.py - Pass tool_engine to app

```python
app = Hermes(
    client=client,
    react_handler=None,
    idle_handler=None,
    elpis_client=None,
    mnemosyne_client=None,
    tool_engine=tool_engine,  # ADD THIS
)
```

## Verification

### Test 1: Client-side file tools
1. Start psyche-server: `psyche-server`
2. Connect hermes: `hermes --server http://localhost:8741`
3. Ask Psyche: "Read the contents of pyproject.toml"
4. Verify:
   - Psyche returns tool_call for `read_file`
   - Hermes executes tool locally
   - ToolActivity widget shows progress
   - Psyche responds with file information

### Test 2: Server-side memory tools
1. Ask Psyche: "Remember that my favorite color is blue"
2. Verify:
   - Psyche uses `store_memory` internally (no tool_call returned to client)
   - Response confirms memory was stored
3. Start new session, ask: "What's my favorite color?"
4. Verify:
   - Psyche uses `recall_memory` internally
   - Response correctly recalls "blue"

### Test 3: Mixed tools
1. Ask Psyche: "Read README.md and remember the key points"
2. Verify:
   - `read_file` returned to client, executed by Hermes
   - `store_memory` executed server-side after file contents received

---

## Tool Organization (Target Architecture)

### Target Tool Map

| Tool | Owner | Execution Location | Notes |
|------|-------|-------------------|-------|
| `recall_memory` | **Psyche Server** | Server-side | Server has Mnemosyne connection |
| `store_memory` | **Psyche Server** | Server-side | Server has Mnemosyne connection |
| `read_file` | **Hermes Client** | Client-side | Needs local workspace |
| `create_file` | **Hermes Client** | Client-side | Needs local workspace |
| `edit_file` | **Hermes Client** | Client-side | Needs local workspace |
| `execute_bash` | **Hermes Client** | Client-side | Security - local only |
| `search_codebase` | **Hermes Client** | Client-side | Needs local codebase |
| `list_directory` | **Hermes Client** | Client-side | Needs local workspace |

### Changes Required

**Psyche Server (`http.py`):**
- Memory tools already in system prompt (lines 87-92 in server.py)
- Add: Internal execution loop for memory tool_calls
- Add: Continue generation after memory tool execution
- Return: Only non-memory tool_calls to client

**Hermes Client:**
- Remove: Memory tools from tool definitions sent to server
- Keep: File/bash/search tools in ToolEngine
- Add: Tool execution loop in `_process_via_client()`

### Why This Architecture?

**Memory tools server-side:**
- Server already has Mnemosyne connection
- Memory is part of Psyche's "self" - not client's business
- Automatic memory retrieval already happens server-side

**Other tools client-side:**
- Need local filesystem access
- Security isolation (client controls what server can access)
- OpenAI-compatible pattern (server returns tool_calls, client executes)
