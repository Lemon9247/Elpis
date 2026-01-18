"""
HTTP Server - OpenAI-compatible API for Psyche.

Provides /v1/chat/completions endpoint that external agents
(Aider, OpenCode, Continue, etc.) can connect to.

Key behaviors:
- Accepts tool definitions from client (client owns tools)
- Returns tool_calls in response, does NOT execute
- Supports streaming via SSE
- Tracks connections for dream scheduling
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel

if TYPE_CHECKING:
    from psyche.core.server import PsycheCore
    from psyche.server.daemon import PsycheDaemon


# Memory tools that should be executed server-side
MEMORY_TOOLS = {"recall_memory", "store_memory"}

# Maximum characters for memory results to prevent context overflow
MAX_MEMORY_RESULT_CHARS = 2000


# --- OpenAI-compatible request/response models ---


class ChatMessage(BaseModel):
    """OpenAI chat message format."""

    role: str  # "system", "user", "assistant", "tool"
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None  # For tool messages


class ToolFunction(BaseModel):
    """Tool function definition."""

    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class Tool(BaseModel):
    """Tool definition in OpenAI format."""

    type: str = "function"
    function: ToolFunction


class ChatCompletionRequest(BaseModel):
    """OpenAI chat completion request."""

    model: str = "psyche"
    messages: List[ChatMessage]
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[str] = "auto"
    stream: bool = False
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    user: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    """Single choice in completion response."""

    index: int
    message: ChatMessage
    finish_reason: str


class Usage(BaseModel):
    """Token usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


class ChatCompletionChunk(BaseModel):
    """Streaming chunk for SSE."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[Dict[str, Any]]


# --- HTTP Server ---


@dataclass
class HTTPServerConfig:
    """Configuration for HTTP server."""

    host: str = "127.0.0.1"
    port: int = 8741
    model_name: str = "psyche"


class PsycheHTTPServer:
    """
    FastAPI server providing OpenAI-compatible chat completions.

    The server:
    - Accepts chat messages and tool definitions from clients
    - Uses PsycheCore for memory-enriched, emotionally-modulated inference
    - Returns responses with tool_calls (client executes tools)
    - Supports streaming via Server-Sent Events
    """

    def __init__(
        self,
        core: PsycheCore,
        daemon: Optional[PsycheDaemon] = None,
        config: Optional[HTTPServerConfig] = None,
    ):
        """
        Initialize HTTP server.

        Args:
            core: PsycheCore instance for inference
            daemon: Optional daemon for connection tracking
            config: Server configuration
        """
        self.core = core
        self.daemon = daemon
        self.config = config or HTTPServerConfig()

        self.app = FastAPI(
            title="Psyche Server",
            description="OpenAI-compatible API with emotional memory",
            version="0.1.0",
        )
        self._setup_routes()

        # Active connections for tracking
        self._connections: Dict[str, float] = {}

    def _setup_routes(self) -> None:
        """Set up FastAPI routes."""

        @self.app.get("/health")
        async def health() -> Dict[str, Any]:
            """Health check endpoint."""
            return {
                "status": "ok",
                "connections": len(self._connections),
                "model": self.config.model_name,
            }

        @self.app.get("/v1/models")
        async def list_models() -> Dict[str, Any]:
            """List available models (OpenAI-compatible)."""
            return {
                "object": "list",
                "data": [
                    {
                        "id": self.config.model_name,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "psyche",
                    }
                ],
            }

        @self.app.post("/v1/chat/completions")
        async def chat_completions(
            request: ChatCompletionRequest,
            raw_request: Request,
        ) -> Any:
            """
            OpenAI-compatible chat completions.

            Psyche internally:
            1. Retrieves relevant memories
            2. Builds enriched context
            3. Calls Elpis for inference
            4. Auto-stores important exchanges
            5. Returns response (may include tool_calls)
            """
            # Generate connection ID for tracking
            client_id = raw_request.client.host if raw_request.client else "unknown"
            connection_id = f"{client_id}:{uuid.uuid4().hex[:8]}"

            try:
                # Track connection
                self._on_connect(connection_id)

                # Update tool descriptions if provided
                if request.tools:
                    tool_desc = self._format_tool_descriptions(request.tools)
                    self.core.set_tool_descriptions(tool_desc)

                # Process messages
                await self._process_messages(request.messages)

                # Generate response
                if request.stream:
                    return StreamingResponse(
                        self._stream_response(request, connection_id),
                        media_type="text/event-stream",
                    )
                else:
                    return await self._generate_response(request, connection_id)

            except Exception as e:
                logger.error(f"Error in chat completion: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    def _on_connect(self, connection_id: str) -> None:
        """Handle client connection."""
        self._connections[connection_id] = time.time()
        logger.debug(f"Client connected: {connection_id}")

        if self.daemon:
            self.daemon.on_client_connect(connection_id)

    def _on_disconnect(self, connection_id: str) -> None:
        """Handle client disconnection."""
        self._connections.pop(connection_id, None)
        logger.debug(f"Client disconnected: {connection_id}")

        if self.daemon:
            self.daemon.on_client_disconnect(connection_id)

    def _format_tool_descriptions(self, tools: List[Tool]) -> str:
        """Convert OpenAI tool format to description string for system prompt."""
        lines = ["### Available Tools\n"]

        for tool in tools:
            func = tool.function
            lines.append(f"**{func.name}**")
            if func.description:
                lines.append(f"  {func.description}")

            if func.parameters:
                props = func.parameters.get("properties", {})
                required = func.parameters.get("required", [])

                if props:
                    lines.append("  Parameters:")
                    for name, schema in props.items():
                        req_mark = " (required)" if name in required else ""
                        desc = schema.get("description", "")
                        ptype = schema.get("type", "any")
                        lines.append(f"    - {name}: {ptype}{req_mark} - {desc}")

            lines.append("")

        return "\n".join(lines)

    async def _process_messages(self, messages: List[ChatMessage]) -> None:
        """
        Process incoming messages into context.

        The OpenAI API pattern is stateless - the client sends the full
        message history each time. We clear the context and rebuild it
        from the provided messages to match this pattern.

        Memory retrieval still happens for the latest user message.
        """
        # Clear existing context to avoid duplication
        self.core.clear_context()

        # Process all messages except the last user message
        # (which gets special handling for memory retrieval)
        last_user_idx = -1
        for i, msg in enumerate(messages):
            if msg.role == "user":
                last_user_idx = i

        for i, msg in enumerate(messages):
            if msg.role == "system":
                # Skip system messages - PsycheCore has its own system prompt
                continue
            elif msg.role == "user" and msg.content:
                if i == last_user_idx:
                    # Last user message - use add_user_message for memory retrieval
                    await self.core.add_user_message(msg.content)
                else:
                    # Historical user message - add directly to context
                    self.core._context.add_message("user", msg.content)
            elif msg.role == "tool" and msg.content:
                tool_name = msg.name or "unknown"
                self.core.add_tool_result(tool_name, msg.content)
            elif msg.role == "assistant" and msg.content:
                # Historical assistant message - add directly to context
                self.core._context.add_message("assistant", msg.content)

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
                try:
                    result = await self.core.generate(
                        max_tokens=request.max_tokens or 2048,
                        temperature=request.temperature,
                    )
                except RuntimeError as e:
                    # Handle context overflow
                    error_msg = str(e)
                    if "exceed context" in error_msg.lower() or "context window" in error_msg.lower():
                        logger.warning(f"Context overflow during memory tool loop: {e}")
                        accumulated_content += "\n\n[Context limit reached - some memory results may be truncated]"
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
                    else:
                        raise

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

                # Stream tokens (with error handling for context overflow)
                try:
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
                except RuntimeError as e:
                    # Handle context overflow - stream error message and finish
                    error_msg = str(e)
                    if "exceed context" in error_msg.lower() or "context window" in error_msg.lower():
                        logger.warning(f"Context overflow during memory tool loop: {e}")
                        error_content = "\n\n[Context limit reached - some memory results may be truncated]"
                        chunk = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": self.config.model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": error_content},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                        # Send finish and done
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
                    else:
                        raise  # Re-raise non-context errors

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

    def _parse_tool_calls(self, content: str) -> Optional[List[Dict[str, Any]]]:
        """
        Parse tool calls from response content.

        Looks for ```tool_call blocks in the response.
        """
        import re

        pattern = r"```tool_call\s*\n?(.*?)\n?```"
        matches = re.findall(pattern, content, re.DOTALL)

        if not matches:
            return None

        tool_calls = []
        for i, match in enumerate(matches):
            try:
                call_data = json.loads(match.strip())
                tool_calls.append(
                    {
                        "id": f"call_{uuid.uuid4().hex[:12]}",
                        "type": "function",
                        "function": {
                            "name": call_data.get("name", ""),
                            "arguments": json.dumps(call_data.get("arguments", {})),
                        },
                    }
                )
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse tool call: {match[:100]}")
                continue

        return tool_calls if tool_calls else None

    def _strip_tool_calls(self, content: str) -> str:
        """Remove tool call blocks from content."""
        import re

        return re.sub(r"```tool_call\s*\n?.*?\n?```", "", content, flags=re.DOTALL).strip()

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

            # Use core's store_memory method
            success = await self.core.store_memory(
                content=content,
                importance=0.7,  # Default importance for explicit stores
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

    @property
    def connection_count(self) -> int:
        """Get number of active connections."""
        return len(self._connections)
