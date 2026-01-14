"""MCP server entry point for Elpis emotional inference."""

import asyncio
import json
import sys
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    TextContent,
    Tool,
)

from elpis.config.settings import Settings
from elpis.emotion.state import EmotionalState
from elpis.emotion.regulation import HomeostasisRegulator
from elpis.llm.inference import LlamaInference


@dataclass
class StreamState:
    """Tracks state of an active streaming generation."""
    buffer: List[str] = field(default_factory=list)
    cursor: int = 0  # Position of last read
    is_complete: bool = False
    error: Optional[str] = None


# Global state (initialized at startup)
llm: Optional[LlamaInference] = None
emotion_state: Optional[EmotionalState] = None
regulator: Optional[HomeostasisRegulator] = None
server = Server("elpis-inference")
active_streams: Dict[str, StreamState] = {}


def _ensure_initialized() -> None:
    """Ensure server components are initialized."""
    if llm is None or emotion_state is None or regulator is None:
        raise RuntimeError("Server not initialized. Call initialize() first.")


@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available MCP tools."""
    return [
        Tool(
            name="generate",
            description="Generate text completion with emotional modulation",
            inputSchema={
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "description": "Chat messages in OpenAI format",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string"},
                                "content": {"type": "string"},
                            },
                            "required": ["role", "content"],
                        },
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum tokens to generate",
                        "default": 2048,
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Override temperature (null = emotionally modulated)",
                    },
                    "emotional_modulation": {
                        "type": "boolean",
                        "description": "Whether to apply emotional parameter modulation",
                        "default": True,
                    },
                },
                "required": ["messages"],
            },
        ),
        Tool(
            name="function_call",
            description="Generate function/tool calls with emotional modulation",
            inputSchema={
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "description": "Chat messages in OpenAI format",
                        "items": {"type": "object"},
                    },
                    "tools": {
                        "type": "array",
                        "description": "Available tools in OpenAI format",
                        "items": {"type": "object"},
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Override temperature",
                    },
                },
                "required": ["messages", "tools"],
            },
        ),
        Tool(
            name="update_emotion",
            description="Manually trigger an emotional event",
            inputSchema={
                "type": "object",
                "properties": {
                    "event_type": {
                        "type": "string",
                        "description": "Event category (success, failure, novelty, frustration, etc.)",
                    },
                    "intensity": {
                        "type": "number",
                        "description": "Event intensity multiplier (0.0 to 2.0)",
                        "default": 1.0,
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional description for logging",
                    },
                },
                "required": ["event_type"],
            },
        ),
        Tool(
            name="reset_emotion",
            description="Reset emotional state to baseline (neutral homeostasis)",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="get_emotion",
            description="Get current emotional state",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="generate_stream_start",
            description="Start streaming text generation. Returns stream_id for polling.",
            inputSchema={
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "description": "Chat messages in OpenAI format",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string"},
                                "content": {"type": "string"},
                            },
                            "required": ["role", "content"],
                        },
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum tokens to generate",
                        "default": 2048,
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Override temperature (null = emotionally modulated)",
                    },
                    "emotional_modulation": {
                        "type": "boolean",
                        "description": "Whether to apply emotional parameter modulation",
                        "default": True,
                    },
                },
                "required": ["messages"],
            },
        ),
        Tool(
            name="generate_stream_read",
            description="Read new tokens from an active stream. Returns new content since last read.",
            inputSchema={
                "type": "object",
                "properties": {
                    "stream_id": {
                        "type": "string",
                        "description": "Stream ID from generate_stream_start",
                    },
                },
                "required": ["stream_id"],
            },
        ),
        Tool(
            name="generate_stream_cancel",
            description="Cancel an active stream and clean up resources.",
            inputSchema={
                "type": "object",
                "properties": {
                    "stream_id": {
                        "type": "string",
                        "description": "Stream ID to cancel",
                    },
                },
                "required": ["stream_id"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls."""
    _ensure_initialized()

    try:
        if name == "generate":
            result = await _handle_generate(arguments)
        elif name == "function_call":
            result = await _handle_function_call(arguments)
        elif name == "update_emotion":
            result = await _handle_update_emotion(arguments)
        elif name == "reset_emotion":
            result = await _handle_reset_emotion()
        elif name == "get_emotion":
            result = await _handle_get_emotion()
        elif name == "generate_stream_start":
            result = await _handle_generate_stream_start(arguments)
        elif name == "generate_stream_read":
            result = await _handle_generate_stream_read(arguments)
        elif name == "generate_stream_cancel":
            result = await _handle_generate_stream_cancel(arguments)
        else:
            result = {"error": f"Unknown tool: {name}"}

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        logger.exception(f"Tool call failed: {name}")
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def _handle_generate(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handle generate tool call."""
    messages = args["messages"]
    max_tokens = args.get("max_tokens", 2048)
    temperature = args.get("temperature")
    emotional_modulation = args.get("emotional_modulation", True)

    # Apply emotional modulation if enabled and no override
    if emotional_modulation and temperature is None:
        params = emotion_state.get_modulated_params()
        temperature = params["temperature"]
        top_p = params["top_p"]
    else:
        top_p = None

    # Get steering coefficients for emotional expression
    emotion_coefficients = emotion_state.get_steering_coefficients()

    # Run inference
    content = await llm.chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        emotion_coefficients=emotion_coefficients,
    )

    # Update emotional state based on response
    regulator.process_response(content)

    return {
        "content": content,
        "emotional_state": emotion_state.to_dict(),
        "modulated_params": emotion_state.get_modulated_params(),
    }


async def _handle_function_call(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handle function_call tool call."""
    messages = args["messages"]
    tools = args["tools"]
    temperature = args.get("temperature")

    # Apply emotional modulation if no override
    if temperature is None:
        params = emotion_state.get_modulated_params()
        temperature = params["temperature"]

    # Get steering coefficients for emotional expression
    emotion_coefficients = emotion_state.get_steering_coefficients()

    tool_calls = await llm.function_call(
        messages=messages,
        tools=tools,
        temperature=temperature,
        emotion_coefficients=emotion_coefficients,
    )

    return {
        "tool_calls": tool_calls,
        "emotional_state": emotion_state.to_dict(),
    }


async def _handle_update_emotion(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handle update_emotion tool call."""
    event_type = args["event_type"]
    intensity = args.get("intensity", 1.0)
    context = args.get("context")

    regulator.process_event(event_type, intensity, context)

    return emotion_state.to_dict()


async def _handle_reset_emotion() -> Dict[str, Any]:
    """Handle reset_emotion tool call."""
    emotion_state.reset()
    return emotion_state.to_dict()


async def _handle_get_emotion() -> Dict[str, Any]:
    """Handle get_emotion tool call."""
    return emotion_state.to_dict()


async def _handle_generate_stream_start(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle generate_stream_start tool call.

    Starts a background task that generates tokens and stores them in a buffer.
    Returns a stream_id that can be polled with generate_stream_read.
    """
    messages = args["messages"]
    max_tokens = args.get("max_tokens", 2048)
    temperature = args.get("temperature")
    emotional_modulation = args.get("emotional_modulation", True)

    # Apply emotional modulation if enabled and no override
    if emotional_modulation and temperature is None:
        params = emotion_state.get_modulated_params()
        temperature = params["temperature"]
        top_p = params["top_p"]
    else:
        top_p = None

    # Create stream state
    stream_id = str(uuid.uuid4())
    stream_state = StreamState()
    active_streams[stream_id] = stream_state

    # Get steering coefficients for emotional expression
    emotion_coefficients = emotion_state.get_steering_coefficients()

    # Start background task to generate tokens
    async def stream_producer():
        try:
            async for token in llm.chat_completion_stream(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                emotion_coefficients=emotion_coefficients,
            ):
                stream_state.buffer.append(token)
        except Exception as e:
            logger.error(f"Stream generation error: {e}")
            stream_state.error = str(e)
        finally:
            stream_state.is_complete = True
            # Update emotional state based on full response
            if stream_state.buffer:
                full_content = "".join(stream_state.buffer)
                regulator.process_response(full_content)

    asyncio.create_task(stream_producer())

    return {
        "stream_id": stream_id,
        "status": "started",
        "emotional_state": emotion_state.to_dict(),
    }


async def _handle_generate_stream_read(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle generate_stream_read tool call.

    Returns any new tokens since the last read, along with completion status.
    """
    stream_id = args["stream_id"]

    if stream_id not in active_streams:
        return {"error": f"Unknown stream_id: {stream_id}"}

    stream_state = active_streams[stream_id]

    # Get new tokens since last read
    new_tokens = stream_state.buffer[stream_state.cursor:]
    stream_state.cursor = len(stream_state.buffer)
    new_content = "".join(new_tokens)

    result = {
        "new_content": new_content,
        "is_complete": stream_state.is_complete,
        "total_tokens": len(stream_state.buffer),
    }

    if stream_state.error:
        result["error"] = stream_state.error

    # Clean up completed streams
    if stream_state.is_complete:
        result["full_content"] = "".join(stream_state.buffer)
        result["emotional_state"] = emotion_state.to_dict()
        del active_streams[stream_id]

    return result


async def _handle_generate_stream_cancel(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle generate_stream_cancel tool call.

    Cancels an active stream and cleans up resources.
    Note: This marks the stream as cancelled but cannot stop the LLM mid-generation.
    """
    stream_id = args["stream_id"]

    if stream_id not in active_streams:
        return {"error": f"Unknown stream_id: {stream_id}"}

    # Mark as complete and remove
    stream_state = active_streams[stream_id]
    stream_state.is_complete = True
    del active_streams[stream_id]

    return {
        "status": "cancelled",
        "stream_id": stream_id,
    }


@server.list_resources()
async def list_resources() -> List[Resource]:
    """List available MCP resources."""
    return [
        Resource(
            uri="emotion://state",
            name="Emotional State",
            description="Current emotional state as JSON",
            mimeType="application/json",
        ),
        Resource(
            uri="emotion://events",
            name="Event Types",
            description="Available emotional event types and their effects",
            mimeType="application/json",
        ),
    ]


@server.read_resource()
async def read_resource(uri: str) -> str:
    """Read an MCP resource."""
    _ensure_initialized()

    if uri == "emotion://state":
        return json.dumps(emotion_state.to_dict(), indent=2)
    elif uri == "emotion://events":
        return json.dumps(regulator.get_available_events(), indent=2)
    else:
        raise ValueError(f"Unknown resource: {uri}")


def initialize(settings: Optional[Settings] = None) -> None:
    """
    Initialize server components.

    Args:
        settings: Optional settings object (uses defaults if not provided)
    """
    global llm, emotion_state, regulator

    if settings is None:
        settings = Settings()

    # Configure logging
    logger.remove()

    # Check if we should suppress stderr logging (e.g., when run as subprocess by Psyche TUI)
    # ELPIS_QUIET env var is set by Psyche to prevent logging from breaking the TUI
    import os
    quiet_mode = os.environ.get("ELPIS_QUIET", "").lower() in ("1", "true", "yes")

    if quiet_mode:
        # Log to file when running as subprocess of a TUI
        from pathlib import Path
        log_dir = Path.home() / ".elpis"
        log_dir.mkdir(exist_ok=True)
        logger.add(
            log_dir / "elpis-server.log",
            level=settings.logging.level.upper(),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            rotation="10 MB",
        )
    else:
        # Log to stderr when running standalone
        logger.add(
            sys.stderr,
            level=settings.logging.level.upper(),
            format="<level>{level: <8}</level> | {message}",
        )

    logger.info("Initializing Elpis inference server...")

    # Initialize emotional state
    emotion_state = EmotionalState()
    regulator = HomeostasisRegulator(emotion_state)
    logger.info("Emotional system initialized")

    # Initialize LLM
    llm = LlamaInference(settings.model)
    logger.info("LLM initialized")


async def run_server() -> None:
    """Run the MCP server."""
    logger.info("Starting Elpis MCP server...")

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main() -> None:
    """Main entry point for elpis-server command."""
    try:
        initialize()
        asyncio.run(run_server())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.exception(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
