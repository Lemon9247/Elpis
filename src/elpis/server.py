"""
Elpis MCP Server - Emotional inference engine.

Provides LLM inference with emotional modulation via:
- Valence-arousal emotional state model
- Sampling parameter modulation (temperature, top_p)
- Steering vectors (transformers backend, experimental)
- Homeostatic regulation with decay toward baseline

MCP Tools:
- generate: Text generation with emotional modulation
- generate_stream: Streaming generation (start/poll/stop)
- set_emotion: Directly set emotional state
- process_event: Update emotion via events (success, failure, etc.)
- get_emotion: Query current emotional state
- get_capabilities: Query model context length and configuration

Backends:
- llama-cpp: GGUF models, sampling parameter modulation
- transformers: HuggingFace models, steering vector support
"""

import asyncio
import json
import os
import signal
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Stream management limits
MAX_ACTIVE_STREAMS = 100
STREAM_TTL_SECONDS = 600  # 10 minutes

from loguru import logger
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    TextContent,
    Tool,
)

from elpis.config.settings import Settings
from elpis.emotion.state import EmotionalState, TrajectoryConfig
from elpis.emotion.regulation import HomeostasisRegulator
from elpis.llm.base import InferenceEngine
from elpis.llm.backends import create_backend


@dataclass
class StreamState:
    """Tracks state of an active streaming generation."""
    buffer: List[str] = field(default_factory=list)
    cursor: int = 0  # Position of last read
    is_complete: bool = False
    error: Optional[str] = None
    task: Optional[asyncio.Task] = None  # Reference to producer task
    created_at: float = field(default_factory=time.monotonic)  # For TTL enforcement


@dataclass
class ServerContext:
    """Container for all server dependencies.

    This dataclass holds all the initialized components needed for the
    server to function, providing a clean dependency injection pattern
    instead of global variables.

    Attributes:
        llm: The inference engine for LLM operations
        emotion_state: Current emotional state (valence/arousal)
        regulator: Homeostasis regulator for emotional updates
        settings: Server configuration settings
        active_streams: Dict mapping stream IDs to their state
    """
    llm: InferenceEngine
    emotion_state: EmotionalState
    regulator: HomeostasisRegulator
    settings: Settings
    active_streams: Dict[str, StreamState] = field(default_factory=dict)


# Global context and server (initialized at startup)
_context: Optional[ServerContext] = None
server = Server("elpis-inference")


def get_context() -> ServerContext:
    """Get the current server context.

    Returns:
        The initialized ServerContext

    Raises:
        RuntimeError: If server is not initialized
    """
    if _context is None:
        raise RuntimeError("Server not initialized. Call initialize() first.")
    return _context


async def _cleanup_stale_streams(ctx: ServerContext) -> int:
    """Remove streams that have exceeded TTL.

    Args:
        ctx: Server context containing active_streams

    Returns:
        Number of streams removed
    """
    now = time.monotonic()
    stale_ids = [
        stream_id
        for stream_id, state in ctx.active_streams.items()
        if now - state.created_at > STREAM_TTL_SECONDS
    ]

    for stream_id in stale_ids:
        state = ctx.active_streams.pop(stream_id, None)
        if state and state.task and not state.task.done():
            state.task.cancel()
            try:
                await state.task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.warning(f"Error cancelling stale stream {stream_id}: {e}")
        logger.debug(f"Cleaned up stale stream {stream_id}")

    if stale_ids:
        logger.info(f"Cleaned up {len(stale_ids)} stale stream(s)")

    return len(stale_ids)


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
        Tool(
            name="get_capabilities",
            description="Get server capabilities including context window size and model info.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls."""
    ctx = get_context()

    try:
        if name == "generate":
            result = await _handle_generate(ctx, arguments)
        elif name == "function_call":
            result = await _handle_function_call(ctx, arguments)
        elif name == "update_emotion":
            result = await _handle_update_emotion(ctx, arguments)
        elif name == "reset_emotion":
            result = await _handle_reset_emotion(ctx)
        elif name == "get_emotion":
            result = await _handle_get_emotion(ctx)
        elif name == "generate_stream_start":
            result = await _handle_generate_stream_start(ctx, arguments)
        elif name == "generate_stream_read":
            result = await _handle_generate_stream_read(ctx, arguments)
        elif name == "generate_stream_cancel":
            result = await _handle_generate_stream_cancel(ctx, arguments)
        elif name == "get_capabilities":
            result = _handle_get_capabilities(ctx)
        else:
            result = {"error": f"Unknown tool: {name}"}

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        logger.exception(f"Tool call failed: {name}")
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def _handle_generate(ctx: ServerContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """Handle generate tool call."""
    messages = args["messages"]
    max_tokens = args.get("max_tokens", 2048)
    temperature = args.get("temperature")
    emotional_modulation = args.get("emotional_modulation", True)

    # Apply emotional modulation if enabled and no override
    if emotional_modulation and temperature is None:
        params = ctx.emotion_state.get_modulated_params()
        temperature = params["temperature"]
        top_p = params["top_p"]
    else:
        top_p = None

    # Get steering coefficients for emotional expression
    emotion_coefficients = ctx.emotion_state.get_steering_coefficients()

    # Run inference
    content = await ctx.llm.chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        emotion_coefficients=emotion_coefficients,
    )

    # Update emotional state based on response
    ctx.regulator.process_response(content)

    return {
        "content": content,
        "emotional_state": ctx.emotion_state.to_dict(),
        "modulated_params": ctx.emotion_state.get_modulated_params(),
    }


async def _handle_function_call(ctx: ServerContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """Handle function_call tool call."""
    messages = args["messages"]
    tools = args["tools"]
    temperature = args.get("temperature")

    # Apply emotional modulation if no override
    if temperature is None:
        params = ctx.emotion_state.get_modulated_params()
        temperature = params["temperature"]

    # Get steering coefficients for emotional expression
    emotion_coefficients = ctx.emotion_state.get_steering_coefficients()

    tool_calls = await ctx.llm.function_call(
        messages=messages,
        tools=tools,
        temperature=temperature,
        emotion_coefficients=emotion_coefficients,
    )

    return {
        "tool_calls": tool_calls,
        "emotional_state": ctx.emotion_state.to_dict(),
    }


async def _handle_update_emotion(ctx: ServerContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """Handle update_emotion tool call."""
    event_type = args["event_type"]
    intensity = args.get("intensity", 1.0)
    context = args.get("context")

    ctx.regulator.process_event(event_type, intensity, context)

    return ctx.emotion_state.to_dict()


async def _handle_reset_emotion(ctx: ServerContext) -> Dict[str, Any]:
    """Handle reset_emotion tool call."""
    ctx.emotion_state.reset()
    return ctx.emotion_state.to_dict()


async def _handle_get_emotion(ctx: ServerContext) -> Dict[str, Any]:
    """Handle get_emotion tool call."""
    return ctx.emotion_state.to_dict()


async def _handle_generate_stream_start(ctx: ServerContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle generate_stream_start tool call.

    Starts a background task that generates tokens and stores them in a buffer.
    Returns a stream_id that can be polled with generate_stream_read.
    """
    # Clean up stale streams before checking limits
    await _cleanup_stale_streams(ctx)

    # Check stream limit
    if len(ctx.active_streams) >= MAX_ACTIVE_STREAMS:
        return {
            "error": f"Maximum concurrent streams ({MAX_ACTIVE_STREAMS}) reached. "
            "Cancel existing streams or wait for them to complete.",
        }

    messages = args["messages"]
    max_tokens = args.get("max_tokens", 2048)
    temperature = args.get("temperature")
    emotional_modulation = args.get("emotional_modulation", True)

    # Apply emotional modulation if enabled and no override
    if emotional_modulation and temperature is None:
        params = ctx.emotion_state.get_modulated_params()
        temperature = params["temperature"]
        top_p = params["top_p"]
    else:
        top_p = None

    # Create stream state
    stream_id = str(uuid.uuid4())
    stream_state = StreamState()
    ctx.active_streams[stream_id] = stream_state

    # Get steering coefficients for emotional expression
    emotion_coefficients = ctx.emotion_state.get_steering_coefficients()

    # Capture context references for the closure
    llm = ctx.llm
    regulator = ctx.regulator

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

    # Store task reference for proper lifecycle management
    stream_state.task = asyncio.create_task(stream_producer())

    return {
        "stream_id": stream_id,
        "status": "started",
        "emotional_state": ctx.emotion_state.to_dict(),
    }


async def _handle_generate_stream_read(ctx: ServerContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle generate_stream_read tool call.

    Returns any new tokens since the last read, along with completion status.
    """
    stream_id = args["stream_id"]

    if stream_id not in ctx.active_streams:
        return {"error": f"Unknown stream_id: {stream_id}"}

    stream_state = ctx.active_streams[stream_id]

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
        result["emotional_state"] = ctx.emotion_state.to_dict()
        # Wait for task to fully complete before cleanup
        if stream_state.task and not stream_state.task.done():
            try:
                await asyncio.wait_for(stream_state.task, timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            except Exception as e:
                logger.warning(f"Stream task error during cleanup: {e}")
        # Safe deletion (handles concurrent access)
        ctx.active_streams.pop(stream_id, None)

    return result


async def _handle_generate_stream_cancel(ctx: ServerContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle generate_stream_cancel tool call.

    Cancels an active stream and cleans up resources.
    """
    stream_id = args["stream_id"]

    stream_state = ctx.active_streams.get(stream_id)
    if stream_state is None:
        return {"error": f"Unknown stream_id: {stream_id}"}

    # Cancel the producer task if still running
    if stream_state.task and not stream_state.task.done():
        stream_state.task.cancel()
        try:
            await stream_state.task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"Error cancelling stream task: {e}")

    # Mark as complete and remove safely
    stream_state.is_complete = True
    ctx.active_streams.pop(stream_id, None)

    return {
        "status": "cancelled",
        "stream_id": stream_id,
    }


def _handle_get_capabilities(ctx: ServerContext) -> Dict[str, Any]:
    """
    Handle get_capabilities tool call.

    Returns server capabilities including context window size and model info.
    """
    return {
        "context_length": ctx.settings.model.context_length,
        "max_tokens": ctx.settings.model.max_tokens,
        "backend": ctx.settings.model.backend,
        "model_path": ctx.settings.model.path,
        "temperature": ctx.settings.model.temperature,
        "top_p": ctx.settings.model.top_p,
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
    ctx = get_context()

    if uri == "emotion://state":
        return json.dumps(ctx.emotion_state.to_dict(), indent=2)
    elif uri == "emotion://events":
        return json.dumps(ctx.regulator.get_available_events(), indent=2)
    else:
        raise ValueError(f"Unknown resource: {uri}")


def initialize(settings: Optional[Settings] = None) -> ServerContext:
    """
    Initialize server components.

    Args:
        settings: Optional settings object (uses defaults if not provided)

    Returns:
        Initialized ServerContext
    """
    global _context

    if settings is None:
        settings = Settings()

    # Configure logging
    logger.remove()

    # Check if we should suppress stderr logging (e.g., when run as subprocess by Psyche TUI)
    # ELPIS_QUIET env var is set by Psyche to prevent logging from breaking the TUI
    quiet_mode = os.environ.get("ELPIS_QUIET", "").lower() in ("1", "true", "yes")

    if quiet_mode:
        # Log to file when running as subprocess of a TUI
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

    # Initialize emotional state with settings
    emotion_state = EmotionalState(
        baseline_valence=settings.emotion.baseline_valence,
        baseline_arousal=settings.emotion.baseline_arousal,
        steering_strength=settings.emotion.steering_strength,
    )
    # Apply trajectory config from settings
    emotion_state._trajectory_config = TrajectoryConfig.from_settings(settings.emotion)
    regulator = HomeostasisRegulator(emotion_state)
    logger.info("Emotional system initialized")

    # Initialize LLM using the backend factory
    try:
        llm = create_backend(settings.model)
        logger.info(f"Backend initialized: {settings.model.backend}")
        logger.info(f"  - Supports steering: {llm.SUPPORTS_STEERING}")
        logger.info(f"  - Modulation type: {llm.MODULATION_TYPE}")
    except ValueError as e:
        logger.error(f"Failed to create backend: {e}")
        raise
    except ImportError as e:
        logger.error(
            f"Failed to import backend '{settings.model.backend}': {e}. "
            "Check that required dependencies are installed."
        )
        raise

    # Create and store context
    _context = ServerContext(
        llm=llm,
        emotion_state=emotion_state,
        regulator=regulator,
        settings=settings,
    )

    return _context


async def run_server() -> None:
    """Run the MCP server."""
    logger.info("Starting Elpis MCP server...")

    try:
        async with stdio_server() as (read_stream, write_stream):
            logger.debug("stdio streams connected, running MCP server...")
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )
            logger.debug("MCP server.run() completed")
    except Exception as e:
        logger.error(f"MCP server error: {type(e).__name__}: {e}")
        raise
    finally:
        logger.info("MCP server shutting down")


def main() -> None:
    """Main entry point for elpis-server command."""
    # Set up signal handlers for diagnostics
    def signal_handler(signum: int, frame: Any) -> None:
        sig_name = signal.Signals(signum).name
        logger.warning(f"Received signal {sig_name} ({signum})")
        sys.exit(128 + signum)

    # Only handle SIGTERM - SIGINT is handled by asyncio
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        initialize()
        asyncio.run(run_server())
        logger.info("Server exited normally")
    except KeyboardInterrupt:
        logger.info("Server stopped by user (SIGINT)")
    except Exception as e:
        logger.exception(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
