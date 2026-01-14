#!/usr/bin/env python3
"""MCP Server example: Using Elpis as an MCP server.

This example demonstrates how to interact with Elpis MCP server,
including emotional state management and modulated generation.

Note: This requires the MCP server to be running separately.
Start it with: elpis-server
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


async def call_mcp_tool(tool_name: str, arguments: dict) -> dict:
    """
    Simulate an MCP tool call.

    In a real scenario, this would use the MCP client protocol to
    communicate with the server over stdio.
    """
    print(f"\n📞 Calling MCP tool: {tool_name}")
    print(f"   Arguments: {json.dumps(arguments, indent=2)}")

    # This is a simulation - in reality you'd use an MCP client
    # For demonstration purposes, we'll show what the call would look like
    return {"status": "simulated", "tool": tool_name, "args": arguments}


async def main():
    """Demonstrate MCP server usage patterns."""
    print("=== Elpis MCP Server Usage Example ===\n")

    print("This example demonstrates how to use Elpis as an MCP server.")
    print("The server provides emotional inference with state management.\n")

    # Example 1: Get current emotional state
    print("\n" + "=" * 60)
    print("Example 1: Getting Emotional State")
    print("=" * 60)

    result = await call_mcp_tool("get_emotion", {})
    print("\nExpected response:")
    print(json.dumps({
        "valence": 0.0,
        "arousal": 0.0,
        "quadrant": "excited",  # At (0,0), defaults to excited
        "baseline_valence": 0.0,
        "baseline_arousal": 0.0,
        "update_count": 0,
    }, indent=2))

    # Example 2: Update emotional state
    print("\n" + "=" * 60)
    print("Example 2: Triggering Emotional Events")
    print("=" * 60)

    events = [
        ("success", 1.0, "Successfully completed a task"),
        ("novelty", 1.2, "Discovered something new and interesting"),
        ("frustration", 1.5, "Encountered a blocking issue"),
        ("failure", 0.8, "Test failed unexpectedly"),
    ]

    for event_type, intensity, context in events:
        result = await call_mcp_tool("update_emotion", {
            "event_type": event_type,
            "intensity": intensity,
            "context": context,
        })

        print(f"\n✨ Event: {event_type} (intensity={intensity})")
        print(f"   Context: {context}")
        print(f"   → Emotional state updated")

    # Example 3: Generate with emotional modulation
    print("\n" + "=" * 60)
    print("Example 3: Emotionally Modulated Generation")
    print("=" * 60)

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "How are you feeling about our progress?"},
    ]

    result = await call_mcp_tool("generate", {
        "messages": messages,
        "max_tokens": 150,
        "emotional_modulation": True,  # Enable emotional modulation
    })

    print("\nExpected response:")
    print(json.dumps({
        "content": "[Generated response reflecting current emotional state]",
        "emotional_state": {
            "valence": 0.3,  # Updated by events
            "arousal": 0.5,
            "quadrant": "excited",
        },
        "modulated_params": {
            "temperature": 0.65,  # Adjusted based on arousal
            "top_p": 0.92,        # Adjusted based on valence
        },
    }, indent=2))

    # Example 4: Streaming generation
    print("\n" + "=" * 60)
    print("Example 4: Streaming with Emotional Modulation")
    print("=" * 60)

    # Start stream
    result = await call_mcp_tool("generate_stream_start", {
        "messages": messages,
        "max_tokens": 100,
        "emotional_modulation": True,
    })

    print("\nExpected response:")
    print(json.dumps({
        "stream_id": "uuid-here",
        "status": "started",
        "emotional_state": {"valence": 0.3, "arousal": 0.5},
    }, indent=2))

    # Read from stream (would be done in a loop)
    stream_id = "uuid-here"
    result = await call_mcp_tool("generate_stream_read", {
        "stream_id": stream_id,
    })

    print("\nStream read response:")
    print(json.dumps({
        "new_content": "I'm feeling quite energized ",
        "is_complete": False,
        "total_tokens": 5,
    }, indent=2))

    # Example 5: Reset emotional state
    print("\n" + "=" * 60)
    print("Example 5: Resetting Emotional State")
    print("=" * 60)

    result = await call_mcp_tool("reset_emotion", {})

    print("\nExpected response:")
    print(json.dumps({
        "valence": 0.0,
        "arousal": 0.0,
        "quadrant": "excited",
        "update_count": 0,
    }, indent=2))

    # Summary
    print("\n" + "=" * 60)
    print("MCP Server Tools Summary")
    print("=" * 60)

    tools = [
        ("generate", "Generate text with emotional modulation"),
        ("generate_stream_start", "Start streaming generation"),
        ("generate_stream_read", "Read tokens from active stream"),
        ("generate_stream_cancel", "Cancel active stream"),
        ("function_call", "Generate tool/function calls"),
        ("update_emotion", "Trigger an emotional event"),
        ("reset_emotion", "Reset state to baseline"),
        ("get_emotion", "Get current emotional state"),
    ]

    for tool, desc in tools:
        print(f"\n  {tool:25s} - {desc}")

    print("\n" + "=" * 60)
    print("\nTo actually use these tools:")
    print("  1. Start the MCP server: elpis-server")
    print("  2. Connect via MCP client (e.g., from Psyche)")
    print("  3. Call tools using the MCP protocol")
    print("\nOr use the psyche CLI for an integrated experience:")
    print("  psyche")


if __name__ == "__main__":
    asyncio.run(main())
