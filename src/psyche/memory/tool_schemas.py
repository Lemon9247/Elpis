"""Memory tool schemas for LLM tool definitions.

These definitions are shared between:
- psyche/server/http.py: OpenAI-compatible API (uses function format)
- psyche/server/mcp.py: MCP server (uses Tool format)
"""

# Memory tool definitions in OpenAI function-calling format
# Used by http.py for the chat completions endpoint
MEMORY_TOOL_DEFINITIONS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": "recall_memory",
            "description": "Search for relevant memories from long-term storage",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "n_results": {
                        "type": "integer",
                        "default": 5,
                        "description": "Number of results",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "store_memory",
            "description": "Store important information in long-term memory",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Content to store"},
                    "importance": {
                        "type": "number",
                        "description": "Importance score 0.0-1.0 (default: 0.5)",
                        "default": 0.5,
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for categorization",
                    },
                },
                "required": ["content"],
            },
        },
    },
]

# Memory tool schemas in MCP format
# Used by mcp.py for the MCP server
RECALL_MEMORY_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Search query for memories",
        },
        "n": {
            "type": "integer",
            "description": "Number of memories to retrieve (default: 5)",
            "default": 5,
        },
    },
    "required": ["query"],
}

STORE_MEMORY_SCHEMA = {
    "type": "object",
    "properties": {
        "content": {
            "type": "string",
            "description": "Memory content to store",
        },
        "importance": {
            "type": "number",
            "description": "Importance score 0.0-1.0 (default: 0.5)",
            "default": 0.5,
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional tags for categorization",
        },
    },
    "required": ["content"],
}
