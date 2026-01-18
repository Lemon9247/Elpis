# LLM API Standards Research Report

**Researcher**: API Standards Researcher
**Date**: 2026-01-12
**Task**: Phase 2 Architecture - API Standards Investigation

---

## Executive Summary

This report provides a comprehensive analysis of LLM API standards, with particular focus on OpenAI's API specification as the de facto industry standard. Key findings include:

1. **OpenAI is transitioning to a new Responses API** - The new `/v1/responses` endpoint is replacing `/v1/chat/completions` as the recommended API for new projects, offering 40-80% better cache utilization and 3% intelligence improvements.

2. **Tool/function calling is converging** - While OpenAI and Anthropic use different JSON schema structures, both support sophisticated tool use with structured outputs and strict schema adherence.

3. **No universal LLM API standard exists** - However, OpenAI's format has become the de facto standard, with many providers offering OpenAI-compatible interfaces.

4. **Versioning strategies are critical** - Both providers use model pinning and API versioning, with long deprecation cycles (6-12 months) for backward compatibility.

### Recommendations for Elpis Implementation

1. **Primary compatibility target**: OpenAI Chat Completions API (for current compatibility) with Responses API consideration (for future-proofing)
2. **Tool calling format**: Support both OpenAI (`parameters`) and Anthropic (`input_schema`) styles
3. **Versioning**: Implement URI-based versioning (`/v1/`) with header-based fine-grained control
4. **Authentication**: Support Bearer token authentication with organization/project headers
5. **Streaming**: Implement SSE (Server-Sent Events) protocol for streaming responses

---

## Table of Contents

1. [OpenAI API Specification](#openai-api-specification)
2. [Anthropic Claude API](#anthropic-claude-api)
3. [Tool/Function Calling Standards](#tool-function-calling-standards)
4. [Streaming Protocols](#streaming-protocols)
5. [Authentication and Security](#authentication-and-security)
6. [Error Handling](#error-handling)
7. [API Versioning and Compatibility](#api-versioning-and-compatibility)
8. [Emerging Standards](#emerging-standards)
9. [OpenAPI/Swagger Best Practices](#openapi-swagger-best-practices)
10. [Implementation Recommendations for Elpis](#implementation-recommendations-for-elpis)

---

## 1. OpenAI API Specification

### 1.1 Chat Completions API (Current Standard)

The Chat Completions API is currently the most widely used format for LLM interactions.

**Endpoint**: `POST https://api.openai.com/v1/chat/completions`

#### Request Schema

```json
{
  "model": "gpt-4o",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Hello, how are you?"
    },
    {
      "role": "assistant",
      "content": "I'm doing well, thank you!"
    },
    {
      "role": "user",
      "content": "Can you help me with a task?"
    }
  ],
  "temperature": 0.7,
  "max_tokens": 1000,
  "top_p": 1.0,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0,
  "stream": false,
  "stop": null,
  "n": 1
}
```

#### Key Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `model` | string | Model identifier (e.g., "gpt-4o", "gpt-5.2") | Required |
| `messages` | array | Array of message objects with role and content | Required |
| `temperature` | float | Sampling temperature (0-2). Higher = more random | 1.0 |
| `max_tokens` | integer | Maximum tokens to generate | Model-specific |
| `top_p` | float | Nucleus sampling parameter | 1.0 |
| `stream` | boolean | Enable streaming responses via SSE | false |
| `tools` | array | Array of tool/function definitions | null |
| `tool_choice` | string/object | Control tool calling behavior | "auto" |
| `response_format` | object | Specify output format (e.g., JSON) | null |

#### Response Schema (Non-streaming)

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "gpt-4o",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "I'd be happy to help you with your task. What do you need assistance with?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 18,
    "total_tokens": 43
  }
}
```

#### Message Roles

- **`system`**: Sets the behavior/context for the assistant
- **`user`**: User messages in the conversation
- **`assistant`**: Assistant responses (can include previous turns or prefilling)
- **`tool`**: Tool/function call results (when using function calling)

### 1.2 Responses API (Next Generation)

**Status**: Recommended for new projects as of 2026
**Endpoint**: `POST https://api.openai.com/v1/responses`

#### Key Improvements Over Chat Completions

1. **Better Performance**: 3% improvement on SWE-bench with reasoning models
2. **Cache Efficiency**: 40-80% improvement in cache utilization
3. **Stateful Context**: Use `store: true` to maintain state across turns
4. **Built-in Tools**: Native support for web search, file search, code interpreter
5. **Advanced Chain of Thought**: Support for passing reasoning between turns

#### Data Structure Changes

| Aspect | Chat Completions | Responses API |
|--------|-----------------|---------------|
| Input | Array of `messages` | Array of `items` |
| Output | Array of `choices` with `message` | Array of `items` with `output` |
| Tool Format | `tools` parameter | Different tool schema |
| State Management | Stateless | Optional stateful with `store: true` |

#### Migration Considerations

- Chat Completions remains supported indefinitely
- Assistants API deprecated (sunset: August 26, 2026)
- Responses API recommended for new projects
- Different tool calling schema requires code changes

---

## 2. Anthropic Claude API

### 2.1 Messages API

**Endpoint**: `POST https://api.anthropic.com/v1/messages`

#### Request Schema

```json
{
  "model": "claude-sonnet-4-5",
  "max_tokens": 1024,
  "system": "You are a helpful assistant.",
  "messages": [
    {
      "role": "user",
      "content": "Hello, Claude!"
    },
    {
      "role": "assistant",
      "content": "Hello! How can I help you today?"
    },
    {
      "role": "user",
      "content": "I need help with a coding task."
    }
  ],
  "temperature": 1.0,
  "top_p": 1.0,
  "top_k": 0,
  "metadata": {},
  "stop_sequences": [],
  "stream": false
}
```

#### Key Differences from OpenAI

| Feature | OpenAI | Anthropic |
|---------|--------|-----------|
| System prompt | Part of messages array | Top-level `system` parameter |
| Message limit | Unlimited | 100,000 messages max |
| Required params | `model`, `messages` | `model`, `messages`, `max_tokens` |
| Multimodal | Limited models | Native support in most models |
| Beta features | API versions | Beta headers (e.g., `anthropic-beta`) |

#### Response Schema

```json
{
  "id": "msg_abc123",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "I'd be happy to help you with your coding task. What are you working on?"
    }
  ],
  "model": "claude-sonnet-4-5",
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 25,
    "output_tokens": 18
  }
}
```

#### Content Types

Anthropic supports multiple content types in a single message:

```json
{
  "role": "user",
  "content": [
    {
      "type": "text",
      "text": "What's in this image?"
    },
    {
      "type": "image",
      "source": {
        "type": "base64",
        "media_type": "image/jpeg",
        "data": "/9j/4AAQSkZJRg..."
      }
    }
  ]
}
```

Supported image formats: `image/jpeg`, `image/png`, `image/gif`, `image/webp`

### 2.2 Beta Features and Headers

Anthropic uses beta headers to enable experimental features:

```
anthropic-beta: structured-outputs-2025-11-13
anthropic-beta: context-management-2025-06-27
```

Multiple beta features can be comma-separated:

```
anthropic-beta: structured-outputs-2025-11-13,context-management-2025-06-27
```

---

## 3. Tool/Function Calling Standards

### 3.1 OpenAI Function Calling

#### Tool Definition Schema

```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "City and state, e.g., 'San Francisco, CA'"
            },
            "unit": {
              "type": "string",
              "enum": ["celsius", "fahrenheit"],
              "description": "Temperature unit"
            }
          },
          "required": ["location"],
          "additionalProperties": false
        },
        "strict": true
      }
    }
  ],
  "tool_choice": "auto"
}
```

#### Tool Choice Options

- `"auto"`: Model decides whether to call tools (default)
- `"none"`: Model will not call any tools
- `"required"`: Model must call at least one tool
- `{"type": "function", "function": {"name": "get_weather"}}`: Force specific tool

#### Tool Call Response

```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_abc123",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"location\": \"San Francisco, CA\", \"unit\": \"fahrenheit\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ]
}
```

#### Providing Tool Results

```json
{
  "messages": [
    // ... previous messages
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [/* tool calls from above */]
    },
    {
      "role": "tool",
      "tool_call_id": "call_abc123",
      "content": "{\"temperature\": 72, \"condition\": \"sunny\"}"
    }
  ]
}
```

#### Structured Outputs (Strict Mode)

When `"strict": true` is set in function definition:

- **Guarantee**: Arguments will exactly match the JSON Schema
- **Supported models**: gpt-4o-mini, gpt-4o (recent versions)
- **Benefit**: Eliminates schema validation errors
- **Trade-off**: Slightly higher latency

### 3.2 Anthropic Tool Use

#### Tool Definition Schema

```json
{
  "tools": [
    {
      "name": "get_weather",
      "description": "Get the current weather in a given location",
      "input_schema": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "City and state, e.g., 'San Francisco, CA'"
          },
          "unit": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"],
            "description": "Temperature unit"
          }
        },
        "required": ["location"]
      },
      "strict": true
    }
  ]
}
```

#### Key Differences from OpenAI

| Aspect | OpenAI | Anthropic |
|--------|--------|-----------|
| Schema key | `parameters` | `input_schema` |
| Tool wrapper | `{"type": "function", "function": {...}}` | Direct tool object |
| Strict mode | `"strict": true` in function | `"strict": true` at tool level |
| Result format | `role: "tool"` message | `tool_result` content block |

#### Tool Use Response

```json
{
  "content": [
    {
      "type": "text",
      "text": "Let me check the weather for you."
    },
    {
      "type": "tool_use",
      "id": "toolu_abc123",
      "name": "get_weather",
      "input": {
        "location": "San Francisco, CA",
        "unit": "fahrenheit"
      }
    }
  ],
  "stop_reason": "tool_use"
}
```

#### Providing Tool Results

```json
{
  "messages": [
    // ... previous messages
    {
      "role": "user",
      "content": [
        {
          "type": "tool_result",
          "tool_use_id": "toolu_abc123",
          "content": "{\"temperature\": 72, \"condition\": \"sunny\"}"
        }
      ]
    }
  ]
}
```

#### Model Context Protocol (MCP)

Anthropic has introduced an open standard called Model Context Protocol:

- **Purpose**: Modular integration with external tools
- **Features**: Supports search engines, coding environments, calculators
- **Distinction**: Client tools vs. server tools
  - **Client tools**: Developer handles execution (like OpenAI)
  - **Server tools**: Claude handles execution (e.g., built-in web search)
- **Status**: Open standard for Claude Skills (released 2025)

### 3.3 Comparison and Best Practices

#### Similarities

- Both use JSON Schema for tool definitions
- Both support strict schema adherence
- Both support parallel tool calls
- Both use similar description patterns

#### Best Practices (Cross-Provider)

1. **Comprehensive descriptions**: Tool and parameter descriptions are critical for LLM understanding
2. **Explicit typing**: Always specify types, enums, and constraints
3. **Required fields**: Clearly mark required vs. optional parameters
4. **Error handling**: Document possible errors in tool execution
5. **Examples**: Provide example values in descriptions
6. **Naming**: Use clear, descriptive names (snake_case recommended)

#### Example: Cross-Compatible Tool Definition Strategy

```python
# Common tool definition
TOOL_SPEC = {
    "name": "search_database",
    "description": "Search the product database for items matching criteria",
    "parameters": {
        "query": {
            "type": "string",
            "description": "Search query string, e.g., 'red shoes size 10'"
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of results to return",
            "default": 10,
            "minimum": 1,
            "maximum": 100
        }
    },
    "required": ["query"]
}

# Convert to OpenAI format
def to_openai_format(spec):
    return {
        "type": "function",
        "function": {
            "name": spec["name"],
            "description": spec["description"],
            "parameters": {
                "type": "object",
                "properties": spec["parameters"],
                "required": spec["required"]
            }
        }
    }

# Convert to Anthropic format
def to_anthropic_format(spec):
    return {
        "name": spec["name"],
        "description": spec["description"],
        "input_schema": {
            "type": "object",
            "properties": spec["parameters"],
            "required": spec["required"]
        }
    }
```

---

## 4. Streaming Protocols

### 4.1 Server-Sent Events (SSE)

Both OpenAI and Anthropic use SSE for streaming responses.

#### SSE Protocol Basics

```
data: {"content": "Hello"}

data: {"content": " world"}

data: [DONE]
```

**Key characteristics**:
- Lines prefixed with `data:`
- Each event ends with double newline (`\n\n`)
- Stream ends with `data: [DONE]`
- MIME type: `text/event-stream`
- One-way communication (server to client)

#### OpenAI Streaming Format

**Request**:
```json
{
  "model": "gpt-4o",
  "messages": [...],
  "stream": true
}
```

**Response** (Chat Completions):
```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1677652288,"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1677652288,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1677652288,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":" there"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1677652288,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

**Key events** (Responses API):
- `response.created`
- `response.output_text.delta`
- `response.completed`
- `error`

#### Anthropic Streaming Format

**Request**:
```json
{
  "model": "claude-sonnet-4-5",
  "messages": [...],
  "stream": true
}
```

**Response**:
```
event: message_start
data: {"type": "message_start", "message": {"id": "msg_abc123", "type": "message", "role": "assistant", "content": [], "model": "claude-sonnet-4-5", "stop_reason": null, "stop_sequence": null, "usage": {"input_tokens": 25, "output_tokens": 0}}}

event: content_block_start
data: {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}

event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}

event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " there"}}

event: content_block_stop
data: {"type": "content_block_stop", "index": 0}

event: message_delta
data: {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": null}, "usage": {"output_tokens": 18}}

event: message_stop
data: {"type": "message_stop"}
```

**Event types**:
- `message_start`: Start of message
- `content_block_start`: Start of content block
- `content_block_delta`: Content chunk
- `content_block_stop`: End of content block
- `message_delta`: Message metadata update
- `message_stop`: End of message
- `ping`: Keep-alive event

#### Key Differences

| Aspect | OpenAI | Anthropic |
|--------|--------|-----------|
| Event types | Unnamed (data only) | Named events (event: type) |
| Delta structure | `choices[].delta` | `delta` at event level |
| Tool calls | Streamed in deltas | Separate content blocks |
| Completion marker | `[DONE]` | `message_stop` event |
| Mixed content | Text and tool calls mixed | Separate content blocks |

### 4.2 Implementation Best Practices

#### Client-Side Parsing

```python
import json
import requests

def stream_openai(api_key, messages):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4o",
        "messages": messages,
        "stream": True
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=data,
        stream=True
    )

    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                data_str = line[6:]  # Remove 'data: ' prefix
                if data_str == '[DONE]':
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk['choices'][0]['delta']
                    if 'content' in delta:
                        yield delta['content']
                except json.JSONDecodeError:
                    continue

def stream_anthropic(api_key, messages):
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2025-01-01",
        "Content-Type": "application/json"
    }
    data = {
        "model": "claude-sonnet-4-5",
        "max_tokens": 1024,
        "messages": messages,
        "stream": True
    }

    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json=data,
        stream=True
    )

    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                data_str = line[6:]
                try:
                    event = json.loads(data_str)
                    if event.get('type') == 'content_block_delta':
                        delta = event.get('delta', {})
                        if delta.get('type') == 'text_delta':
                            yield delta['text']
                except json.JSONDecodeError:
                    continue
```

#### Server-Side Implementation

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI()

async def generate_sse():
    """Generator function for SSE responses"""
    for i in range(10):
        # Simulate async work
        await asyncio.sleep(0.1)

        # Format as SSE
        data = {"content": f"chunk {i}"}
        yield f"data: {json.dumps(data)}\n\n"

    # Send completion marker
    yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def chat_completions(request: dict):
    if request.get("stream"):
        return StreamingResponse(
            generate_sse(),
            media_type="text/event-stream"
        )
    else:
        # Non-streaming response
        return {"choices": [...]}
```

#### Error Handling in Streams

```python
async def generate_sse_with_errors():
    try:
        for chunk in generate_chunks():
            yield f"data: {json.dumps(chunk)}\n\n"
    except Exception as e:
        # Send error event
        error_data = {
            "error": {
                "message": str(e),
                "type": "server_error",
                "code": "internal_error"
            }
        }
        yield f"data: {json.dumps(error_data)}\n\n"
        yield "data: [DONE]\n\n"
```

---

## 5. Authentication and Security

### 5.1 OpenAI Authentication

#### Bearer Token Format

```
Authorization: Bearer sk-proj-abc123...
```

#### Optional Headers

```
OpenAI-Organization: org-abc123
OpenAI-Project: proj-xyz789
```

#### Best Practices

1. **Never embed keys in code**: Use environment variables
   ```python
   import os
   api_key = os.environ.get("OPENAI_API_KEY")
   ```

2. **Use project-scoped keys**: Limit key permissions to specific projects

3. **Rotate keys regularly**: Generate new keys periodically

4. **Monitor usage**: Track API usage per key for anomaly detection

### 5.2 Anthropic Authentication

#### API Key Header

```
x-api-key: sk-ant-abc123...
```

#### Version Header (Required)

```
anthropic-version: 2025-01-01
```

#### Optional Beta Headers

```
anthropic-beta: structured-outputs-2025-11-13
```

#### Workspace Scoping

Each API key is scoped to a specific Workspace in the Anthropic Console.

### 5.3 Security Recommendations

1. **HTTPS Only**: Always use HTTPS for API requests
2. **Key Storage**: Use secure key management systems (AWS KMS, HashiCorp Vault)
3. **Rate Limiting**: Implement client-side rate limiting to avoid 429 errors
4. **Request Validation**: Validate all inputs before sending to API
5. **Error Masking**: Don't expose API keys in error messages or logs
6. **IP Whitelisting**: Use IP restrictions where available
7. **Audit Logging**: Log all API requests for security auditing

---

## 6. Error Handling

### 6.1 HTTP Status Codes

| Code | Name | Meaning | Action |
|------|------|---------|--------|
| 200 | OK | Request successful | Process response |
| 400 | Bad Request | Invalid request format | Fix request |
| 401 | Unauthorized | Invalid/missing API key | Check authentication |
| 403 | Forbidden | Access denied | Check permissions |
| 404 | Not Found | Endpoint doesn't exist | Check URL |
| 429 | Rate Limited | Too many requests | Implement backoff |
| 500 | Internal Server Error | Server-side error | Retry with backoff |
| 502 | Bad Gateway | Upstream error | Retry with backoff |
| 503 | Service Unavailable | Temporary outage | Retry later |

### 6.2 OpenAI Error Response Format

```json
{
  "error": {
    "message": "Invalid API key provided",
    "type": "invalid_request_error",
    "param": null,
    "code": "invalid_api_key"
  }
}
```

**Error types**:
- `invalid_request_error`: Problem with request
- `authentication_error`: Authentication failed
- `permission_error`: Insufficient permissions
- `rate_limit_error`: Rate limit exceeded
- `api_error`: Server-side error

### 6.3 Anthropic Error Response Format

```json
{
  "type": "error",
  "error": {
    "type": "authentication_error",
    "message": "Invalid API key"
  }
}
```

**Error types**:
- `authentication_error`: Invalid API key
- `invalid_request_error`: Malformed request
- `permission_error`: Insufficient permissions
- `rate_limit_error`: Rate limit exceeded
- `api_error`: Internal server error

### 6.4 Retry Strategies

#### Exponential Backoff

```python
import time
import random

def exponential_backoff(func, max_retries=5):
    for attempt in range(max_retries):
        try:
            return func()
        except RateLimitError:
            if attempt == max_retries - 1:
                raise

            # Exponential backoff with jitter
            wait_time = (2 ** attempt) + random.random()
            time.sleep(wait_time)
        except ServerError as e:
            if attempt == max_retries - 1:
                raise

            # Check for Retry-After header
            retry_after = e.response.headers.get('Retry-After')
            if retry_after:
                time.sleep(int(retry_after))
            else:
                time.sleep(2 ** attempt)
```

#### Which Errors to Retry

**Retry** (transient errors):
- 429 (Rate Limit)
- 500 (Internal Server Error)
- 502 (Bad Gateway)
- 503 (Service Unavailable)

**Don't Retry** (permanent errors):
- 400 (Bad Request)
- 401 (Unauthorized)
- 403 (Forbidden)
- 404 (Not Found)

---

## 7. API Versioning and Compatibility

### 7.1 Versioning Strategies

#### URI-Based Versioning (Most Common)

```
https://api.openai.com/v1/chat/completions
https://api.anthropic.com/v1/messages
```

**Pros**:
- Clear and explicit
- Easy to route
- Backward compatible

**Cons**:
- URLs change between versions
- Can lead to endpoint proliferation

#### Header-Based Versioning

```
anthropic-version: 2025-01-01
```

**Pros**:
- URLs remain stable
- Fine-grained control
- Can combine with URI versioning

**Cons**:
- Less visible
- Requires header management

### 7.2 OpenAI Versioning Approach

1. **Model Versioning**: Pin to specific model versions
   - `gpt-4o-2024-08-06` (pinned)
   - `gpt-4o` (latest, updates automatically)

2. **API Versioning**: Major versions in URI (`/v1/`)

3. **Deprecation Policy**:
   - 6-12 months notice
   - Clear migration guides
   - Gradual sunset
   - Example: Assistants API (deprecated Aug 2025, sunset Aug 2026)

4. **Changelog**: Maintained at `https://platform.openai.com/docs/changelog`

### 7.3 Anthropic Versioning Approach

1. **Date-Based Versioning**: Required header with date
   ```
   anthropic-version: 2025-01-01
   ```

2. **Beta Features**: Opt-in via beta headers
   ```
   anthropic-beta: structured-outputs-2025-11-13
   ```

3. **Backward Compatibility**: Maintains compatibility within version date

### 7.4 Best Practices for API Providers

1. **Semantic Versioning**: Use clear version numbers (v1, v2, etc.)

2. **Backward Compatibility**:
   - Add new fields, don't modify existing ones
   - Make new fields optional
   - Deprecate gradually, never remove abruptly

3. **Deprecation Timeline**:
   - Announce 6-12 months in advance
   - Provide migration guides
   - Support parallel versions during transition
   - Send runtime warnings for deprecated features

4. **Documentation**:
   - Detailed changelog
   - Version comparison guides
   - Migration examples
   - Clear sunset dates

5. **Testing**:
   - Automated compatibility tests
   - Shadow traffic validation
   - Gradual rollout with feature flags
   - Monitor adoption metrics

---

## 8. Emerging Standards

### 8.1 LiteLLM - Unified Interface

**Purpose**: Call 100+ LLM APIs using OpenAI format

**Supported Providers**:
- OpenAI
- Anthropic
- Azure OpenAI
- AWS Bedrock
- Google VertexAI
- Cohere
- And many more

**Example**:
```python
from litellm import completion

# Unified interface for all providers
response = completion(
    model="claude-sonnet-4-5",  # Anthropic
    messages=[{"role": "user", "content": "Hello"}]
)

response = completion(
    model="gpt-4o",  # OpenAI
    messages=[{"role": "user", "content": "Hello"}]
)
```

### 8.2 Model Context Protocol (MCP)

**Developer**: Anthropic
**Status**: Open Standard (released 2025)
**Purpose**: Standardize agent-to-tool integration

**Key Features**:
- Modular tool integration
- Client vs. server tool distinction
- Skill sharing and discovery
- Claude Skills ecosystem

**Use Cases**:
- Search engine integration
- Code execution environments
- Calculator and math tools
- Database queries
- API integrations

### 8.3 Other Emerging Standards

#### 1. OpenAPI for LLM Function Calling

**Concept**: Convert OpenAPI specs to LLM tool definitions

**Tools**:
- `openapi-llm`: Convert OpenAPI to function definitions
- LLM-OpenAPI-minifier: Optimize specs for token efficiency

**Benefits**:
- Automatic API discovery
- Standardized documentation
- Reduced hallucinations

#### 2. agents.json (Wildcard)

**Purpose**: Schema for describing agents built on OpenAPI

**Features**:
- Agent capability description
- Tool/API mappings
- Agent discovery

#### 3. /llms.txt

**Purpose**: Make documentation accessible to LLMs

**Adopted by**: Cursor, Perplexity, Coinbase, Anthropic (using Mintlify)

**Concept**: Standard format for LLM-readable documentation

---

## 9. OpenAPI/Swagger Best Practices

### 9.1 Why OpenAPI for LLM APIs?

OpenAPI (Swagger) specifications provide:
- Standardized API documentation
- Automatic client generation
- LLM function calling support
- Clear contracts between services

### 9.2 Best Practices for LLM-Friendly OpenAPI Specs

#### 1. Comprehensive Descriptions

```yaml
paths:
  /chat/completions:
    post:
      summary: Create chat completion
      description: |
        Creates a model response for the given chat conversation.
        This endpoint supports streaming via Server-Sent Events when
        the 'stream' parameter is set to true.
      operationId: createChatCompletion
      parameters:
        - name: model
          in: body
          required: true
          description: |
            ID of the model to use. See the model endpoint compatibility
            table for details on which models work with the Chat API.
            Examples: 'gpt-4o', 'gpt-5.2'
          schema:
            type: string
```

#### 2. Detailed Parameter Documentation

```yaml
parameters:
  - name: temperature
    in: body
    schema:
      type: number
      minimum: 0
      maximum: 2
      default: 1
      example: 0.7
    description: |
      Controls randomness in the output. Lower values like 0.2 make
      the output more focused and deterministic. Higher values like
      0.8 make it more random and creative. We generally recommend
      altering this or top_p but not both.
```

#### 3. Error Response Documentation

```yaml
responses:
  '400':
    description: Bad Request - Invalid parameters
    content:
      application/json:
        schema:
          $ref: '#/components/schemas/Error'
        example:
          error:
            message: "Invalid value for 'temperature': must be between 0 and 2"
            type: "invalid_request_error"
            param: "temperature"
            code: "invalid_value"
  '401':
    description: Unauthorized - Invalid or missing API key
    content:
      application/json:
        schema:
          $ref: '#/components/schemas/Error'
        example:
          error:
            message: "Invalid API key provided"
            type: "authentication_error"
            code: "invalid_api_key"
```

#### 4. Reusable Components

```yaml
components:
  schemas:
    Message:
      type: object
      required:
        - role
        - content
      properties:
        role:
          type: string
          enum: [system, user, assistant, tool]
          description: The role of the message author
        content:
          type: string
          description: The content of the message
        name:
          type: string
          description: Optional name for the message author

    Error:
      type: object
      required:
        - error
      properties:
        error:
          type: object
          required:
            - message
            - type
          properties:
            message:
              type: string
            type:
              type: string
            param:
              type: string
              nullable: true
            code:
              type: string
              nullable: true

  securitySchemes:
    BearerAuth:
      type: http
      scheme: bearer
      description: Use your API key as a Bearer token
```

#### 5. Token Efficiency

**Problem**: OpenAPI specs can be very large (waste tokens)

**Solutions**:
- Remove unnecessary descriptions for internal endpoints
- Use `$ref` to avoid duplication
- Consider minified versions for LLM consumption
- Focus on public API documentation

#### 6. Example Requests/Responses

```yaml
requestBody:
  required: true
  content:
    application/json:
      schema:
        $ref: '#/components/schemas/ChatCompletionRequest'
      examples:
        simple:
          summary: Simple chat request
          value:
            model: gpt-4o
            messages:
              - role: user
                content: "Hello, how are you?"
        with_tools:
          summary: Request with function calling
          value:
            model: gpt-4o
            messages:
              - role: user
                content: "What's the weather in SF?"
            tools:
              - type: function
                function:
                  name: get_weather
                  description: Get current weather
                  parameters:
                    type: object
                    properties:
                      location:
                        type: string
```

### 9.3 OpenAPI Version Recommendations

- **Use OpenAPI 3.1.0**: Latest stable version
- **Support JSON Schema**: For tool definitions
- **Include security schemes**: Document authentication
- **Provide examples**: For all major operations

---

## 10. Implementation Recommendations for Elpis

Based on this research, here are specific recommendations for implementing the Elpis Phase 2 API.

### 10.1 API Design Decisions

#### 1. Primary Compatibility Target

**Recommendation**: OpenAI Chat Completions API (v1)

**Rationale**:
- Industry de facto standard
- Widest ecosystem support
- Most client libraries available
- Clear, proven design

**Implementation**:
```
POST /v1/chat/completions
```

Support core OpenAI parameters:
- `model`
- `messages` (array with role/content)
- `temperature`, `max_tokens`, `top_p`
- `stream` (boolean)
- `tools` (for function calling)
- `tool_choice`

#### 2. Future-Proofing

**Consideration**: Monitor OpenAI's Responses API

While Chat Completions should be the primary target, keep architecture flexible enough to support Responses API patterns:
- Item-based structure (vs. message-based)
- Stateful context (`store: true`)
- Built-in tools

**Recommendation**: Design internal architecture to be format-agnostic, with translation layers.

### 10.2 Tool/Function Calling

#### Support Both Formats

Create a unified internal representation that can translate to both formats:

```python
class ToolDefinition:
    """Internal tool representation"""
    name: str
    description: str
    parameters: dict  # JSON Schema
    required: list[str]
    strict: bool = False

    def to_openai_format(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required,
                    "additionalProperties": False
                },
                "strict": self.strict
            }
        }

    def to_anthropic_format(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.parameters,
                "required": self.required
            },
            "strict": self.strict
        }
```

### 10.3 Versioning Strategy

**Recommendation**: Hybrid approach

1. **URI Versioning** for major versions:
   ```
   /v1/chat/completions
   /v2/chat/completions (future)
   ```

2. **Header-Based** for minor versions/features:
   ```
   Elpis-Version: 2026-01-12
   Elpis-Beta: extended-context,multi-agent
   ```

3. **Deprecation Policy**:
   - 6 months minimum notice
   - Clear migration documentation
   - Support parallel versions for 3 months
   - Runtime warnings for deprecated features

### 10.4 Authentication

**Recommendation**: Bearer token with optional organization scoping

```
Authorization: Bearer elpis_sk_...
Elpis-Organization: org-abc123 (optional)
Elpis-Project: proj-xyz789 (optional)
```

**Security Features**:
- API key scoping (project/organization level)
- Rate limiting per key
- Usage tracking and quotas
- Key rotation support

### 10.5 Streaming

**Recommendation**: SSE with OpenAI-compatible format

**Implementation**:
```python
async def stream_completion(request):
    # Set SSE headers
    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no"  # Disable nginx buffering
    }

    async def generate():
        try:
            async for chunk in llm_generate(request):
                data = {
                    "id": request.id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": chunk},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(data)}\n\n"

            # Final chunk
            yield f"data: {json.dumps({...,'finish_reason':'stop'})}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            error_data = {
                "error": {
                    "message": str(e),
                    "type": "server_error"
                }
            }
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(generate(), headers=headers)
```

### 10.6 Error Handling

**Recommendation**: OpenAI-compatible error format

```json
{
  "error": {
    "message": "Human-readable error message",
    "type": "error_type",
    "param": "parameter_name",
    "code": "error_code"
  }
}
```

**Error Types**:
- `invalid_request_error`: Malformed request
- `authentication_error`: Auth failure
- `permission_error`: Insufficient permissions
- `rate_limit_error`: Rate limit exceeded
- `context_length_exceeded`: Input too long
- `server_error`: Internal error

**Retry Headers**:
```
Retry-After: 60
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1677652288
```

### 10.7 Request/Response Schemas

#### Chat Completion Request

```typescript
interface ChatCompletionRequest {
  // Required
  model: string;
  messages: Message[];

  // Optional
  temperature?: number;  // 0-2, default 1
  max_tokens?: number;
  top_p?: number;  // 0-1, default 1
  stream?: boolean;  // default false
  stop?: string | string[];

  // Function calling
  tools?: Tool[];
  tool_choice?: "auto" | "none" | "required" | {type: "function", function: {name: string}};

  // Other
  user?: string;  // End-user ID for tracking
  response_format?: {type: "text" | "json_object"};
}

interface Message {
  role: "system" | "user" | "assistant" | "tool";
  content: string | ContentPart[];
  name?: string;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
}

interface Tool {
  type: "function";
  function: {
    name: string;
    description: string;
    parameters: object;  // JSON Schema
    strict?: boolean;
  };
}
```

#### Chat Completion Response

```typescript
interface ChatCompletionResponse {
  id: string;
  object: "chat.completion";
  created: number;  // Unix timestamp
  model: string;
  choices: Choice[];
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

interface Choice {
  index: number;
  message: Message;
  finish_reason: "stop" | "length" | "tool_calls" | "content_filter";
}
```

### 10.8 OpenAPI Specification

**Recommendation**: Provide comprehensive OpenAPI 3.1.0 spec

**Location**: `/openapi.json` and `/openapi.yaml`

**Benefits**:
- Automatic client generation
- Interactive documentation (Swagger UI)
- LLM function calling integration
- Clear API contracts

**Example Structure**:
```yaml
openapi: 3.1.0
info:
  title: Elpis API
  version: 1.0.0
  description: |
    Elpis provides a powerful LLM API compatible with OpenAI's
    Chat Completions format, with extended features for context
    management and multi-agent orchestration.

servers:
  - url: https://api.elpis.ai/v1
    description: Production server

security:
  - BearerAuth: []

paths:
  /chat/completions:
    post:
      summary: Create chat completion
      operationId: createChatCompletion
      # ... full specification
```

### 10.9 Additional Elpis-Specific Features

While maintaining OpenAI compatibility, consider these extensions:

#### 1. Context Management Extensions

```json
{
  "model": "elpis-chat",
  "messages": [...],
  "elpis_config": {
    "context_strategy": "adaptive",  // or "sliding", "summary"
    "max_context_tokens": 100000,
    "importance_threshold": 0.7
  }
}
```

#### 2. Multi-Agent Extensions

```json
{
  "model": "elpis-agents",
  "messages": [...],
  "elpis_config": {
    "agents": ["coder", "tester", "reviewer"],
    "coordination_mode": "sequential"  // or "parallel"
  }
}
```

#### 3. Memory Extensions

```json
{
  "model": "elpis-chat",
  "messages": [...],
  "elpis_config": {
    "memory_enabled": true,
    "memory_scope": "user",  // or "session", "global"
    "memory_retrieval": "semantic"  // or "recency", "importance"
  }
}
```

**Implementation Strategy**:
- Use `elpis_config` field for extensions
- Ignore unknown fields (forward compatibility)
- Document extensions clearly
- Make all extensions optional

### 10.10 Testing and Validation

1. **Compatibility Testing**:
   - Test with OpenAI client libraries
   - Validate against OpenAPI spec
   - Cross-provider comparison tests

2. **Load Testing**:
   - Streaming performance
   - Rate limiting accuracy
   - Error handling under load

3. **Integration Testing**:
   - Tool calling round-trips
   - Context management
   - Multi-turn conversations

---

## References and Sources

### OpenAI Documentation
- [Chat Completions API Reference](https://platform.openai.com/docs/api-reference/chat)
- [Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
- [Migrate to Responses API](https://platform.openai.com/docs/guides/migrate-to-responses)
- [OpenAI API Error Codes](https://platform.openai.com/docs/guides/error-codes)
- [Structured Outputs Guide](https://platform.openai.com/docs/guides/structured-outputs)
- [Streaming API Responses](https://platform.openai.com/docs/guides/streaming-responses)
- [OpenAI API Changelog](https://platform.openai.com/docs/changelog)

### Anthropic Documentation
- [Messages API Reference](https://docs.claude.com/en/api/messages)
- [Using the Messages API](https://platform.claude.com/docs/en/build-with-claude/working-with-messages)
- [How to Implement Tool Use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use)
- [Structured Outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)
- [Anthropic API Overview](https://platform.claude.com/docs/en/api/overview)

### API Comparison and Standards
- [How LLM APIs use the OpenAPI spec for function calling](https://medium.com/percolation-labs/how-llm-apis-use-the-openapi-spec-for-function-calling-f37d76e0fef3)
- [Anthropic vs OpenAI: Which Models Fit Your Product Better?](https://www.ninetwothree.co/blog/anthropic-vs-openai)
- [Structured Output Comparison across popular LLM providers](https://medium.com/@rosgluk/structured-output-comparison-across-popular-llm-providers-openai-gemini-anthropic-mistral-and-1a5d42fa612a)
- [Demystifying OpenAI Function Calling vs Anthropic's MCP](https://evgeniisaurov.medium.com/demystifying-openai-function-calling-vs-anthropics-model-context-protocol-mcp-b5e4c7b59ac2)
- [Comparing 7 AI Agent-to-API Standards](https://nordicapis.com/comparing-7-ai-agent-to-api-standards/)

### Versioning and Best Practices
- [API Versioning and Backward Compatibility Best Practices](https://zuplo.com/learning-center/api-versioning-backward-compatibility-best-practices)
- [API Versioning Strategies: Best Practices Guide](https://daily.dev/blog/api-versioning-strategies-best-practices-guide)
- [Server Sent Events in OpenAPI best practices](https://www.speakeasy.com/openapi/content/server-sent-events)

### Tools and Libraries
- [LiteLLM - Unified LLM API Interface](https://github.com/BerriAI/litellm)
- [OpenAPI-LLM - Convert OpenAPI to LLM Tools](https://github.com/vblagoje/openapi-llm)
- [Anthropic OpenAPI Spec (Unofficial)](https://github.com/laszukdawid/anthropic-openapi-spec)

### Additional Resources
- [OpenAI's Responses API compared to Chat Completions](https://platform.openai.com/docs/guides/responses-vs-chat-completions)
- [Understanding OpenAI's New Responses API Streaming Model](https://madhub081011.medium.com/understanding-openais-new-responses-api-streaming-model-a6d932e481e8)
- [Top OpenAI API HTTP Errors: Root Causes and Fixes](https://wizardstool.com/openai-api-http-errors-guide/)
- [Anthropic Launches Skills Open Standard for Claude](https://aibusiness.com/foundation-models/anthropic-launches-skills-open-standard-claude)

---

## Conclusion

The LLM API landscape in 2026 is characterized by:

1. **OpenAI's dominance as the de facto standard** - Most providers offer OpenAI-compatible interfaces
2. **Evolution toward more sophisticated APIs** - Responses API, MCP, structured outputs
3. **Convergence in tool calling** - Despite format differences, capabilities are similar
4. **Emphasis on developer experience** - Comprehensive docs, SDKs, compatibility
5. **Long-term stability** - Careful deprecation policies, backward compatibility

For Elpis Phase 2, the recommendation is to:
- **Maintain OpenAI Chat Completions compatibility** for broad ecosystem support
- **Add Elpis-specific extensions** for context management and multi-agent features
- **Design with flexibility** to support future standards (Responses API, MCP)
- **Provide excellent documentation** including OpenAPI specs
- **Implement robust error handling and retry logic**

This approach balances immediate compatibility with long-term extensibility, positioning Elpis as a powerful yet familiar API for LLM interactions.

---

**End of Report**
