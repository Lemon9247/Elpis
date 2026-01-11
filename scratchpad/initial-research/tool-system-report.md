# Tool System Research Report - Elpis Project

**Compiled by:** Tool-System-Agent
**Date:** January 2026
**Status:** Complete Research

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Tool Definition Schemas](#tool-definition-schemas)
3. [Function Calling Formats](#function-calling-formats)
4. [Safe Execution Engine Design](#safe-execution-engine-design)
5. [Implementation Patterns](#implementation-patterns)
6. [Code Examples](#code-examples)
7. [Best Practices](#best-practices)
8. [Recommendations for Elpis](#recommendations-for-elpis)

---

## Executive Summary

After comprehensive research into tool execution patterns for LLM-based agents, we have identified industry-standard approaches, safety mechanisms, and implementation patterns suitable for the Elpis emotional coding agent. The key findings are:

1. **Tool Definition**: OpenAI-compatible JSON schema format has become the standard
2. **Function Calling**: Multiple mature approaches exist; llama.cpp now has native OpenAI-compatible support
3. **Safety**: Multi-layer approach combining OS-level sandboxing, whitelisting, validation, and monitoring
4. **Implementation**: Pydantic for validation, subprocess with timeout/isolation for command execution
5. **Error Handling**: ReAct pattern (Reason-Act-Observe) with retry strategies and graceful degradation

---

## Tool Definition Schemas

### Standard Format: JSON Schema (OpenAI Compatible)

The industry standard for defining tools is based on OpenAI's function calling specification, which has been adopted by vLLM, llama.cpp, Google Vertex AI, and other platforms.

#### Basic Structure

```json
{
  "name": "read_file",
  "description": "Read the complete contents of a file from the filesystem",
  "parameters": {
    "type": "object",
    "properties": {
      "file_path": {
        "type": "string",
        "description": "The absolute path to the file to read"
      },
      "max_lines": {
        "type": "integer",
        "description": "Maximum number of lines to read (optional)",
        "default": 2000
      }
    },
    "required": ["file_path"]
  }
}
```

#### Key Schema Constraints

- **Type Declaration**: All parameters must have explicit types (string, integer, number, boolean, array, object)
- **Description Requirements**: Every parameter must have a clear description for the LLM to understand its purpose
- **Required Array**: Specify which parameters are mandatory
- **Defaults**: Optional parameters can have default values
- **Enums**: Use enum arrays to restrict parameter values to specific choices
- **Nested Objects**: Complex tools can use nested object schemas
- **Strict Mode**: OpenAI's "strict: true" enforces exact schema compliance

### Schema Validation Considerations

When implementing strict schema validation:
- All fields in required array must be present in properties
- Root objects cannot use anyOf/oneOf patterns
- No pattern-only constraints without explicit types
- Array items must have explicit type definitions
- Recursive objects require careful termination conditions

### Pydantic Integration for Python

Using Pydantic models automatically generates correct JSON schemas:

```python
from pydantic import BaseModel, Field
from typing import Optional

class ReadFileInput(BaseModel):
    file_path: str = Field(description="Absolute path to file")
    max_lines: Optional[int] = Field(
        default=2000,
        description="Maximum lines to read"
    )

# Pydantic automatically generates:
# - JSON schema from the model
# - Runtime validation
# - Error messages for invalid inputs
```

---

## Function Calling Formats

### Format 1: OpenAI Chat Completion Format

The standard format used by OpenAI, llama.cpp, and other platforms:

```python
# Tool calls are returned as part of chat completion
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
              "name": "read_file",
              "arguments": "{\"file_path\": \"/path/to/file.py\"}"
            }
          }
        ]
      }
    }
  ]
}
```

**Advantages:**
- Industry standard across multiple platforms
- Native llama.cpp support
- Compatible with OpenAI client libraries
- Supports parallel tool calls
- Clear tool ID for result mapping

### Format 2: Python List Representation

Some newer frameworks use Python lists for improved readability:

```python
[
  {
    "tool": "read_file",
    "args": {"file_path": "/path/to/file.py"}
  }
]
```

**Advantages:**
- More readable for Python developers
- Easier debugging and inspection
- Natural for Python-based agents
- Removes JSON serialization ambiguity

**Trade-offs:**
- Less standard across platforms
- Requires custom parsing

### Format 3: Structured Output with Final Tool

Pydantic AI pattern where the LLM calls a special "final_result" tool:

```python
{
  "tool_calls": [
    {
      "function": {
        "name": "final_result",
        "arguments": {
          "result": "The parsed JSON structure"
        }
      }
    }
  ]
}
```

Used when you want guaranteed structured outputs conforming to a Pydantic model.

### Recommendation for Elpis

**Use OpenAI Chat Completion Format** for these reasons:
- Native support in llama.cpp (both Llama 3.1 and Mistral v0.3)
- Compatible with llama-cpp-python library
- Industry standard makes future integration easier
- Supports parallel tool calls (future feature)
- Clear separation of concerns

---

## Safe Execution Engine Design

### Architecture Overview

A safe execution engine must implement multiple layers of protection:

```
┌─────────────────────────────────────────────────────┐
│ LLM Generated Tool Call                              │
└──────────────┬──────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────┐
│ Layer 1: Input Validation & Sanitization            │
│ - Parse JSON/arguments                              │
│ - Validate against schema (Pydantic)                │
│ - Check parameter types and ranges                  │
└──────────────┬──────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────┐
│ Layer 2: Tool Authorization                         │
│ - Check if tool is in whitelist                     │
│ - Verify user has permission                        │
│ - Check tool availability/status                    │
└──────────────┬──────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────┐
│ Layer 3: Argument Sanitization                      │
│ - Path traversal checks (for file tools)            │
│ - Command injection prevention                      │
│ - Deny list pattern matching                        │
│ - Resource limit checks (file size, timeout)        │
└──────────────┬──────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────┐
│ Layer 4: Sandboxed Execution                        │
│ - OS-level isolation (bubblewrap, seatbelt)         │
│ - Filesystem restrictions                           │
│ - Network isolation                                 │
│ - Resource limits (CPU, memory, time)               │
└──────────────┬──────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────┐
│ Layer 5: Monitoring & Logging                       │
│ - Capture stdout/stderr                             │
│ - Monitor execution time                            │
│ - Log all tool calls and results                    │
│ - Alert on suspicious patterns                      │
└──────────────┬──────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────┐
│ Layer 6: Error Handling & Recovery                  │
│ - Graceful error messages                           │
│ - Retry with exponential backoff                    │
│ - Fallback strategies                               │
│ - LLM re-prompt with error context                  │
└─────────────────────────────────────────────────────┘
```

### Layer 1: Input Validation & Sanitization

```python
from pydantic import BaseModel, Field, field_validator
from typing import Optional

class ToolInput(BaseModel):
    """Base class for all tool inputs"""

    class Config:
        # Validate on assignment for immediate feedback
        validate_assignment = True

    @field_validator('*', mode='before')
    @classmethod
    def strip_whitespace(cls, v):
        """Remove leading/trailing whitespace from strings"""
        if isinstance(v, str):
            return v.strip()
        return v

class ReadFileInput(ToolInput):
    file_path: str = Field(
        description="Absolute path to file",
        min_length=1,
        max_length=4096
    )
    max_lines: Optional[int] = Field(
        default=2000,
        ge=1,
        le=100000,
        description="Maximum lines to read"
    )

    @field_validator('file_path')
    @classmethod
    def validate_path(cls, v):
        """Ensure path is absolute and doesn't contain null bytes"""
        if not v.startswith('/'):
            raise ValueError("Must be absolute path")
        if '\x00' in v:
            raise ValueError("Null bytes not allowed")
        return v
```

### Layer 2: Tool Authorization

```python
class ToolAuthorizer:
    """Manage which tools are available and who can use them"""

    def __init__(self):
        # Whitelist of available tools
        self.available_tools = {
            'read_file',
            'write_file',
            'execute_bash',
            'search_codebase',
            'list_directory'
        }

        # Dangerous tools that require approval
        self.dangerous_tools = {
            'execute_bash',  # Can modify system
            'write_file',    # Can destroy files
        }

        # Per-tool rate limits (calls per minute)
        self.rate_limits = {
            'execute_bash': 10,
            'read_file': 100,
        }

    def authorize(self, tool_name: str, user_id: str) -> bool:
        """Check if user can execute this tool"""
        if tool_name not in self.available_tools:
            raise ToolNotFound(f"Tool '{tool_name}' not available")

        if tool_name in self.dangerous_tools:
            # Check rate limit and user permissions
            if self.get_call_count(user_id, tool_name) >= self.rate_limits[tool_name]:
                raise RateLimitExceeded(f"Tool '{tool_name}' rate limit exceeded")

        return True
```

### Layer 3: Argument Sanitization

```python
import os
import re
from pathlib import Path

class ArgumentSanitizer:
    """Sanitize tool arguments before execution"""

    ALLOWED_WORKSPACE = Path("/home/lemoneater/Devel/elpis/workspace").resolve()

    # Deny list patterns for command injection
    DANGEROUS_PATTERNS = [
        r'[;&|`$\(].*\n',      # Newlines in commands
        r'\$\{.*\}',            # Variable expansion
        r'\$\(.*\)',            # Command substitution
        r'`.*`',                # Backtick execution
        r'&&|\|\|',             # Command chaining
    ]

    @classmethod
    def sanitize_path(cls, path: str) -> Path:
        """
        Validate and resolve file paths.
        Prevents path traversal attacks.
        """
        try:
            # Resolve to absolute path
            resolved = Path(path).resolve()

            # Check it's within allowed workspace
            if not str(resolved).startswith(str(cls.ALLOWED_WORKSPACE)):
                raise ValueError(f"Path {path} is outside workspace")

            # Check for suspicious patterns
            if '..' in path:
                raise ValueError("Path traversal (..) not allowed")

            return resolved

        except Exception as e:
            raise ValueError(f"Invalid path: {e}")

    @classmethod
    def sanitize_bash_command(cls, command: str) -> str:
        """
        Validate bash commands for injection risks.
        Returns safe command or raises error.
        """
        # Check for dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, command):
                raise ValueError(f"Dangerous pattern detected: {pattern}")

        # Limit command length
        if len(command) > 10000:
            raise ValueError("Command too long")

        return command
```

### Layer 4: Sandboxed Execution

```python
import subprocess
import signal
import os
from contextlib import contextmanager

class SandboxExecutor:
    """Execute commands with OS-level isolation"""

    def __init__(self, workspace_dir: str, timeout_seconds: int = 30):
        self.workspace_dir = workspace_dir
        self.timeout_seconds = timeout_seconds

    def execute_bash(self, command: str) -> dict:
        """
        Execute bash command in isolated sandbox.
        Returns: {
            'success': bool,
            'stdout': str,
            'stderr': str,
            'returncode': int,
            'timeout': bool
        }
        """
        try:
            # Use subprocess with explicit isolation
            process = subprocess.Popen(
                ['bash', '-c', command],
                cwd=self.workspace_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                # Isolate from parent environment
                env={
                    'PATH': '/usr/local/bin:/usr/bin:/bin',
                    'HOME': self.workspace_dir,
                    # Disable dangerous commands
                    'HISTFILE': '/dev/null',
                },
                preexec_fn=os.setsid,  # Create new process group
            )

            # Run with timeout
            try:
                stdout, stderr = process.communicate(
                    timeout=self.timeout_seconds
                )
                return {
                    'success': process.returncode == 0,
                    'stdout': stdout.decode('utf-8', errors='replace'),
                    'stderr': stderr.decode('utf-8', errors='replace'),
                    'returncode': process.returncode,
                    'timeout': False
                }

            except subprocess.TimeoutExpired:
                # Kill entire process group
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                return {
                    'success': False,
                    'stdout': '',
                    'stderr': 'Command timeout',
                    'returncode': -1,
                    'timeout': True
                }

        except Exception as e:
            return {
                'success': False,
                'stdout': '',
                'stderr': str(e),
                'returncode': -1,
                'timeout': False
            }
```

### Layer 5: Monitoring & Logging

```python
import json
import logging
from datetime import datetime
from typing import Any

class ToolCallLogger:
    """Log all tool calls for audit trail"""

    def __init__(self, log_file: str):
        self.logger = logging.getLogger('tool_system')
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def log_tool_call(self,
                      tool_name: str,
                      args: dict,
                      result: Any,
                      duration_ms: float,
                      success: bool):
        """Log a single tool call with context"""

        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'tool_name': tool_name,
            'success': success,
            'duration_ms': duration_ms,
            'args_keys': list(args.keys()),  # Don't log sensitive args
        }

        if success:
            self.logger.info(json.dumps(log_entry))
        else:
            log_entry['error'] = str(result)
            self.logger.error(json.dumps(log_entry))
```

### Layer 6: Error Handling & Recovery

```python
from enum import Enum
import time

class RetryStrategy(Enum):
    NONE = 0
    SIMPLE = 1
    EXPONENTIAL_BACKOFF = 2
    ADAPTIVE = 3

class ToolExecutor:
    """Execute tools with error handling and recovery"""

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    def execute(self, tool_name: str, args: dict) -> dict:
        """Execute tool with automatic retry on transient errors"""

        last_error = None

        for attempt in range(self.max_retries):
            try:
                # Check if tool is available
                if not self.is_tool_available(tool_name):
                    return self.graceful_degradation(tool_name, args)

                # Execute tool
                result = self._execute_tool(tool_name, args)

                if result['success']:
                    return result
                else:
                    last_error = result.get('error', 'Unknown error')

            except TransientError as e:
                # Retry on transient errors with backoff
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    time.sleep(wait_time)
                    continue

            except PermanentError as e:
                # Don't retry on permanent errors
                last_error = e
                break

        # All retries failed - return graceful error
        return {
            'success': False,
            'error': f"Tool execution failed: {last_error}",
            'suggestions': self.get_recovery_suggestions(tool_name, last_error)
        }

    def get_recovery_suggestions(self, tool_name: str, error: Exception) -> list:
        """Suggest recovery actions based on error type"""

        if isinstance(error, FileNotFoundError):
            return [
                "Check that file path exists",
                "Try listing directory contents first",
                "Verify absolute path is correct"
            ]
        elif isinstance(error, PermissionError):
            return [
                "Check file permissions",
                "Ensure you're in the correct workspace",
                "Try a different tool or file"
            ]
        elif isinstance(error, TimeoutError):
            return [
                "Command took too long",
                "Try breaking task into smaller steps",
                "Check for infinite loops or hanging processes"
            ]
        else:
            return ["Check tool documentation", "Try with simpler inputs"]
```

---

## Implementation Patterns

### Pattern 1: read_file Tool

```python
from pathlib import Path
from typing import Optional
import re

class ReadFileTool:
    """Read file contents with safety checks"""

    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

    def __init__(self, workspace_dir: str):
        self.workspace_dir = Path(workspace_dir).resolve()

    def read(self,
             file_path: str,
             max_lines: Optional[int] = None,
             search_pattern: Optional[str] = None) -> dict:
        """
        Read file with optional filtering.

        Args:
            file_path: Absolute path to file
            max_lines: Maximum lines to return
            search_pattern: Regex to filter lines

        Returns:
            {'success': bool, 'content': str, 'error': str}
        """
        try:
            # Sanitize path
            path = ArgumentSanitizer.sanitize_path(file_path)

            # Check file exists and is readable
            if not path.is_file():
                return {
                    'success': False,
                    'error': f"Not a file: {file_path}"
                }

            # Check file size
            if path.stat().st_size > self.MAX_FILE_SIZE:
                return {
                    'success': False,
                    'error': f"File too large: {path.stat().st_size} bytes"
                }

            # Read file
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()

            # Apply filters
            if search_pattern:
                regex = re.compile(search_pattern)
                lines = [l for l in lines if regex.search(l)]

            if max_lines:
                lines = lines[:max_lines]

            content = ''.join(lines)

            return {
                'success': True,
                'content': content,
                'line_count': len(lines),
                'file_path': str(path)
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
```

### Pattern 2: write_file Tool

```python
class WriteFileTool:
    """Write file contents with safety checks"""

    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

    def __init__(self, workspace_dir: str):
        self.workspace_dir = Path(workspace_dir).resolve()

    def write(self,
              file_path: str,
              content: str,
              create_dirs: bool = True,
              backup: bool = True) -> dict:
        """
        Write content to file safely.

        Args:
            file_path: Absolute path to file
            content: Content to write
            create_dirs: Create parent directories if needed
            backup: Backup existing file before overwriting

        Returns:
            {'success': bool, 'file_path': str, 'error': str}
        """
        try:
            # Sanitize path
            path = ArgumentSanitizer.sanitize_path(file_path)

            # Check size
            if len(content) > self.MAX_FILE_SIZE:
                return {
                    'success': False,
                    'error': f"Content too large: {len(content)} bytes"
                }

            # Create parent directories if needed
            if create_dirs:
                path.parent.mkdir(parents=True, exist_ok=True)

            # Backup existing file
            if backup and path.exists():
                backup_path = path.with_suffix(path.suffix + '.bak')
                path.rename(backup_path)

            # Write new content
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)

            return {
                'success': True,
                'file_path': str(path),
                'size_bytes': len(content),
                'lines_written': content.count('\n') + 1
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
```

### Pattern 3: execute_bash Tool

```python
class ExecuteBashTool:
    """Execute bash commands with sandboxing"""

    TIMEOUT = 30  # seconds

    # Whitelist of safe commands (allow by default, deny dangerous)
    DANGEROUS_COMMANDS = {
        'rm -rf',
        'dd if=/dev/zero',
        'mkfs',
        ':(){:|:&};:',  # Fork bomb
        'sudo',
        'systemctl stop',
    }

    def __init__(self, workspace_dir: str):
        self.workspace_dir = Path(workspace_dir).resolve()
        self.executor = SandboxExecutor(str(self.workspace_dir))

    def execute(self, command: str) -> dict:
        """
        Execute bash command safely.

        Args:
            command: Bash command to execute

        Returns:
            {'success': bool, 'stdout': str, 'stderr': str, ...}
        """
        try:
            # Validate command
            command = ArgumentSanitizer.sanitize_bash_command(command)

            # Check for dangerous patterns
            for dangerous in self.DANGEROUS_COMMANDS:
                if dangerous in command:
                    return {
                        'success': False,
                        'error': f"Dangerous command blocked: {dangerous}"
                    }

            # Execute in sandbox
            result = self.executor.execute_bash(command)

            return {
                'success': result['success'],
                'stdout': result['stdout'][:100000],  # Limit output
                'stderr': result['stderr'][:10000],
                'returncode': result['returncode'],
                'timeout': result['timeout']
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
```

### Pattern 4: search_codebase Tool

```python
import subprocess

class SearchCodebaseTool:
    """Search files using ripgrep (safe wrapper)"""

    def __init__(self, workspace_dir: str):
        self.workspace_dir = Path(workspace_dir).resolve()

    def search(self,
               pattern: str,
               file_glob: Optional[str] = None,
               context_lines: int = 0) -> dict:
        """
        Search files with pattern.

        Args:
            pattern: Regex pattern to search for
            file_glob: Glob pattern to filter files
            context_lines: Lines of context around matches

        Returns:
            {'success': bool, 'matches': [{'file': str, 'line': int, 'text': str}]}
        """
        try:
            # Validate pattern
            re.compile(pattern)  # Will raise if invalid regex

            if len(pattern) > 1000:
                return {
                    'success': False,
                    'error': "Pattern too long"
                }

            # Build ripgrep command
            cmd = [
                'rg',
                '--with-filename',
                '--line-number',
                '--context', str(context_lines),
                '--max-count', '100',  # Limit results
                pattern
            ]

            if file_glob:
                cmd.extend(['--glob', file_glob])

            # Execute with timeout
            result = subprocess.run(
                cmd,
                cwd=str(self.workspace_dir),
                capture_output=True,
                timeout=10,
                text=True
            )

            # Parse results
            matches = []
            for line in result.stdout.split('\n')[:100]:  # Limit output
                if line.strip():
                    matches.append(line)

            return {
                'success': result.returncode == 0,
                'matches': matches,
                'match_count': len(matches)
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
```

### Pattern 5: list_directory Tool

```python
class ListDirectoryTool:
    """List directory contents safely"""

    def __init__(self, workspace_dir: str):
        self.workspace_dir = Path(workspace_dir).resolve()

    def list(self,
             dir_path: str,
             recursive: bool = False,
             pattern: Optional[str] = None) -> dict:
        """
        List directory contents.

        Args:
            dir_path: Directory to list
            recursive: Include subdirectories
            pattern: Glob pattern to filter

        Returns:
            {'success': bool, 'entries': [{'name': str, 'type': 'file'|'dir', 'size': int}]}
        """
        try:
            # Sanitize path
            path = ArgumentSanitizer.sanitize_path(dir_path)

            if not path.is_dir():
                return {
                    'success': False,
                    'error': f"Not a directory: {dir_path}"
                }

            entries = []

            if recursive:
                iterator = path.rglob('*')
            else:
                iterator = path.iterdir()

            for entry in list(iterator)[:1000]:  # Limit to 1000 entries
                if pattern and not entry.name.endswith(pattern):
                    continue

                entries.append({
                    'name': entry.name,
                    'relative_path': str(entry.relative_to(self.workspace_dir)),
                    'type': 'dir' if entry.is_dir() else 'file',
                    'size': entry.stat().st_size if entry.is_file() else 0,
                })

            return {
                'success': True,
                'entries': entries,
                'count': len(entries)
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
```

---

## Code Examples

### Complete Tool Engine Example

```python
import json
import time
from typing import Callable, Dict, Any
from dataclasses import dataclass
from enum import Enum

@dataclass
class ToolDefinition:
    """Schema for a single tool"""
    name: str
    description: str
    parameters: dict
    handler: Callable
    requires_approval: bool = False

class ToolEngine:
    """Complete tool execution engine"""

    def __init__(self, workspace_dir: str):
        self.workspace_dir = workspace_dir
        self.tools = {}
        self.executor = SandboxExecutor(workspace_dir)
        self.authorizer = ToolAuthorizer()
        self.logger = ToolCallLogger("/tmp/tool-calls.log")

        self._register_tools()

    def _register_tools(self):
        """Register all available tools"""

        read_tool = ReadFileTool(self.workspace_dir)
        write_tool = WriteFileTool(self.workspace_dir)
        bash_tool = ExecuteBashTool(self.workspace_dir)
        search_tool = SearchCodebaseTool(self.workspace_dir)
        list_tool = ListDirectoryTool(self.workspace_dir)

        self.tools['read_file'] = ToolDefinition(
            name='read_file',
            description='Read file contents',
            parameters={
                'type': 'object',
                'properties': {
                    'file_path': {'type': 'string'},
                    'max_lines': {'type': 'integer', 'default': 2000}
                },
                'required': ['file_path']
            },
            handler=read_tool.read
        )

        self.tools['write_file'] = ToolDefinition(
            name='write_file',
            description='Write to file',
            parameters={
                'type': 'object',
                'properties': {
                    'file_path': {'type': 'string'},
                    'content': {'type': 'string'},
                    'create_dirs': {'type': 'boolean', 'default': True}
                },
                'required': ['file_path', 'content']
            },
            handler=write_tool.write,
            requires_approval=True
        )

        self.tools['execute_bash'] = ToolDefinition(
            name='execute_bash',
            description='Execute bash command',
            parameters={
                'type': 'object',
                'properties': {
                    'command': {'type': 'string'}
                },
                'required': ['command']
            },
            handler=bash_tool.execute,
            requires_approval=True
        )

        self.tools['search_codebase'] = ToolDefinition(
            name='search_codebase',
            description='Search files with regex',
            parameters={
                'type': 'object',
                'properties': {
                    'pattern': {'type': 'string'},
                    'file_glob': {'type': 'string'},
                    'context_lines': {'type': 'integer', 'default': 0}
                },
                'required': ['pattern']
            },
            handler=search_tool.search
        )

        self.tools['list_directory'] = ToolDefinition(
            name='list_directory',
            description='List directory contents',
            parameters={
                'type': 'object',
                'properties': {
                    'dir_path': {'type': 'string'},
                    'recursive': {'type': 'boolean', 'default': False},
                    'pattern': {'type': 'string'}
                },
                'required': ['dir_path']
            },
            handler=list_tool.list
        )

    def get_tool_schemas(self) -> list:
        """Return all tool schemas for LLM"""
        return [
            {
                'type': 'function',
                'function': {
                    'name': tool.name,
                    'description': tool.description,
                    'parameters': tool.parameters
                }
            }
            for tool in self.tools.values()
        ]

    def execute_tool_call(self, tool_call: dict, user_id: str = "default") -> dict:
        """
        Execute a single tool call from LLM.

        Args:
            tool_call: {
                'id': str,
                'type': 'function',
                'function': {
                    'name': str,
                    'arguments': str (JSON)
                }
            }
            user_id: User executing the tool

        Returns:
            {'tool_call_id': str, 'success': bool, 'result': Any, 'duration_ms': float}
        """
        start_time = time.time()

        try:
            tool_name = tool_call['function']['name']
            args_json = tool_call['function']['arguments']

            # Parse arguments
            try:
                args = json.loads(args_json)
            except json.JSONDecodeError as e:
                return {
                    'tool_call_id': tool_call.get('id'),
                    'success': False,
                    'result': f"Invalid JSON arguments: {e}",
                    'duration_ms': (time.time() - start_time) * 1000
                }

            # Check authorization
            if not self.authorizer.authorize(tool_name, user_id):
                return {
                    'tool_call_id': tool_call.get('id'),
                    'success': False,
                    'result': f"Not authorized to use {tool_name}",
                    'duration_ms': (time.time() - start_time) * 1000
                }

            # Validate arguments
            tool_def = self.tools[tool_name]
            try:
                # In production, use Pydantic models here
                validated_args = self._validate_arguments(tool_def, args)
            except ValueError as e:
                return {
                    'tool_call_id': tool_call.get('id'),
                    'success': False,
                    'result': f"Invalid arguments: {e}",
                    'duration_ms': (time.time() - start_time) * 1000
                }

            # Execute tool
            result = tool_def.handler(**validated_args)

            # Log the call
            duration_ms = (time.time() - start_time) * 1000
            self.logger.log_tool_call(
                tool_name,
                validated_args,
                result,
                duration_ms,
                result.get('success', False)
            )

            return {
                'tool_call_id': tool_call.get('id'),
                'success': result.get('success', False),
                'result': result,
                'duration_ms': duration_ms
            }

        except Exception as e:
            return {
                'tool_call_id': tool_call.get('id'),
                'success': False,
                'result': f"Tool execution error: {str(e)}",
                'duration_ms': (time.time() - start_time) * 1000
            }

    def _validate_arguments(self, tool_def: ToolDefinition, args: dict) -> dict:
        """Validate arguments against tool schema"""
        required = tool_def.parameters.get('required', [])

        # Check required parameters
        for param in required:
            if param not in args:
                raise ValueError(f"Missing required parameter: {param}")

        return args
```

---

## Best Practices

### 1. Always Use Absolute Paths

Bad:
```python
# DON'T DO THIS
open("file.txt")  # Relative to CWD
open("../other/file.txt")  # Path traversal risk
```

Good:
```python
# DO THIS
Path("/home/lemoneater/Devel/elpis/workspace/file.txt").resolve()
```

### 2. Separate Concerns

```python
# Tool implementation separate from validation
class ReadFileInput(BaseModel):
    """Handles validation"""
    file_path: str

class ReadFileTool:
    """Handles execution"""
    def read(self, input: ReadFileInput):
        pass
```

### 3. Use Type Hints Everywhere

```python
def execute_bash(self, command: str) -> Dict[str, Any]:
    """Clear input/output types help catch bugs"""
    pass
```

### 4. Log Everything

```python
# Critical for debugging and security auditing
self.logger.log_tool_call(
    tool_name='read_file',
    args={'file_path': '/path/to/file'},
    result={'success': True, 'lines': 42},
    duration_ms=15.5,
    success=True
)
```

### 5. Handle Errors Gracefully

```python
# Return structured error info, not just False
return {
    'success': False,
    'error': 'File not found',
    'suggestions': [
        'Check that file exists',
        'Try listing directory first'
    ]
}
```

### 6. Set Reasonable Limits

```python
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_COMMAND_LENGTH = 10000
MAX_SEARCH_RESULTS = 100
TIMEOUT_SECONDS = 30
```

### 7. Use Pydantic for Validation

```python
from pydantic import BaseModel, Field, validator

class BashCommandInput(BaseModel):
    command: str = Field(
        max_length=10000,
        description="Bash command to execute"
    )

    @validator('command')
    def no_semicolons(cls, v):
        if ';' in v:
            raise ValueError('Semicolons not allowed')
        return v
```

### 8. Implement Retry Logic

```python
for attempt in range(max_retries):
    try:
        result = execute_tool(...)
        if result['success']:
            return result
    except TransientError:
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)  # Exponential backoff
```

### 9. Monitor Resource Usage

```python
# Track execution time and resource usage
start_time = time.time()
result = execute_tool(...)
duration_ms = (time.time() - start_time) * 1000

if duration_ms > TIMEOUT_THRESHOLD:
    alert_user(f"Tool took {duration_ms}ms")
```

### 10. Provide Clear Error Messages

```python
# Bad
return {'success': False, 'error': 'Failed'}

# Good
return {
    'success': False,
    'error': 'File /path/to/file.txt not found',
    'file_path': '/path/to/file.txt',
    'suggestions': [
        'Check that file path exists',
        'Try using list_directory to browse'
    ]
}
```

---

## Recommendations for Elpis

### Phase 1 Implementation (Weeks 1-2)

1. **Start with read_file and list_directory**
   - Low risk, teaches pattern
   - Good for understanding file system

2. **Use OpenAI Function Calling format**
   - Compatible with llama-cpp-python
   - Industry standard
   - Easy to extend

3. **Implement basic validation**
   - Pydantic models for inputs
   - Path traversal checks
   - Size limits

4. **Simple sandbox**
   - Workspace directory restriction
   - Timeout enforcement
   - Logging

### Phase 2 Enhancement (Weeks 3-4)

1. **Add execute_bash with safety checks**
   - Command whitelisting approach
   - Deny dangerous patterns
   - Time limits

2. **Implement proper error handling**
   - Catch exceptions gracefully
   - Return helpful messages
   - Retry logic

3. **Add monitoring/logging**
   - Log all tool calls
   - Track execution times
   - Alert on issues

### Phase 3+ Advanced (Weeks 5+)

1. **OS-level sandboxing**
   - Consider Docker/containers
   - Network isolation
   - Resource limits

2. **Tool composition**
   - Tools that call other tools
   - Sub-task spawning
   - Parallel execution

3. **Integration with emotional system**
   - Tool success/failure affects emotions
   - Emotion state affects tool selection
   - Learning from tool failures

---

## Key References

### Tool Definition & Function Calling
- [vLLM Tool Calling Documentation](https://docs.vllm.ai/en/latest/features/tool_calling/)
- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
- [Prompt Engineering Guide on Function Calling](https://www.promptingguide.ai/applications/function_calling)

### Implementation Libraries
- [llama-cpp-python Documentation](https://llama-cpp-python.readthedocs.io/)
- [Pydantic AI Framework](https://ai.pydantic.dev/)
- [LangChain Agent Documentation](https://docs.langchain.com/oss/python/langchain/agents)

### Security & Sandboxing
- [Claude Code Sandboxing](https://code.claude.com/docs/en/sandboxing)
- [Trail of Bits: Prompt Injection to RCE](https://blog.trailofbits.com/2025/10/22/prompt-injection-to-rce-in-ai-agents/)
- [LLM Agent Sandboxing Techniques](https://www.codeant.ai/blogs/agentic-rag-shell-sandboxing)

### Error Handling
- [LangChain Agent Error Handling](https://apxml.com/courses/langchain-production-llm/chapter-2-sophisticated-agents-tools/agent-error-handling)
- [Error Recovery Strategies](https://www.gocodeo.com/post/error-recovery-and-fallback-strategies-in-ai-agent-development)
- [ReAct Pattern](https://agent-patterns.readthedocs.io/en/stable/patterns/re_act.html)

---

## Summary

The Elpis Tool System should implement:

1. **Standard Interfaces**: OpenAI-compatible function calling format
2. **Robust Validation**: Pydantic models for all tool inputs
3. **Multi-Layer Security**: Validation → Authorization → Sanitization → Sandboxing
4. **Error Resilience**: Graceful degradation, retry logic, helpful error messages
5. **Full Observability**: Complete logging and monitoring
6. **Emotional Integration**: Tool outcomes feed emotional system

This approach balances safety, usability, and extensibility for the Elpis emotional coding agent.

---

**Document Status:** Complete
**Last Updated:** January 2026
**Next Steps:** Implementation planning for Phase 1
