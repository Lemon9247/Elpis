# Phase 1 Implementation Plan: Elpis Agent Harness

## Overview

This plan details the implementation of Phase 1 for the Elpis emotional coding agent: the basic agent harness with LLM inference, tool execution, and interactive REPL. Memory and emotion systems will come in Phase 2-3.

**Key User Preferences:**
- GPU Support: Both NVIDIA (CUDA) and AMD (ROCm), with CPU fallback
- Model Download: Automatic script (create but don't run during implementation)
- Testing: Comprehensive (>80% coverage)
- Concurrency: Async from the start

## Project Structure

```
/home/lemoneater/Devel/elpis/
├── src/
│   └── elpis/
│       ├── __init__.py
│       ├── cli.py                        # Async CLI entry point
│       │
│       ├── config/
│       │   ├── __init__.py
│       │   ├── settings.py               # Pydantic settings models
│       │   └── defaults.toml             # Default configuration
│       │
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── inference.py              # Async LLM wrapper (llama-cpp-python)
│       │   └── prompts.py                # System prompts
│       │
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── tool_engine.py            # Async tool orchestrator
│       │   ├── tool_definitions.py       # Pydantic tool models
│       │   └── implementations/
│       │       ├── __init__.py
│       │       ├── file_tools.py         # read_file, write_file
│       │       ├── bash_tool.py          # execute_bash
│       │       ├── search_tool.py        # search_codebase (ripgrep)
│       │       └── directory_tool.py     # list_directory
│       │
│       ├── agent/
│       │   ├── __init__.py
│       │   ├── orchestrator.py           # Async ReAct loop
│       │   └── repl.py                   # Async REPL
│       │
│       └── utils/
│           ├── __init__.py
│           ├── logging.py                # Loguru configuration
│           ├── exceptions.py             # Custom exceptions
│           └── hardware.py               # GPU detection (CUDA/ROCm/CPU)
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                       # Pytest async fixtures
│   ├── unit/
│   │   ├── test_llm_inference.py
│   │   ├── test_tool_engine.py
│   │   ├── test_file_tools.py
│   │   ├── test_bash_tool.py
│   │   ├── test_search_tool.py
│   │   ├── test_directory_tool.py
│   │   ├── test_orchestrator.py
│   │   └── test_hardware_detection.py
│   └── integration/
│       ├── test_agent_workflow.py
│       └── test_tool_execution.py
│
├── workspace/                            # Agent's sandboxed directory
│   └── .gitkeep
│
├── data/
│   └── models/                           # Downloaded LLM weights
│       └── .gitkeep
│
├── logs/
│   └── .gitkeep
│
├── configs/
│   ├── config.default.toml              # Development config
│   └── config.local.toml.example        # Local override template
│
├── scripts/
│   ├── download_model.py                # Auto-download Llama 3.1 8B
│   └── setup.sh                         # Environment setup
│
├── pyproject.toml                        # Project metadata
├── .pre-commit-config.yaml              # Code quality hooks
├── .gitignore
├── README.md
└── LICENSE
```

## Critical Components

### 1. Configuration System (src/elpis/config/settings.py)

**Pydantic Settings Models:**

```python
class ModelSettings(BaseSettings):
    """LLM configuration"""
    path: str                              # Path to GGUF model
    context_length: int = 8192            # Context window
    gpu_layers: int = 35                  # GPU offloading (0 = CPU only)
    n_threads: int = 8                    # CPU threads
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    hardware_backend: str = "auto"        # "cuda", "rocm", "cpu", "auto"

class ToolSettings(BaseSettings):
    """Tool execution configuration"""
    workspace_dir: str = "./workspace"
    max_bash_timeout: int = 30
    max_file_size: int = 10485760         # 10MB
    enable_dangerous_commands: bool = False

class LoggingSettings(BaseSettings):
    """Logging configuration"""
    level: str = "INFO"
    output_file: str = "./logs/elpis.log"
    format: str = "json"

class Settings(BaseSettings):
    """Root configuration"""
    model: ModelSettings = ModelSettings()
    tools: ToolSettings = ToolSettings()
    logging: LoggingSettings = LoggingSettings()

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
```

**configs/config.default.toml:**

```toml
[model]
path = "./data/models/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf"
context_length = 8192
gpu_layers = 35
n_threads = 8
temperature = 0.7
top_p = 0.9
max_tokens = 2048
hardware_backend = "auto"

[tools]
workspace_dir = "./workspace"
max_bash_timeout = 30
max_file_size = 10485760
enable_dangerous_commands = false

[logging]
level = "INFO"
output_file = "./logs/elpis.log"
format = "json"
```

### 2. Hardware Detection (src/elpis/utils/hardware.py)

**GPU Detection for CUDA/ROCm:**

```python
import subprocess
import platform
from enum import Enum
from typing import Optional

class HardwareBackend(Enum):
    """Available hardware backends"""
    CUDA = "cuda"      # NVIDIA GPU
    ROCM = "rocm"      # AMD GPU
    CPU = "cpu"        # CPU only

def detect_hardware() -> HardwareBackend:
    """
    Detect available GPU hardware.
    Priority: CUDA > ROCm > CPU
    """
    # Check for NVIDIA GPU (CUDA)
    if check_cuda_available():
        return HardwareBackend.CUDA

    # Check for AMD GPU (ROCm)
    if check_rocm_available():
        return HardwareBackend.ROCM

    # Fallback to CPU
    return HardwareBackend.CPU

def check_cuda_available() -> bool:
    """Check if NVIDIA GPU with CUDA is available"""
    try:
        result = subprocess.run(
            ['nvidia-smi'],
            capture_output=True,
            timeout=2
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

def check_rocm_available() -> bool:
    """Check if AMD GPU with ROCm is available"""
    try:
        result = subprocess.run(
            ['rocm-smi'],
            capture_output=True,
            timeout=2
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

def get_recommended_gpu_layers(backend: HardwareBackend) -> int:
    """Get recommended GPU layers for backend"""
    return {
        HardwareBackend.CUDA: 35,    # Offload all layers
        HardwareBackend.ROCM: 35,    # Offload all layers
        HardwareBackend.CPU: 0       # No GPU offloading
    }[backend]
```

### 3. LLM Inference (src/elpis/llm/inference.py)

**Async LLM Wrapper with GPU Support:**

```python
import asyncio
from typing import List, Optional, Dict, Any
from llama_cpp import Llama
from elpis.config.settings import ModelSettings
from elpis.utils.hardware import detect_hardware, HardwareBackend

class LlamaInference:
    """Async wrapper around llama-cpp-python"""

    def __init__(self, settings: ModelSettings):
        self.settings = settings
        self.backend = self._detect_backend()
        self.model = self._load_model()

    def _detect_backend(self) -> HardwareBackend:
        """Detect or use configured hardware backend"""
        if self.settings.hardware_backend == "auto":
            return detect_hardware()
        return HardwareBackend(self.settings.hardware_backend)

    def _load_model(self) -> Llama:
        """Load GGUF model with appropriate backend"""
        return Llama(
            model_path=self.settings.path,
            n_ctx=self.settings.context_length,
            n_gpu_layers=self.settings.gpu_layers,
            n_threads=self.settings.n_threads,
            chat_format="llama-3",
            verbose=False
        )

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> str:
        """Generate chat completion (run in thread pool)"""
        return await asyncio.to_thread(
            self._chat_completion_sync,
            messages,
            max_tokens,
            temperature,
            top_p
        )

    def _chat_completion_sync(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int],
        temperature: Optional[float],
        top_p: Optional[float]
    ) -> str:
        """Synchronous chat completion"""
        response = self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens or self.settings.max_tokens,
            temperature=temperature or self.settings.temperature,
            top_p=top_p or self.settings.top_p
        )
        return response['choices'][0]['message']['content']

    async def function_call(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        temperature: Optional[float] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """Generate function calls using OpenAI-compatible format"""
        return await asyncio.to_thread(
            self._function_call_sync,
            messages,
            tools,
            temperature
        )

    def _function_call_sync(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        temperature: Optional[float]
    ) -> Optional[List[Dict[str, Any]]]:
        """Synchronous function call generation"""
        response = self.model.create_chat_completion(
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=temperature or self.settings.temperature
        )

        message = response['choices'][0]['message']
        return message.get('tool_calls')
```

### 4. Tool System (src/elpis/tools/)

**tool_definitions.py - Pydantic Models:**

```python
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Callable, Dict, Any, Type
from pathlib import Path

class ToolInput(BaseModel):
    """Base class for all tool inputs"""
    class Config:
        validate_assignment = True

class ReadFileInput(ToolInput):
    file_path: str = Field(description="Absolute path to file")
    max_lines: Optional[int] = Field(default=2000, ge=1, le=100000)

    @field_validator('file_path')
    @classmethod
    def validate_path(cls, v: str) -> str:
        if '\x00' in v:
            raise ValueError("Null bytes not allowed in path")
        return v

class WriteFileInput(ToolInput):
    file_path: str = Field(description="Absolute path to file")
    content: str = Field(description="Content to write")
    create_dirs: bool = Field(default=True)

    @field_validator('file_path')
    @classmethod
    def validate_path(cls, v: str) -> str:
        if '\x00' in v:
            raise ValueError("Null bytes not allowed in path")
        return v

class ExecuteBashInput(ToolInput):
    command: str = Field(description="Bash command to execute", max_length=10000)

class SearchCodebaseInput(ToolInput):
    pattern: str = Field(description="Regex pattern to search")
    file_glob: Optional[str] = Field(default=None, description="File glob pattern")
    context_lines: int = Field(default=0, ge=0, le=10)

class ListDirectoryInput(ToolInput):
    dir_path: str = Field(description="Directory path")
    recursive: bool = Field(default=False)
    pattern: Optional[str] = Field(default=None)

class ToolDefinition:
    """Schema for a single tool"""
    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        input_model: Type[BaseModel],
        handler: Callable
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.input_model = input_model
        self.handler = handler
```

**tool_engine.py - Async Tool Orchestrator:**

```python
import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from loguru import logger

from elpis.tools.tool_definitions import ToolDefinition
from elpis.tools.implementations.file_tools import FileTools
from elpis.tools.implementations.bash_tool import BashTool
from elpis.tools.implementations.search_tool import SearchTool
from elpis.tools.implementations.directory_tool import DirectoryTool
from elpis.utils.exceptions import ToolExecutionError

class ToolEngine:
    """Async tool execution orchestrator"""

    def __init__(self, workspace_dir: str, settings):
        self.workspace_dir = Path(workspace_dir).resolve()
        self.settings = settings
        self.tools: Dict[str, ToolDefinition] = {}
        self._register_tools()

    def _register_tools(self):
        """Register all available tools"""
        file_tools = FileTools(self.workspace_dir, self.settings)
        bash_tool = BashTool(self.workspace_dir, self.settings)
        search_tool = SearchTool(self.workspace_dir, self.settings)
        directory_tool = DirectoryTool(self.workspace_dir, self.settings)

        # Register each tool with its schema
        # (see full implementation in tool_engine.py)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return OpenAI-compatible tool schemas"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            }
            for tool in self.tools.values()
        ]

    async def execute_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single tool call from LLM"""
        start_time = time.time()

        try:
            tool_name = tool_call['function']['name']
            args_json = tool_call['function']['arguments']

            # Parse and validate arguments
            args = json.loads(args_json)
            tool_def = self.tools[tool_name]
            validated_args = tool_def.input_model(**args)

            # Execute tool (async)
            result = await tool_def.handler(**validated_args.model_dump())

            duration_ms = (time.time() - start_time) * 1000
            logger.info(f"Tool {tool_name} executed in {duration_ms:.2f}ms")

            return {
                'tool_call_id': tool_call.get('id'),
                'success': result.get('success', False),
                'result': result,
                'duration_ms': duration_ms
            }

        except Exception as e:
            logger.exception(f"Tool execution failed: {e}")
            return {
                'tool_call_id': tool_call.get('id'),
                'success': False,
                'result': {'error': str(e)},
                'duration_ms': (time.time() - start_time) * 1000
            }

    def sanitize_path(self, path: str) -> Path:
        """Sanitize and validate file path"""
        resolved = Path(path).resolve()

        # Check within workspace
        try:
            resolved.relative_to(self.workspace_dir)
        except ValueError:
            raise ToolExecutionError(
                f"Path {path} is outside workspace {self.workspace_dir}"
            )

        return resolved
```

**implementations/file_tools.py:**

```python
import asyncio
from pathlib import Path
from typing import Dict, Any

class FileTools:
    """File operation tools"""

    def __init__(self, workspace_dir: Path, settings):
        self.workspace_dir = workspace_dir
        self.settings = settings

    async def read_file(
        self,
        file_path: str,
        max_lines: int = 2000
    ) -> Dict[str, Any]:
        """Read file contents asynchronously"""
        return await asyncio.to_thread(
            self._read_file_sync,
            file_path,
            max_lines
        )

    def _read_file_sync(self, file_path: str, max_lines: int) -> Dict[str, Any]:
        """Synchronous file read implementation"""
        try:
            path = self.workspace_dir / file_path
            path = path.resolve()

            # Validate within workspace
            path.relative_to(self.workspace_dir)

            # Check file size
            if path.stat().st_size > self.settings.tools.max_file_size:
                return {
                    'success': False,
                    'error': f'File too large: {path.stat().st_size} bytes'
                }

            # Read file
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()[:max_lines]

            return {
                'success': True,
                'content': ''.join(lines),
                'line_count': len(lines),
                'file_path': str(path)
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    async def write_file(
        self,
        file_path: str,
        content: str,
        create_dirs: bool = True
    ) -> Dict[str, Any]:
        """Write file contents asynchronously"""
        return await asyncio.to_thread(
            self._write_file_sync,
            file_path,
            content,
            create_dirs
        )

    def _write_file_sync(
        self,
        file_path: str,
        content: str,
        create_dirs: bool
    ) -> Dict[str, Any]:
        """Synchronous file write implementation"""
        try:
            path = self.workspace_dir / file_path
            path = path.resolve()

            # Validate within workspace
            path.relative_to(self.workspace_dir)

            # Create directories
            if create_dirs:
                path.parent.mkdir(parents=True, exist_ok=True)

            # Backup existing file
            if path.exists():
                backup = path.with_suffix(path.suffix + '.bak')
                path.rename(backup)

            # Write new content
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)

            return {
                'success': True,
                'file_path': str(path),
                'size_bytes': len(content)
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
```

### 5. Agent Orchestrator (src/elpis/agent/orchestrator.py)

**Async ReAct Pattern Loop:**

```python
import asyncio
from typing import List, Dict, Any, Optional
from loguru import logger

from elpis.llm.inference import LlamaInference
from elpis.llm.prompts import build_system_prompt
from elpis.tools.tool_engine import ToolEngine

class AgentOrchestrator:
    """Main agent reasoning loop using ReAct pattern"""

    def __init__(
        self,
        llm: LlamaInference,
        tools: ToolEngine,
        settings
    ):
        self.llm = llm
        self.tools = tools
        self.settings = settings
        self.message_history: List[Dict[str, Any]] = []

    async def process(self, user_input: str) -> str:
        """
        Process user input using ReAct pattern:
        1. REASON: Build prompt with context
        2. ACT: Generate LLM response (text or tool calls)
        3. OBSERVE: Execute tools if needed
        4. REPEAT: Loop until final answer
        """
        # Add user message to history
        self.message_history.append({
            'role': 'user',
            'content': user_input
        })

        # ReAct loop with max iterations
        max_iterations = 10
        for iteration in range(max_iterations):
            logger.debug(f"ReAct iteration {iteration + 1}")

            # Build messages with system prompt
            messages = self._build_messages()

            # Try to get function calls from LLM
            tool_calls = await self.llm.function_call(
                messages=messages,
                tools=self.tools.get_tool_schemas()
            )

            if tool_calls:
                # ACT: Execute tools
                logger.info(f"Executing {len(tool_calls)} tool calls")
                results = await self._execute_tools(tool_calls)

                # OBSERVE: Add tool results to history
                self.message_history.append({
                    'role': 'assistant',
                    'content': None,
                    'tool_calls': tool_calls
                })
                self.message_history.append({
                    'role': 'tool',
                    'content': self._format_tool_results(results)
                })

                # Continue loop for next reasoning step
                continue
            else:
                # No tool calls - get text response
                response = await self.llm.chat_completion(messages)

                # Add to history
                self.message_history.append({
                    'role': 'assistant',
                    'content': response
                })

                # Return final answer
                return response

        # Max iterations reached
        return "I apologize, but I've reached my reasoning limit. Please rephrase your request."

    def _build_messages(self) -> List[Dict[str, str]]:
        """Build message list with system prompt"""
        system_prompt = build_system_prompt(
            list(self.tools.tools.values())
        )

        messages = [
            {'role': 'system', 'content': system_prompt}
        ]
        messages.extend(self.message_history)

        return messages

    async def _execute_tools(
        self,
        tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Execute tool calls concurrently"""
        tasks = [
            self.tools.execute_tool_call(call)
            for call in tool_calls
        ]
        return await asyncio.gather(*tasks)

    def _format_tool_results(self, results: List[Dict[str, Any]]) -> str:
        """Format tool results for LLM"""
        formatted = []
        for result in results:
            if result['success']:
                formatted.append(
                    f"Tool succeeded: {result['result']}"
                )
            else:
                formatted.append(
                    f"Tool failed: {result['result']['error']}"
                )
        return "\n\n".join(formatted)

    def clear_history(self):
        """Clear message history"""
        self.message_history = []
```

### 6. REPL Interface (src/elpis/agent/repl.py)

**Async REPL with Rich Formatting:**

```python
import asyncio
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown

from elpis.agent.orchestrator import AgentOrchestrator

class ElpisREPL:
    """Interactive async REPL interface"""

    def __init__(self, agent: AgentOrchestrator):
        self.agent = agent
        self.console = Console()
        self.session = PromptSession(
            history=FileHistory('.elpis_history')
        )

    async def run(self):
        """Main REPL loop"""
        self._display_welcome()

        while True:
            try:
                # Get user input
                user_input = await self.session.prompt_async(
                    'elpis> ',
                    multiline=False
                )

                # Handle special commands
                if user_input.startswith('/'):
                    await self._handle_special_command(user_input)
                    continue

                if not user_input.strip():
                    continue

                # Process with agent
                self.console.print("[dim]Thinking...[/dim]")
                response = await self.agent.process(user_input)

                # Display response
                self._display_response(response)

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Interrupted[/yellow]")
                continue
            except EOFError:
                self.console.print("\n[cyan]Goodbye![/cyan]")
                break
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")

    def _display_welcome(self):
        """Display welcome banner"""
        welcome = """
# Elpis - Emotional Coding Agent

Phase 1: Basic Agent Harness

Available commands:
- /help    Show this help
- /clear   Clear conversation history
- /exit    Exit REPL

Just type naturally to interact with the agent.
        """
        self.console.print(Panel(Markdown(welcome), title="Welcome"))

    async def _handle_special_command(self, command: str):
        """Handle special / commands"""
        if command == '/help':
            self._display_welcome()
        elif command == '/clear':
            self.agent.clear_history()
            self.console.print("[green]History cleared[/green]")
        elif command == '/exit':
            raise EOFError()
        else:
            self.console.print(f"[red]Unknown command: {command}[/red]")

    def _display_response(self, response: str):
        """Display agent response with formatting"""
        # Try to detect code blocks
        if '```' in response:
            self.console.print(Markdown(response))
        else:
            self.console.print(Panel(response, title="Elpis"))
```

### 7. CLI Entry Point (src/elpis/cli.py)

```python
import asyncio
import click
from pathlib import Path
from loguru import logger

from elpis.config.settings import Settings
from elpis.utils.logging import configure_logging
from elpis.llm.inference import LlamaInference
from elpis.tools.tool_engine import ToolEngine
from elpis.agent.orchestrator import AgentOrchestrator
from elpis.agent.repl import ElpisREPL

@click.command()
@click.option('--config', default='configs/config.default.toml', help='Configuration file')
@click.option('--debug', is_flag=True, help='Enable debug logging')
def main(config: str, debug: bool):
    """Elpis - Emotional Coding Agent (Phase 1)"""

    # Load configuration
    settings = Settings()
    if debug:
        settings.logging.level = "DEBUG"

    # Configure logging
    configure_logging(settings.logging)
    logger.info("Starting Elpis...")

    # Check model exists
    model_path = Path(settings.model.path)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        logger.info("Run: python scripts/download_model.py")
        return

    # Initialize components
    llm = LlamaInference(settings.model)
    tools = ToolEngine(settings.tools.workspace_dir, settings)
    agent = AgentOrchestrator(llm, tools, settings)
    repl = ElpisREPL(agent)

    # Run REPL
    asyncio.run(repl.run())

if __name__ == '__main__':
    main()
```

### 8. Model Download Script (scripts/download_model.py)

```python
#!/usr/bin/env python3
"""
Download Llama 3.1 8B Instruct model from HuggingFace.
This script is NOT run automatically - user must execute manually.
"""

import sys
from pathlib import Path
from huggingface_hub import hf_hub_download
from rich.progress import Progress

MODEL_REPO = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
MODEL_FILE = "Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf"
LOCAL_DIR = "./data/models"

def download_model():
    """Download model with progress tracking"""

    print(f"Downloading {MODEL_FILE} from {MODEL_REPO}")
    print(f"Size: ~6.8GB")
    print(f"Destination: {LOCAL_DIR}")
    print()

    # Create directory
    Path(LOCAL_DIR).mkdir(parents=True, exist_ok=True)

    # Download with progress
    try:
        model_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False
        )

        print(f"\n✓ Download complete: {model_path}")
        return 0

    except Exception as e:
        print(f"\n✗ Download failed: {e}", file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(download_model())
```

## Dependencies (pyproject.toml)

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "elpis"
version = "0.1.0"
description = "Emotional coding agent with biologically-inspired memory and emotional modulation"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "GPL-3.0"}
authors = [{name = "Willow Sparks", email = "willow.sparks@gmail.com"}]

dependencies = [
    "llama-cpp-python>=0.2.0",      # LLM inference (CPU/CUDA/ROCm)
    "pydantic>=2.0",                # Data validation
    "pydantic-settings>=2.0",       # Configuration
    "loguru>=0.7.0",                # Logging
    "prompt-toolkit>=3.0.52",       # Async REPL
    "rich>=13.0",                   # Output formatting
    "click>=8.0",                   # CLI framework
    "huggingface-hub>=0.20.0",      # Model downloads
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "pytest-asyncio>=0.21",
    "ruff>=0.1.0",
    "mypy>=1.5",
    "pre-commit>=3.4",
]

cuda = [
    # Install with: CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python
]

rocm = [
    # Install with: CMAKE_ARGS="-DLLAMA_HIPBLAS=on" pip install llama-cpp-python
]

[project.scripts]
elpis = "elpis.cli:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --cov=src/elpis --cov-report=term-missing --cov-report=html"
asyncio_mode = "auto"

[tool.coverage.run]
source = ["src/elpis"]
omit = ["*/tests/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "if __name__ == .__main__.:",
    "raise NotImplementedError",
]

[tool.ruff]
target-version = "py310"
line-length = 100
select = ["E", "F", "I", "N", "UP", "W", "B", "A", "C4", "RUF"]
ignore = ["E501"]

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
check_untyped_defs = true
no_implicit_optional = true

[[tool.mypy.overrides]]
module = ["llama_cpp.*"]
ignore_missing_imports = true
```

## Implementation Steps

### Week 1: Foundation & Configuration

**Day 1-2: Project Setup**
- [ ] Create complete directory structure
- [ ] Set up pyproject.toml with all dependencies
- [ ] Initialize git repository (.gitignore)
- [ ] Set up pre-commit hooks (ruff, mypy)
- [ ] Create README with setup instructions

**Day 3-4: Configuration & Logging**
- [ ] Implement Pydantic settings models (config/settings.py)
- [ ] Create default TOML configuration (configs/config.default.toml)
- [ ] Implement hardware detection (utils/hardware.py)
- [ ] Set up loguru logging (utils/logging.py)
- [ ] Write tests for configuration system (>80% coverage)

**Day 5-7: LLM Layer**
- [ ] Implement async LLM wrapper (llm/inference.py)
- [ ] Add GPU backend support (CUDA/ROCm/CPU)
- [ ] Create system prompts (llm/prompts.py)
- [ ] Create model download script (scripts/download_model.py)
- [ ] Write comprehensive tests for LLM layer
- [ ] Document GPU setup instructions

### Week 2: Tool System

**Day 8-10: Tool Framework**
- [ ] Create Pydantic tool input models (tools/tool_definitions.py)
- [ ] Implement async ToolEngine (tools/tool_engine.py)
- [ ] Add path sanitization and validation
- [ ] Add command sanitization for bash
- [ ] Write unit tests for tool framework

**Day 11-14: Tool Implementations**
- [ ] Implement async file_tools.py (read_file, write_file)
- [ ] Implement async bash_tool.py (execute_bash)
- [ ] Implement async search_tool.py (search_codebase)
- [ ] Implement async directory_tool.py (list_directory)
- [ ] Write comprehensive unit tests for each tool
- [ ] Test error handling and edge cases
- [ ] Register all tools in ToolEngine

### Week 3: Agent & REPL

**Day 15-18: Agent Orchestrator**
- [ ] Implement async AgentOrchestrator (agent/orchestrator.py)
- [ ] Implement ReAct pattern loop
- [ ] Add message history management
- [ ] Handle concurrent tool execution
- [ ] Write unit tests for orchestrator
- [ ] Write integration tests for agent workflow

**Day 19-21: REPL Interface**
- [ ] Implement async REPL (agent/repl.py)
- [ ] Add prompt_toolkit integration
- [ ] Add Rich output formatting
- [ ] Implement special commands (/help, /clear, /exit)
- [ ] Add Markdown rendering for responses
- [ ] Add syntax highlighting for code blocks
- [ ] Test interactive session

### Week 4: Integration & Testing

**Day 22-24: CLI & Integration**
- [ ] Create async CLI entry point (cli.py)
- [ ] Wire up all components
- [ ] Add command-line argument parsing
- [ ] Write integration tests for full workflow
- [ ] Test all Phase 1 success criteria:
  - [ ] Read Python file and explain
  - [ ] Write simple function to file
  - [ ] Run tests via bash command

**Day 25-26: Comprehensive Testing**
- [ ] Ensure >80% test coverage
- [ ] Test async behavior thoroughly
- [ ] Test error handling paths
- [ ] Test GPU backend switching
- [ ] Test tool safety mechanisms
- [ ] Performance testing

**Day 27-28: Documentation & Polish**
- [ ] Write comprehensive README
- [ ] Document GPU setup (CUDA/ROCm/CPU)
- [ ] Document model download process
- [ ] Create configuration guide
- [ ] Write troubleshooting guide
- [ ] Add usage examples
- [ ] Code quality review (ruff, mypy)
- [ ] Tag Phase 1 release

## Verification Steps

### Phase 1 Success Criteria

**Functional Tests:**
1. **Read and explain file**: User asks to read a Python file, agent calls read_file and explains contents
2. **Write function**: User asks to write a simple function, agent calls write_file and creates correct file
3. **Run bash command**: User asks to run tests, agent calls execute_bash and reports results
4. **Search codebase**: User asks to find pattern, agent calls search_codebase and returns matches
5. **List directory**: User asks to see files, agent calls list_directory and displays results

**Technical Tests:**
1. **GPU Detection**: Verify CUDA/ROCm/CPU detection works correctly
2. **Async Execution**: All tools execute asynchronously without blocking
3. **Error Handling**: Tool failures are handled gracefully with helpful messages
4. **Path Safety**: Attempts to access files outside workspace are blocked
5. **Command Safety**: Dangerous bash commands are blocked
6. **Test Coverage**: >80% code coverage achieved

**Performance Tests:**
1. LLM inference: 35-50 tokens/sec on GPU, 15-25 on CPU
2. Tool execution: <2s for file operations
3. Total response time: <10s for typical queries
4. Memory usage: <10GB total

## Notes

**GPU Support:**
- Code supports both NVIDIA (CUDA) and AMD (ROCm)
- Auto-detection falls back to CPU if no GPU found
- Build instructions differ by platform (see README)

**Model Download:**
- Automatic download script provided
- User must run manually before first use
- Validates model file exists before starting

**Async Architecture:**
- All I/O operations run asynchronously
- LLM inference runs in thread pool (blocking operation)
- Tools execute concurrently when multiple calls made
- REPL uses async prompt_toolkit

**Safety:**
- Multi-layer validation (Pydantic + path checks + command filters)
- Workspace sandboxing enforced
- Dangerous commands blocked by default
- All tool calls logged for audit

**Testing:**
- Comprehensive unit tests for all components
- Integration tests for agent workflows
- pytest-asyncio for async test support
- >80% coverage target

## Critical Files

The following files are most critical for Phase 1 success:

1. **src/elpis/config/settings.py** - Foundation for configuration
2. **src/elpis/utils/hardware.py** - GPU detection (CUDA/ROCm/CPU)
3. **src/elpis/llm/inference.py** - Async LLM wrapper
4. **src/elpis/tools/tool_engine.py** - Async tool orchestrator
5. **src/elpis/agent/orchestrator.py** - ReAct loop implementation
6. **pyproject.toml** - Project structure and dependencies
7. **scripts/download_model.py** - Model download automation
