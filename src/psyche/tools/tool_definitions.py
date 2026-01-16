"""Tool definitions with Pydantic models for input validation."""
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Any, Callable, Dict, Optional, Type


class ToolInput(BaseModel):
    """Base class for all tool inputs."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")


class ReadFileInput(ToolInput):
    """Input model for read_file tool."""
    file_path: str = Field(description="Path to file (relative to workspace or absolute)")
    max_lines: Optional[int] = Field(
        default=2000,
        ge=0,
        le=100000,
        description="Maximum number of lines to read (0 or None = default 2000)"
    )

    @field_validator('file_path')
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Validate file path doesn't contain null bytes."""
        if '\x00' in v:
            raise ValueError("Null bytes not allowed in path")
        if not v.strip():
            raise ValueError("Path cannot be empty")
        return v

    @field_validator('max_lines')
    @classmethod
    def validate_max_lines(cls, v: Optional[int]) -> int:
        """Convert 0 or None to default value."""
        if v is None or v == 0:
            return 2000
        return v


class CreateFileInput(ToolInput):
    """Input model for create_file tool."""
    file_path: str = Field(description="Path to file (relative to workspace or absolute)")
    content: str = Field(description="Content to write to the new file")
    create_dirs: bool = Field(
        default=True,
        description="Create parent directories if they don't exist"
    )

    @field_validator('file_path')
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Validate file path doesn't contain null bytes."""
        if '\x00' in v:
            raise ValueError("Null bytes not allowed in path")
        if not v.strip():
            raise ValueError("Path cannot be empty")
        return v


class EditFileInput(ToolInput):
    """Input model for edit_file tool."""
    file_path: str = Field(description="Path to file (relative to workspace or absolute)")
    old_string: str = Field(description="The exact text to find and replace")
    new_string: str = Field(description="The text to replace it with")

    @field_validator('file_path')
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Validate file path doesn't contain null bytes."""
        if '\x00' in v:
            raise ValueError("Null bytes not allowed in path")
        if not v.strip():
            raise ValueError("Path cannot be empty")
        return v

    @field_validator('old_string')
    @classmethod
    def validate_old_string(cls, v: str) -> str:
        """Validate old_string is not empty."""
        if not v:
            raise ValueError("old_string cannot be empty")
        return v


class ExecuteBashInput(ToolInput):
    """Input model for execute_bash tool."""
    command: str = Field(
        description="Bash command to execute",
        max_length=10000
    )

    @field_validator('command')
    @classmethod
    def validate_command(cls, v: str) -> str:
        """Basic command validation."""
        if not v.strip():
            raise ValueError("Command cannot be empty")
        return v


class SearchCodebaseInput(ToolInput):
    """Input model for search_codebase tool."""
    pattern: str = Field(description="Regex pattern to search for")
    file_glob: Optional[str] = Field(
        default=None,
        description="File glob pattern (e.g., '*.py', '**/*.js')"
    )
    context_lines: int = Field(
        default=0,
        ge=0,
        le=10,
        description="Number of context lines to show around matches"
    )

    @field_validator('pattern')
    @classmethod
    def validate_pattern(cls, v: str) -> str:
        """Validate pattern is not empty."""
        if not v.strip():
            raise ValueError("Pattern cannot be empty")
        return v


class ListDirectoryInput(ToolInput):
    """Input model for list_directory tool."""
    dir_path: str = Field(
        default=".",
        description="Directory path (relative to workspace or absolute)"
    )
    recursive: bool = Field(
        default=False,
        description="List subdirectories recursively"
    )
    pattern: Optional[str] = Field(
        default=None,
        description="File glob pattern to filter results"
    )

    @field_validator('dir_path')
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Validate directory path doesn't contain null bytes."""
        if '\x00' in v:
            raise ValueError("Null bytes not allowed in path")
        return v


class RecallMemoryInput(ToolInput):
    """Input model for recall_memory tool."""
    query: str = Field(description="Search query to find relevant memories")
    n_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of memories to retrieve (1-20)"
    )

    @field_validator('query')
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate query is not empty."""
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v


class StoreMemoryInput(ToolInput):
    """Input model for store_memory tool."""
    content: str = Field(description="Content of the memory to store")
    summary: Optional[str] = Field(
        default=None,
        description="Brief summary of the memory (auto-generated if not provided)"
    )
    memory_type: str = Field(
        default="episodic",
        description="Type of memory: episodic (events), semantic (facts), procedural (how-to), emotional (feelings)"
    )
    tags: Optional[list[str]] = Field(
        default=None,
        description="Optional tags to categorize the memory"
    )

    @field_validator('content')
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Validate content is not empty."""
        if not v.strip():
            raise ValueError("Content cannot be empty")
        return v

    @field_validator('memory_type')
    @classmethod
    def validate_memory_type(cls, v: str) -> str:
        """Validate memory type."""
        valid_types = {"episodic", "semantic", "procedural", "emotional"}
        if v not in valid_types:
            raise ValueError(f"memory_type must be one of: {', '.join(valid_types)}")
        return v


class ToolDefinition:
    """Schema definition for a single tool."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        input_model: Type[ToolInput],
        handler: Callable
    ):
        """
        Initialize a tool definition.

        Args:
            name: Tool name (must match function name)
            description: Human-readable description of what the tool does
            parameters: JSON schema for tool parameters (OpenAI format)
            input_model: Pydantic model for input validation
            handler: Async function that executes the tool
        """
        self.name = name
        self.description = description
        self.parameters = parameters
        self.input_model = input_model
        self.handler = handler

    def to_openai_schema(self) -> Dict[str, Any]:
        """
        Convert tool definition to OpenAI-compatible schema.

        Returns:
            Dictionary in OpenAI function calling format
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }
