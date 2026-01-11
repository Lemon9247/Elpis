"""Tool definitions with Pydantic models for input validation."""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Callable, Dict, Any, Type


class ToolInput(BaseModel):
    """Base class for all tool inputs."""

    class Config:
        validate_assignment = True
        extra = "forbid"  # Prevent extra fields


class ReadFileInput(ToolInput):
    """Input model for read_file tool."""
    file_path: str = Field(description="Path to file (relative to workspace or absolute)")
    max_lines: Optional[int] = Field(
        default=2000,
        ge=1,
        le=100000,
        description="Maximum number of lines to read"
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


class WriteFileInput(ToolInput):
    """Input model for write_file tool."""
    file_path: str = Field(description="Path to file (relative to workspace or absolute)")
    content: str = Field(description="Content to write to file")
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
