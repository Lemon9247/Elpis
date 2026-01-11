"""
Integration tests for tool execution.

Tests the complete tool execution flow from end-to-end:
- Tool registration and schema generation
- Tool execution through ToolEngine
- Path safety and validation
- Error handling and recovery
- Concurrent tool execution
"""

import asyncio
from pathlib import Path

import pytest

from elpis.config.settings import Settings
from elpis.tools.tool_engine import ToolEngine
from elpis.utils.exceptions import PathSafetyError


@pytest.mark.integration
class TestToolExecution:
    """Integration tests for tool execution."""

    @pytest.mark.asyncio
    async def test_read_file_integration(self, test_settings: Settings, workspace_dir: Path):
        """Test reading a file through ToolEngine."""
        # Create a test file
        test_file = workspace_dir / "test.txt"
        test_file.write_text("Hello, world!")

        # Initialize ToolEngine
        engine = ToolEngine(str(workspace_dir), test_settings)

        # Execute read_file tool
        tool_call = {
            "id": "call_test_1",
            "function": {"name": "read_file", "arguments": '{"file_path": "test.txt"}'},
        }

        result = await engine.execute_tool_call(tool_call)

        assert result["success"] is True
        assert result["result"]["success"] is True
        assert "Hello, world!" in result["result"]["content"]

    @pytest.mark.asyncio
    async def test_write_file_integration(self, test_settings: Settings, workspace_dir: Path):
        """Test writing a file through ToolEngine."""
        engine = ToolEngine(str(workspace_dir), test_settings)

        # Execute write_file tool
        tool_call = {
            "id": "call_test_2",
            "function": {
                "name": "write_file",
                "arguments": '{"file_path": "output.txt", "content": "Test content"}',
            },
        }

        result = await engine.execute_tool_call(tool_call)

        assert result["success"] is True
        assert result["result"]["success"] is True

        # Verify file was created
        output_file = workspace_dir / "output.txt"
        assert output_file.exists()
        assert output_file.read_text() == "Test content"

    @pytest.mark.asyncio
    async def test_execute_bash_integration(self, test_settings: Settings, workspace_dir: Path):
        """Test executing a bash command through ToolEngine."""
        engine = ToolEngine(str(workspace_dir), test_settings)

        # Execute safe bash command
        tool_call = {
            "id": "call_test_3",
            "function": {"name": "execute_bash", "arguments": '{"command": "echo Hello"}'},
        }

        result = await engine.execute_tool_call(tool_call)

        assert result["success"] is True
        assert result["result"]["success"] is True
        assert "Hello" in result["result"]["stdout"]

    @pytest.mark.asyncio
    async def test_list_directory_integration(
        self, test_settings: Settings, workspace_dir: Path
    ):
        """Test listing directory contents through ToolEngine."""
        # Create test files
        (workspace_dir / "file1.txt").write_text("content1")
        (workspace_dir / "file2.py").write_text("content2")
        subdir = workspace_dir / "subdir"
        subdir.mkdir()
        (subdir / "file3.txt").write_text("content3")

        engine = ToolEngine(str(workspace_dir), test_settings)

        # Execute list_directory tool
        tool_call = {
            "id": "call_test_4",
            "function": {"name": "list_directory", "arguments": '{"dir_path": "."}'},
        }

        result = await engine.execute_tool_call(tool_call)

        assert result["success"] is True
        assert result["result"]["success"] is True
        assert result["result"]["file_count"] == 2  # file1.txt and file2.py
        assert result["result"]["directory_count"] == 1  # subdir

    @pytest.mark.asyncio
    async def test_concurrent_tool_execution(
        self, test_settings: Settings, workspace_dir: Path
    ):
        """Test concurrent execution of multiple tools."""
        # Create test files
        (workspace_dir / "file1.txt").write_text("content1")
        (workspace_dir / "file2.txt").write_text("content2")

        engine = ToolEngine(str(workspace_dir), test_settings)

        # Create multiple tool calls
        tool_calls = [
            {
                "id": "call_1",
                "function": {"name": "read_file", "arguments": '{"file_path": "file1.txt"}'},
            },
            {
                "id": "call_2",
                "function": {"name": "read_file", "arguments": '{"file_path": "file2.txt"}'},
            },
            {
                "id": "call_3",
                "function": {"name": "list_directory", "arguments": '{"dir_path": "."}'},
            },
        ]

        # Execute concurrently
        results = await asyncio.gather(*[engine.execute_tool_call(call) for call in tool_calls])

        # All should succeed
        assert all(r["success"] for r in results)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_path_safety_enforcement(self, test_settings: Settings, workspace_dir: Path):
        """Test that path safety is enforced."""
        engine = ToolEngine(str(workspace_dir), test_settings)

        # Try to read outside workspace
        tool_call = {
            "id": "call_unsafe",
            "function": {"name": "read_file", "arguments": '{"file_path": "../../../etc/passwd"}'},
        }

        result = await engine.execute_tool_call(tool_call)

        # Should fail due to path safety
        assert result["success"] is False
        assert "outside workspace" in result["result"]["error"].lower()

    @pytest.mark.asyncio
    async def test_dangerous_command_blocking(self, test_settings: Settings, workspace_dir: Path):
        """Test that dangerous commands are blocked."""
        engine = ToolEngine(str(workspace_dir), test_settings)

        # Try dangerous command
        tool_call = {
            "id": "call_dangerous",
            "function": {"name": "execute_bash", "arguments": '{"command": "rm -rf /"}'},
        }

        result = await engine.execute_tool_call(tool_call)

        # Should fail due to safety check
        assert result["success"] is False
        assert "dangerous" in result["result"]["error"].lower()

    @pytest.mark.asyncio
    async def test_tool_schemas_generation(self, test_settings: Settings, workspace_dir: Path):
        """Test that tool schemas are correctly generated."""
        engine = ToolEngine(str(workspace_dir), test_settings)

        schemas = engine.get_tool_schemas()

        # Should have all 5 tools
        assert len(schemas) == 5

        tool_names = [schema["function"]["name"] for schema in schemas]
        assert "read_file" in tool_names
        assert "write_file" in tool_names
        assert "execute_bash" in tool_names
        assert "search_codebase" in tool_names
        assert "list_directory" in tool_names

        # Check schema structure
        for schema in schemas:
            assert schema["type"] == "function"
            assert "name" in schema["function"]
            assert "description" in schema["function"]
            assert "parameters" in schema["function"]

    @pytest.mark.asyncio
    async def test_file_size_limit(self, test_settings: Settings, workspace_dir: Path):
        """Test that file size limits are enforced."""
        # Create a file larger than the limit
        large_file = workspace_dir / "large.txt"
        large_file.write_text("x" * (test_settings.tools.max_file_size + 1))

        engine = ToolEngine(str(workspace_dir), test_settings)

        tool_call = {
            "id": "call_large",
            "function": {"name": "read_file", "arguments": '{"file_path": "large.txt"}'},
        }

        result = await engine.execute_tool_call(tool_call)

        # Should fail due to size limit
        assert result["success"] is False
        assert "too large" in result["result"]["error"].lower()

    @pytest.mark.asyncio
    async def test_write_file_creates_directories(
        self, test_settings: Settings, workspace_dir: Path
    ):
        """Test that write_file creates parent directories."""
        engine = ToolEngine(str(workspace_dir), test_settings)

        tool_call = {
            "id": "call_nested",
            "function": {
                "name": "write_file",
                "arguments": '{"file_path": "a/b/c/nested.txt", "content": "nested"}',
            },
        }

        result = await engine.execute_tool_call(tool_call)

        assert result["success"] is True

        # Verify nested file exists
        nested_file = workspace_dir / "a" / "b" / "c" / "nested.txt"
        assert nested_file.exists()
        assert nested_file.read_text() == "nested"

    @pytest.mark.asyncio
    async def test_write_file_creates_backup(self, test_settings: Settings, workspace_dir: Path):
        """Test that write_file creates backup of existing files."""
        # Create original file
        original_file = workspace_dir / "backup_test.txt"
        original_file.write_text("original content")

        engine = ToolEngine(str(workspace_dir), test_settings)

        # Overwrite file
        tool_call = {
            "id": "call_overwrite",
            "function": {
                "name": "write_file",
                "arguments": '{"file_path": "backup_test.txt", "content": "new content"}',
            },
        }

        result = await engine.execute_tool_call(tool_call)

        assert result["success"] is True

        # Verify backup was created
        backup_file = workspace_dir / "backup_test.txt.bak"
        assert backup_file.exists()
        assert backup_file.read_text() == "original content"

        # Verify new content
        assert original_file.read_text() == "new content"

    @pytest.mark.asyncio
    async def test_bash_timeout(self, test_settings: Settings, workspace_dir: Path):
        """Test that bash commands timeout correctly."""
        engine = ToolEngine(str(workspace_dir), test_settings)

        # Command that will timeout
        tool_call = {
            "id": "call_timeout",
            "function": {"name": "execute_bash", "arguments": '{"command": "sleep 10"}'},
        }

        result = await engine.execute_tool_call(tool_call)

        # Should fail due to timeout (max_bash_timeout is 5 seconds in test settings)
        assert result["success"] is False
        assert "timeout" in result["result"]["error"].lower()

    @pytest.mark.asyncio
    async def test_invalid_tool_name(self, test_settings: Settings, workspace_dir: Path):
        """Test handling of invalid tool names."""
        engine = ToolEngine(str(workspace_dir), test_settings)

        tool_call = {
            "id": "call_invalid",
            "function": {"name": "nonexistent_tool", "arguments": "{}"},
        }

        result = await engine.execute_tool_call(tool_call)

        assert result["success"] is False
        assert "unknown tool" in result["result"]["error"].lower()

    @pytest.mark.asyncio
    async def test_invalid_json_arguments(self, test_settings: Settings, workspace_dir: Path):
        """Test handling of invalid JSON in tool arguments."""
        engine = ToolEngine(str(workspace_dir), test_settings)

        tool_call = {
            "id": "call_bad_json",
            "function": {"name": "read_file", "arguments": "not valid json"},
        }

        result = await engine.execute_tool_call(tool_call)

        assert result["success"] is False
        assert "json" in result["result"]["error"].lower()

    @pytest.mark.asyncio
    async def test_list_directory_recursive(self, test_settings: Settings, workspace_dir: Path):
        """Test recursive directory listing."""
        # Create nested structure
        (workspace_dir / "level1").mkdir()
        (workspace_dir / "level1" / "file1.txt").write_text("content")
        (workspace_dir / "level1" / "level2").mkdir()
        (workspace_dir / "level1" / "level2" / "file2.txt").write_text("content")

        engine = ToolEngine(str(workspace_dir), test_settings)

        tool_call = {
            "id": "call_recursive",
            "function": {
                "name": "list_directory",
                "arguments": '{"dir_path": ".", "recursive": true}',
            },
        }

        result = await engine.execute_tool_call(tool_call)

        assert result["success"] is True
        # Should find files in nested directories
        all_files = result["result"]["files"]
        file_paths = [f["path"] for f in all_files]
        assert any("level2/file2.txt" in path or "level2\\file2.txt" in path for path in file_paths)
