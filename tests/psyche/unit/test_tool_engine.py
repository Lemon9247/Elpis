"""Unit tests for tool engine."""
import pytest
import json
from pathlib import Path

from psyche.tools.tool_engine import ToolEngine, ToolSettings


@pytest.fixture
def settings():
    """Create test settings."""
    return ToolSettings()


@pytest.fixture
def workspace_dir(tmp_path):
    """Create a temporary workspace directory."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


@pytest.fixture
def tool_engine(workspace_dir, settings):
    """Create ToolEngine instance."""
    return ToolEngine(str(workspace_dir), settings)


class TestToolEngineInitialization:
    """Test ToolEngine initialization."""

    def test_engine_initializes(self, tool_engine):
        """Test that engine initializes successfully."""
        assert tool_engine is not None
        assert tool_engine.tools is not None
        assert len(tool_engine.tools) > 0

    def test_workspace_created(self, workspace_dir):
        """Test that workspace directory is created if it doesn't exist."""
        # Create engine with non-existent workspace
        new_workspace = workspace_dir / "new_workspace"
        settings = ToolSettings()

        engine = ToolEngine(str(new_workspace), settings)

        assert new_workspace.exists()
        assert new_workspace.is_dir()

    def test_all_tools_registered(self, tool_engine):
        """Test that all expected tools are registered."""
        expected_tools = [
            'read_file',
            'create_file',
            'edit_file',
            'execute_bash',
            'search_codebase',
            'list_directory'
        ]

        for tool_name in expected_tools:
            assert tool_name in tool_engine.tools


class TestGetToolSchemas:
    """Test get_tool_schemas method."""

    def test_get_tool_schemas(self, tool_engine):
        """Test getting tool schemas."""
        schemas = tool_engine.get_tool_schemas()

        assert isinstance(schemas, list)
        assert len(schemas) == 6  # We have 6 tools

        # Check schema format
        for schema in schemas:
            assert 'type' in schema
            assert schema['type'] == 'function'
            assert 'function' in schema
            assert 'name' in schema['function']
            assert 'description' in schema['function']
            assert 'parameters' in schema['function']

    def test_schema_parameters_format(self, tool_engine):
        """Test that parameters follow JSON schema format."""
        schemas = tool_engine.get_tool_schemas()

        for schema in schemas:
            params = schema['function']['parameters']
            assert params['type'] == 'object'
            assert 'properties' in params
            assert 'required' in params


class TestExecuteToolCall:
    """Test execute_tool_call method."""

    @pytest.mark.asyncio
    async def test_execute_read_file(self, tool_engine, workspace_dir):
        """Test executing read_file tool."""
        # Create test file
        test_file = workspace_dir / 'test.txt'
        test_file.write_text('Hello, World!')

        # Execute tool call
        tool_call = {
            'id': 'call_123',
            'type': 'function',
            'function': {
                'name': 'read_file',
                'arguments': json.dumps({'file_path': 'test.txt'})
            }
        }

        result = await tool_engine.execute_tool_call(tool_call)

        assert result['success'] is True
        assert result['tool_call_id'] == 'call_123'
        assert result['result']['content'] == 'Hello, World!'
        assert 'duration_ms' in result

    @pytest.mark.asyncio
    async def test_execute_create_file(self, tool_engine, workspace_dir):
        """Test executing create_file tool."""
        tool_call = {
            'id': 'call_456',
            'type': 'function',
            'function': {
                'name': 'create_file',
                'arguments': json.dumps({
                    'file_path': 'output.txt',
                    'content': 'Test content'
                })
            }
        }

        result = await tool_engine.execute_tool_call(tool_call)

        assert result['success'] is True
        assert result['tool_call_id'] == 'call_456'

        # Verify file was created
        output_file = workspace_dir / 'output.txt'
        assert output_file.exists()
        assert output_file.read_text() == 'Test content'

    @pytest.mark.asyncio
    async def test_execute_edit_file(self, tool_engine, workspace_dir):
        """Test executing edit_file tool."""
        # Create initial file
        test_file = workspace_dir / 'to_edit.txt'
        test_file.write_text('Hello, World!')

        tool_call = {
            'id': 'call_edit',
            'type': 'function',
            'function': {
                'name': 'edit_file',
                'arguments': json.dumps({
                    'file_path': 'to_edit.txt',
                    'old_string': 'World',
                    'new_string': 'Universe'
                })
            }
        }

        result = await tool_engine.execute_tool_call(tool_call)

        assert result['success'] is True
        assert result['tool_call_id'] == 'call_edit'

        # Verify file was edited
        assert test_file.read_text() == 'Hello, Universe!'

        # Verify backup was created
        backup_file = workspace_dir / 'to_edit.txt.bak'
        assert backup_file.exists()
        assert backup_file.read_text() == 'Hello, World!'

    @pytest.mark.asyncio
    async def test_execute_bash_command(self, tool_engine):
        """Test executing bash command."""
        tool_call = {
            'id': 'call_789',
            'type': 'function',
            'function': {
                'name': 'execute_bash',
                'arguments': json.dumps({
                    'command': 'echo "Hello from bash"'
                })
            }
        }

        result = await tool_engine.execute_tool_call(tool_call)

        assert result['success'] is True
        assert 'Hello from bash' in result['result']['stdout']

    @pytest.mark.asyncio
    async def test_execute_list_directory(self, tool_engine, workspace_dir):
        """Test executing list_directory tool."""
        # Create some test files
        (workspace_dir / 'file1.txt').write_text('content1')
        (workspace_dir / 'file2.txt').write_text('content2')

        tool_call = {
            'id': 'call_list',
            'type': 'function',
            'function': {
                'name': 'list_directory',
                'arguments': json.dumps({'dir_path': '.'})
            }
        }

        result = await tool_engine.execute_tool_call(tool_call)

        assert result['success'] is True
        assert result['result']['file_count'] >= 2

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, tool_engine):
        """Test executing unknown tool returns error."""
        tool_call = {
            'id': 'call_unknown',
            'type': 'function',
            'function': {
                'name': 'nonexistent_tool',
                'arguments': '{}'
            }
        }

        result = await tool_engine.execute_tool_call(tool_call)

        assert result['success'] is False
        assert 'unknown tool' in result['result']['error'].lower()

    @pytest.mark.asyncio
    async def test_execute_with_invalid_json(self, tool_engine):
        """Test executing tool with invalid JSON arguments."""
        tool_call = {
            'id': 'call_invalid',
            'type': 'function',
            'function': {
                'name': 'read_file',
                'arguments': 'invalid json {'
            }
        }

        result = await tool_engine.execute_tool_call(tool_call)

        assert result['success'] is False
        assert 'json' in result['result']['error'].lower()

    @pytest.mark.asyncio
    async def test_execute_with_invalid_arguments(self, tool_engine):
        """Test executing tool with invalid arguments."""
        tool_call = {
            'id': 'call_invalid_args',
            'type': 'function',
            'function': {
                'name': 'read_file',
                'arguments': json.dumps({'file_path': ''})  # Empty path is invalid
            }
        }

        result = await tool_engine.execute_tool_call(tool_call)

        assert result['success'] is False
        assert 'error' in result['result']

    @pytest.mark.asyncio
    async def test_execute_with_dict_arguments(self, tool_engine, workspace_dir):
        """Test executing tool with dict arguments (not JSON string)."""
        # Create test file
        test_file = workspace_dir / 'test.txt'
        test_file.write_text('Content')

        tool_call = {
            'id': 'call_dict',
            'type': 'function',
            'function': {
                'name': 'read_file',
                'arguments': {'file_path': 'test.txt'}  # Dict instead of JSON string
            }
        }

        result = await tool_engine.execute_tool_call(tool_call)

        assert result['success'] is True


class TestExecuteMultipleToolCalls:
    """Test execute_multiple_tool_calls method."""

    @pytest.mark.asyncio
    async def test_execute_multiple_tools(self, tool_engine, workspace_dir):
        """Test executing multiple tool calls concurrently."""
        # Create test file
        (workspace_dir / 'test.txt').write_text('Test content')

        tool_calls = [
            {
                'id': 'call_1',
                'type': 'function',
                'function': {
                    'name': 'read_file',
                    'arguments': json.dumps({'file_path': 'test.txt'})
                }
            },
            {
                'id': 'call_2',
                'type': 'function',
                'function': {
                    'name': 'list_directory',
                    'arguments': json.dumps({'dir_path': '.'})
                }
            },
            {
                'id': 'call_3',
                'type': 'function',
                'function': {
                    'name': 'execute_bash',
                    'arguments': json.dumps({'command': 'echo "test"'})
                }
            }
        ]

        results = await tool_engine.execute_multiple_tool_calls(tool_calls)

        assert len(results) == 3
        assert all(result['success'] for result in results)

    @pytest.mark.asyncio
    async def test_execute_multiple_with_failures(self, tool_engine):
        """Test that failures in one tool don't affect others."""
        tool_calls = [
            {
                'id': 'call_success',
                'type': 'function',
                'function': {
                    'name': 'execute_bash',
                    'arguments': json.dumps({'command': 'echo "success"'})
                }
            },
            {
                'id': 'call_fail',
                'type': 'function',
                'function': {
                    'name': 'read_file',
                    'arguments': json.dumps({'file_path': 'nonexistent.txt'})
                }
            }
        ]

        results = await tool_engine.execute_multiple_tool_calls(tool_calls)

        assert len(results) == 2
        assert results[0]['success'] is True
        assert results[1]['success'] is False


class TestSanitizePath:
    """Test sanitize_path method."""

    def test_sanitize_relative_path(self, tool_engine, workspace_dir):
        """Test sanitizing relative path."""
        result = tool_engine.sanitize_path('test.txt')
        assert result == workspace_dir / 'test.txt'

    def test_sanitize_absolute_path_within_workspace(self, tool_engine, workspace_dir):
        """Test sanitizing absolute path within workspace."""
        abs_path = str(workspace_dir / 'test.txt')
        result = tool_engine.sanitize_path(abs_path)
        assert result == workspace_dir / 'test.txt'

    def test_sanitize_path_outside_workspace_raises(self, tool_engine):
        """Test that paths outside workspace raise error."""
        from psyche.tools.tool_engine import ToolExecutionError

        with pytest.raises(ToolExecutionError):
            tool_engine.sanitize_path('/etc/passwd')
