"""Unit tests for tool definitions and input models."""
import pytest
from pydantic import ValidationError

from elpis.tools.tool_definitions import (
    ToolInput,
    ReadFileInput,
    WriteFileInput,
    ExecuteBashInput,
    SearchCodebaseInput,
    ListDirectoryInput,
    ToolDefinition,
)


class TestToolInput:
    """Test base ToolInput class."""

    def test_base_tool_input_config(self):
        """Test ToolInput configuration."""
        assert ToolInput.model_config.get('extra') == 'forbid'


class TestReadFileInput:
    """Test ReadFileInput model."""

    def test_valid_input(self):
        """Test valid read file input."""
        input_data = ReadFileInput(file_path='test.txt', max_lines=1000)
        assert input_data.file_path == 'test.txt'
        assert input_data.max_lines == 1000

    def test_default_max_lines(self):
        """Test default max_lines value."""
        input_data = ReadFileInput(file_path='test.txt')
        assert input_data.max_lines == 2000

    def test_null_byte_in_path(self):
        """Test that null bytes in path are rejected."""
        with pytest.raises(ValidationError):
            ReadFileInput(file_path='test\x00.txt')

    def test_empty_path(self):
        """Test that empty path is rejected."""
        with pytest.raises(ValidationError):
            ReadFileInput(file_path='')

    def test_whitespace_only_path(self):
        """Test that whitespace-only path is rejected."""
        with pytest.raises(ValidationError):
            ReadFileInput(file_path='   ')

    def test_max_lines_out_of_range(self):
        """Test that max_lines outside valid range is rejected."""
        with pytest.raises(ValidationError):
            ReadFileInput(file_path='test.txt', max_lines=0)

        with pytest.raises(ValidationError):
            ReadFileInput(file_path='test.txt', max_lines=100001)


class TestWriteFileInput:
    """Test WriteFileInput model."""

    def test_valid_input(self):
        """Test valid write file input."""
        input_data = WriteFileInput(
            file_path='test.txt',
            content='Hello, world!',
            create_dirs=False
        )
        assert input_data.file_path == 'test.txt'
        assert input_data.content == 'Hello, world!'
        assert input_data.create_dirs is False

    def test_default_create_dirs(self):
        """Test default create_dirs value."""
        input_data = WriteFileInput(file_path='test.txt', content='content')
        assert input_data.create_dirs is True

    def test_null_byte_in_path(self):
        """Test that null bytes in path are rejected."""
        with pytest.raises(ValidationError):
            WriteFileInput(file_path='test\x00.txt', content='content')

    def test_empty_path(self):
        """Test that empty path is rejected."""
        with pytest.raises(ValidationError):
            WriteFileInput(file_path='', content='content')

    def test_empty_content(self):
        """Test that empty content is allowed."""
        input_data = WriteFileInput(file_path='test.txt', content='')
        assert input_data.content == ''


class TestExecuteBashInput:
    """Test ExecuteBashInput model."""

    def test_valid_input(self):
        """Test valid bash command input."""
        input_data = ExecuteBashInput(command='ls -la')
        assert input_data.command == 'ls -la'

    def test_empty_command(self):
        """Test that empty command is rejected."""
        with pytest.raises(ValidationError):
            ExecuteBashInput(command='')

    def test_whitespace_only_command(self):
        """Test that whitespace-only command is rejected."""
        with pytest.raises(ValidationError):
            ExecuteBashInput(command='   ')

    def test_long_command(self):
        """Test that very long commands are rejected."""
        long_command = 'a' * 10001
        with pytest.raises(ValidationError):
            ExecuteBashInput(command=long_command)

    def test_max_length_command(self):
        """Test command at max length is accepted."""
        max_command = 'a' * 10000
        input_data = ExecuteBashInput(command=max_command)
        assert len(input_data.command) == 10000


class TestSearchCodebaseInput:
    """Test SearchCodebaseInput model."""

    def test_valid_input(self):
        """Test valid search input."""
        input_data = SearchCodebaseInput(
            pattern='test.*pattern',
            file_glob='*.py',
            context_lines=3
        )
        assert input_data.pattern == 'test.*pattern'
        assert input_data.file_glob == '*.py'
        assert input_data.context_lines == 3

    def test_default_values(self):
        """Test default values."""
        input_data = SearchCodebaseInput(pattern='test')
        assert input_data.pattern == 'test'
        assert input_data.file_glob is None
        assert input_data.context_lines == 0

    def test_empty_pattern(self):
        """Test that empty pattern is rejected."""
        with pytest.raises(ValidationError):
            SearchCodebaseInput(pattern='')

    def test_whitespace_only_pattern(self):
        """Test that whitespace-only pattern is rejected."""
        with pytest.raises(ValidationError):
            SearchCodebaseInput(pattern='   ')

    def test_context_lines_out_of_range(self):
        """Test that context_lines outside valid range is rejected."""
        with pytest.raises(ValidationError):
            SearchCodebaseInput(pattern='test', context_lines=-1)

        with pytest.raises(ValidationError):
            SearchCodebaseInput(pattern='test', context_lines=11)


class TestListDirectoryInput:
    """Test ListDirectoryInput model."""

    def test_valid_input(self):
        """Test valid list directory input."""
        input_data = ListDirectoryInput(
            dir_path='src',
            recursive=True,
            pattern='*.py'
        )
        assert input_data.dir_path == 'src'
        assert input_data.recursive is True
        assert input_data.pattern == '*.py'

    def test_default_values(self):
        """Test default values."""
        input_data = ListDirectoryInput()
        assert input_data.dir_path == '.'
        assert input_data.recursive is False
        assert input_data.pattern is None

    def test_null_byte_in_path(self):
        """Test that null bytes in path are rejected."""
        with pytest.raises(ValidationError):
            ListDirectoryInput(dir_path='test\x00')


class TestToolDefinition:
    """Test ToolDefinition class."""

    def test_tool_definition_creation(self):
        """Test creating a tool definition."""
        async def dummy_handler(**kwargs):
            return {'success': True}

        tool_def = ToolDefinition(
            name='test_tool',
            description='A test tool',
            parameters={'type': 'object', 'properties': {}},
            input_model=ToolInput,
            handler=dummy_handler
        )

        assert tool_def.name == 'test_tool'
        assert tool_def.description == 'A test tool'
        assert tool_def.input_model == ToolInput
        assert tool_def.handler == dummy_handler

    def test_to_openai_schema(self):
        """Test conversion to OpenAI schema format."""
        async def dummy_handler(**kwargs):
            return {'success': True}

        parameters = {
            'type': 'object',
            'properties': {
                'arg1': {'type': 'string', 'description': 'First argument'}
            },
            'required': ['arg1']
        }

        tool_def = ToolDefinition(
            name='test_tool',
            description='A test tool',
            parameters=parameters,
            input_model=ToolInput,
            handler=dummy_handler
        )

        schema = tool_def.to_openai_schema()

        assert schema['type'] == 'function'
        assert schema['function']['name'] == 'test_tool'
        assert schema['function']['description'] == 'A test tool'
        assert schema['function']['parameters'] == parameters
