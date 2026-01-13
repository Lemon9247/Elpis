"""Unit tests for file tools."""
import pytest
from pathlib import Path

from psyche.tools.tool_engine import ToolSettings
from psyche.tools.implementations.file_tools import FileTools, PathSafetyError


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
def file_tools(workspace_dir, settings):
    """Create FileTools instance."""
    return FileTools(workspace_dir, settings)


class TestFileToolsPathSanitization:
    """Test path sanitization and validation."""

    def test_sanitize_relative_path(self, file_tools, workspace_dir):
        """Test sanitizing a relative path."""
        result = file_tools._sanitize_path('test.txt')
        assert result == workspace_dir / 'test.txt'

    def test_sanitize_nested_relative_path(self, file_tools, workspace_dir):
        """Test sanitizing a nested relative path."""
        result = file_tools._sanitize_path('subdir/test.txt')
        assert result == workspace_dir / 'subdir' / 'test.txt'

    def test_sanitize_absolute_path_within_workspace(self, file_tools, workspace_dir):
        """Test sanitizing an absolute path within workspace."""
        abs_path = str(workspace_dir / 'test.txt')
        result = file_tools._sanitize_path(abs_path)
        assert result == workspace_dir / 'test.txt'

    def test_sanitize_path_outside_workspace_raises(self, file_tools):
        """Test that paths outside workspace raise PathSafetyError."""
        with pytest.raises(PathSafetyError):
            file_tools._sanitize_path('/etc/passwd')

    def test_sanitize_path_with_parent_references_outside_workspace(self, file_tools):
        """Test that parent references escaping workspace raise error."""
        with pytest.raises(PathSafetyError):
            file_tools._sanitize_path('../../../etc/passwd')


class TestReadFile:
    """Test read_file functionality."""

    @pytest.mark.asyncio
    async def test_read_file_success(self, file_tools, workspace_dir):
        """Test successful file read."""
        # Create test file
        test_file = workspace_dir / 'test.txt'
        test_content = 'Hello, World!\nLine 2\nLine 3'
        test_file.write_text(test_content)

        # Read file
        result = await file_tools.read_file('test.txt')

        assert result['success'] is True
        assert result['content'] == test_content
        assert result['line_count'] == 3
        assert result['total_lines'] == 3
        assert result['truncated'] is False

    @pytest.mark.asyncio
    async def test_read_file_with_max_lines(self, file_tools, workspace_dir):
        """Test reading file with max_lines limit."""
        # Create test file with multiple lines
        test_file = workspace_dir / 'test.txt'
        lines = [f'Line {i}\n' for i in range(100)]
        test_file.write_text(''.join(lines))

        # Read with limit
        result = await file_tools.read_file('test.txt', max_lines=10)

        assert result['success'] is True
        assert result['line_count'] == 10
        assert result['total_lines'] == 100
        assert result['truncated'] is True

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self, file_tools):
        """Test reading nonexistent file returns error."""
        result = await file_tools.read_file('nonexistent.txt')

        assert result['success'] is False
        assert 'not found' in result['error'].lower()

    @pytest.mark.asyncio
    async def test_read_directory_returns_error(self, file_tools, workspace_dir):
        """Test reading a directory returns error."""
        # Create directory
        test_dir = workspace_dir / 'testdir'
        test_dir.mkdir()

        result = await file_tools.read_file('testdir')

        assert result['success'] is False
        assert 'not a file' in result['error'].lower()

    @pytest.mark.asyncio
    async def test_read_file_outside_workspace(self, file_tools):
        """Test reading file outside workspace returns error."""
        result = await file_tools.read_file('/etc/passwd')

        assert result['success'] is False
        assert 'outside workspace' in result['error'].lower()

    @pytest.mark.asyncio
    async def test_read_file_too_large(self, file_tools, workspace_dir, settings):
        """Test reading file larger than max size."""
        # Create large file
        test_file = workspace_dir / 'large.txt'
        large_content = 'x' * (settings.max_file_size + 1000)
        test_file.write_text(large_content)

        result = await file_tools.read_file('large.txt')

        assert result['success'] is False
        assert 'too large' in result['error'].lower()


class TestWriteFile:
    """Test write_file functionality."""

    @pytest.mark.asyncio
    async def test_write_file_success(self, file_tools, workspace_dir):
        """Test successful file write."""
        result = await file_tools.write_file('test.txt', 'Hello, World!')

        assert result['success'] is True
        assert result['size_bytes'] > 0

        # Verify file was created
        test_file = workspace_dir / 'test.txt'
        assert test_file.exists()
        assert test_file.read_text() == 'Hello, World!'

    @pytest.mark.asyncio
    async def test_write_file_creates_directories(self, file_tools, workspace_dir):
        """Test writing file with create_dirs creates parent directories."""
        result = await file_tools.write_file(
            'subdir/nested/test.txt',
            'content',
            create_dirs=True
        )

        assert result['success'] is True

        # Verify directories and file were created
        test_file = workspace_dir / 'subdir' / 'nested' / 'test.txt'
        assert test_file.exists()
        assert test_file.read_text() == 'content'

    @pytest.mark.asyncio
    async def test_write_file_without_create_dirs(self, file_tools, workspace_dir):
        """Test writing file without create_dirs fails if parent doesn't exist."""
        result = await file_tools.write_file(
            'nonexistent/test.txt',
            'content',
            create_dirs=False
        )

        assert result['success'] is False

    @pytest.mark.asyncio
    async def test_write_file_creates_backup(self, file_tools, workspace_dir):
        """Test writing file creates backup of existing file."""
        # Create original file
        test_file = workspace_dir / 'test.txt'
        test_file.write_text('original content')

        # Overwrite file
        result = await file_tools.write_file('test.txt', 'new content')

        assert result['success'] is True
        assert 'backup_created' in result

        # Verify new content
        assert test_file.read_text() == 'new content'

        # Verify backup exists
        backup_file = workspace_dir / 'test.txt.bak'
        assert backup_file.exists()
        assert backup_file.read_text() == 'original content'

    @pytest.mark.asyncio
    async def test_write_file_outside_workspace(self, file_tools):
        """Test writing file outside workspace returns error."""
        result = await file_tools.write_file('/tmp/test.txt', 'content')

        assert result['success'] is False
        assert 'outside workspace' in result['error'].lower()

    @pytest.mark.asyncio
    async def test_write_empty_content(self, file_tools, workspace_dir):
        """Test writing empty content is allowed."""
        result = await file_tools.write_file('empty.txt', '')

        assert result['success'] is True

        test_file = workspace_dir / 'empty.txt'
        assert test_file.exists()
        assert test_file.read_text() == ''

    @pytest.mark.asyncio
    async def test_write_file_unicode_content(self, file_tools, workspace_dir):
        """Test writing file with unicode content."""
        unicode_content = 'Hello 世界 🌍'
        result = await file_tools.write_file('unicode.txt', unicode_content)

        assert result['success'] is True

        test_file = workspace_dir / 'unicode.txt'
        assert test_file.read_text() == unicode_content
