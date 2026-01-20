"""Unit tests for directory tool."""
import pytest
from pathlib import Path

from hermes.tools.tool_engine import ToolSettings
from hermes.tools.implementations.directory_tool import DirectoryTool, PathSafetyError


@pytest.fixture
def settings():
    """Create test settings."""
    return ToolSettings()


@pytest.fixture
def workspace_dir(tmp_path):
    """Create a temporary workspace directory with test structure."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # Create test files in root
    (workspace / "file1.txt").write_text("content1")
    (workspace / "file2.py").write_text("content2")
    (workspace / "file3.md").write_text("content3")

    # Create subdirectories
    subdir1 = workspace / "subdir1"
    subdir1.mkdir()
    (subdir1 / "nested1.txt").write_text("nested content 1")
    (subdir1 / "nested2.py").write_text("nested content 2")

    subdir2 = workspace / "subdir2"
    subdir2.mkdir()
    (subdir2 / "nested3.txt").write_text("nested content 3")

    # Create nested subdirectory
    nested = subdir1 / "deep"
    nested.mkdir()
    (nested / "deep_file.txt").write_text("deep content")

    return workspace


@pytest.fixture
def directory_tool(workspace_dir, settings):
    """Create DirectoryTool instance."""
    return DirectoryTool(workspace_dir, settings)


class TestDirectoryToolPathSanitization:
    """Test path sanitization and validation."""

    def test_sanitize_current_directory(self, directory_tool, workspace_dir):
        """Test sanitizing current directory path."""
        result = directory_tool._sanitize_path('.')
        assert result == workspace_dir

    def test_sanitize_relative_path(self, directory_tool, workspace_dir):
        """Test sanitizing relative path."""
        result = directory_tool._sanitize_path('subdir1')
        assert result == workspace_dir / 'subdir1'

    def test_sanitize_path_outside_workspace_raises(self, directory_tool):
        """Test that paths outside workspace raise PathSafetyError."""
        with pytest.raises(PathSafetyError):
            directory_tool._sanitize_path('/etc')


class TestListDirectory:
    """Test list_directory functionality."""

    @pytest.mark.asyncio
    async def test_list_root_directory(self, directory_tool):
        """Test listing root workspace directory."""
        result = await directory_tool.list_directory('.')

        assert result['success'] is True
        assert result['file_count'] == 3  # file1.txt, file2.py, file3.md
        assert result['directory_count'] == 2  # subdir1, subdir2
        assert result['total_items'] == 5

        # Check that files are present
        file_names = [f['name'] for f in result['files']]
        assert 'file1.txt' in file_names
        assert 'file2.py' in file_names
        assert 'file3.md' in file_names

        # Check that directories are present
        dir_names = [d['name'] for d in result['directories']]
        assert 'subdir1' in dir_names
        assert 'subdir2' in dir_names

    @pytest.mark.asyncio
    async def test_list_subdirectory(self, directory_tool):
        """Test listing a subdirectory."""
        result = await directory_tool.list_directory('subdir1')

        assert result['success'] is True
        assert result['file_count'] == 2  # nested1.txt, nested2.py
        assert result['directory_count'] == 1  # deep

    @pytest.mark.asyncio
    async def test_list_recursive(self, directory_tool):
        """Test recursive directory listing."""
        result = await directory_tool.list_directory('.', recursive=True)

        assert result['success'] is True
        assert result['recursive'] is True

        # Should include all files recursively
        file_names = [f['name'] for f in result['files']]
        assert 'deep_file.txt' in file_names
        assert 'nested1.txt' in file_names

    @pytest.mark.asyncio
    async def test_list_with_pattern(self, directory_tool):
        """Test listing with glob pattern filter."""
        result = await directory_tool.list_directory('.', pattern='*.py')

        assert result['success'] is True
        # Should only include .py files
        file_names = [f['name'] for f in result['files']]
        assert all(name.endswith('.py') for name in file_names)

    @pytest.mark.asyncio
    async def test_list_recursive_with_pattern(self, directory_tool):
        """Test recursive listing with pattern."""
        result = await directory_tool.list_directory('.', recursive=True, pattern='*.txt')

        assert result['success'] is True
        # Should include all .txt files recursively
        file_names = [f['name'] for f in result['files']]
        assert all(name.endswith('.txt') for name in file_names)
        assert 'deep_file.txt' in file_names

    @pytest.mark.asyncio
    async def test_list_nonexistent_directory(self, directory_tool):
        """Test listing nonexistent directory returns error."""
        result = await directory_tool.list_directory('nonexistent')

        assert result['success'] is False
        assert 'not found' in result['error'].lower()

    @pytest.mark.asyncio
    async def test_list_file_instead_of_directory(self, directory_tool):
        """Test listing a file instead of directory returns error."""
        result = await directory_tool.list_directory('file1.txt')

        assert result['success'] is False
        assert 'not a directory' in result['error'].lower()

    @pytest.mark.asyncio
    async def test_list_directory_outside_workspace(self, directory_tool):
        """Test listing directory outside workspace returns error."""
        result = await directory_tool.list_directory('/etc')

        assert result['success'] is False
        assert 'outside workspace' in result['error'].lower()

    @pytest.mark.asyncio
    async def test_list_empty_directory(self, directory_tool, workspace_dir):
        """Test listing empty directory."""
        empty_dir = workspace_dir / 'empty'
        empty_dir.mkdir()

        result = await directory_tool.list_directory('empty')

        assert result['success'] is True
        assert result['file_count'] == 0
        assert result['directory_count'] == 0
        assert result['total_items'] == 0

    @pytest.mark.asyncio
    async def test_file_entries_have_size(self, directory_tool):
        """Test that file entries include size information."""
        result = await directory_tool.list_directory('.')

        assert result['success'] is True
        # Check that files have size field
        for file_entry in result['files']:
            assert 'size' in file_entry
            assert isinstance(file_entry['size'], int)
            assert file_entry['size'] >= 0

    @pytest.mark.asyncio
    async def test_entries_have_type_field(self, directory_tool):
        """Test that all entries have type field."""
        result = await directory_tool.list_directory('.')

        assert result['success'] is True

        # Check files have type='file'
        for file_entry in result['files']:
            assert file_entry['type'] == 'file'

        # Check directories have type='directory'
        for dir_entry in result['directories']:
            assert dir_entry['type'] == 'directory'

    @pytest.mark.asyncio
    async def test_entries_sorted_by_name(self, directory_tool):
        """Test that entries are sorted by name."""
        result = await directory_tool.list_directory('.')

        assert result['success'] is True

        # Check files are sorted
        file_names = [f['name'] for f in result['files']]
        assert file_names == sorted(file_names)

        # Check directories are sorted
        dir_names = [d['name'] for d in result['directories']]
        assert dir_names == sorted(dir_names)
