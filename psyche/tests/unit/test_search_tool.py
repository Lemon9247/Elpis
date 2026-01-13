"""Unit tests for search tool."""
import pytest
import shutil
from pathlib import Path

from psyche.tools.tool_engine import ToolSettings
from psyche.tools.implementations.search_tool import SearchTool


@pytest.fixture
def settings():
    """Create test settings."""
    return ToolSettings()


@pytest.fixture
def workspace_dir(tmp_path):
    """Create a temporary workspace directory with test files."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # Create test files
    (workspace / "test1.py").write_text("def hello():\n    print('hello')\n")
    (workspace / "test2.py").write_text("def world():\n    print('world')\n")
    (workspace / "test.txt").write_text("This is a test file\n")

    # Create subdirectory with files
    subdir = workspace / "subdir"
    subdir.mkdir()
    (subdir / "nested.py").write_text("def nested():\n    print('nested')\n")

    return workspace


@pytest.fixture
def search_tool(workspace_dir, settings):
    """Create SearchTool instance."""
    return SearchTool(workspace_dir, settings)


@pytest.mark.skipif(not shutil.which('rg'), reason="ripgrep not installed")
class TestSearchCodebase:
    """Test search_codebase functionality (requires ripgrep)."""

    @pytest.mark.asyncio
    async def test_search_simple_pattern(self, search_tool):
        """Test searching for a simple pattern."""
        result = await search_tool.search_codebase('hello')

        assert result['success'] is True
        assert result['match_count'] > 0
        assert result['pattern'] == 'hello'
        assert any('hello' in match.get('line', '').lower() for match in result['matches'])

    @pytest.mark.asyncio
    async def test_search_with_file_glob(self, search_tool):
        """Test searching with file glob pattern."""
        result = await search_tool.search_codebase('def', file_glob='*.py')

        assert result['success'] is True
        assert result['match_count'] > 0
        # All matches should be from .py files
        assert all(match['file'].endswith('.py') for match in result['matches'])

    @pytest.mark.asyncio
    async def test_search_no_matches(self, search_tool):
        """Test searching for pattern with no matches."""
        result = await search_tool.search_codebase('nonexistent_pattern_xyz')

        assert result['success'] is True
        assert result['match_count'] == 0
        assert result['matches'] == []

    @pytest.mark.asyncio
    async def test_search_with_context_lines(self, search_tool):
        """Test searching with context lines."""
        result = await search_tool.search_codebase('hello', context_lines=1)

        assert result['success'] is True
        # Context lines are included in ripgrep output

    @pytest.mark.asyncio
    async def test_search_regex_pattern(self, search_tool):
        """Test searching with regex pattern."""
        result = await search_tool.search_codebase('def \\w+\\(\\)')

        assert result['success'] is True
        assert result['match_count'] > 0

    @pytest.mark.asyncio
    async def test_search_case_sensitive(self, search_tool):
        """Test case-sensitive search."""
        result = await search_tool.search_codebase('HELLO')

        # Should not match lowercase 'hello'
        assert result['success'] is True
        assert result['match_count'] == 0


class TestSearchToolWithoutRipgrep:
    """Test search tool behavior when ripgrep is not installed."""

    @pytest.mark.asyncio
    async def test_search_without_ripgrep(self, search_tool, monkeypatch):
        """Test that search returns error when ripgrep is not available."""
        # Mock shutil.which to return None (ripgrep not found)
        monkeypatch.setattr('shutil.which', lambda x: None)

        result = await search_tool.search_codebase('test')

        assert result['success'] is False
        assert 'ripgrep' in result['error'].lower() or 'rg' in result['error'].lower()
