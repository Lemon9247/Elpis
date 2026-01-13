"""Codebase search tool using ripgrep."""
import asyncio
import subprocess
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from psyche.tools.tool_engine import ToolSettings


class SearchTool:
    """Codebase search using ripgrep (rg)."""

    def __init__(self, workspace_dir: Path, settings: "ToolSettings"):
        """
        Initialize search tool.

        Args:
            workspace_dir: Root workspace directory to search within
            settings: Tool settings object
        """
        self.workspace_dir = workspace_dir
        self.settings = settings
        self._check_ripgrep_installed()

    def _check_ripgrep_installed(self) -> None:
        """
        Check if ripgrep is installed.

        Raises:
            ToolExecutionError: If ripgrep is not found
        """
        if not shutil.which('rg'):
            # Don't raise an error, just note it
            # The tool will return an error message when used
            pass

    async def search_codebase(
        self,
        pattern: str,
        file_glob: Optional[str] = None,
        context_lines: int = 0
    ) -> Dict[str, Any]:
        """
        Search codebase for pattern using ripgrep.

        Args:
            pattern: Regex pattern to search for
            file_glob: Optional file glob pattern (e.g., '*.py', '**/*.js')
            context_lines: Number of context lines to show around matches

        Returns:
            Dictionary with success, matches, and match_count
        """
        return await asyncio.to_thread(
            self._search_codebase_sync,
            pattern,
            file_glob,
            context_lines
        )

    def _search_codebase_sync(
        self,
        pattern: str,
        file_glob: Optional[str],
        context_lines: int
    ) -> Dict[str, Any]:
        """
        Synchronous ripgrep search implementation.

        Args:
            pattern: Regex pattern to search
            file_glob: Optional file glob pattern
            context_lines: Context lines around matches

        Returns:
            Dictionary with success, matches, match_count, and pattern
        """
        try:
            # Check if ripgrep is available
            if not shutil.which('rg'):
                return {
                    'success': False,
                    'error': "ripgrep (rg) is not installed. Please install it to use search functionality."
                }

            # Build ripgrep command
            cmd = ['rg', '--json', pattern]

            # Add context lines if requested
            if context_lines > 0:
                cmd.extend(['-C', str(context_lines)])

            # Add file glob if provided
            if file_glob:
                cmd.extend(['-g', file_glob])

            # Execute ripgrep in workspace directory
            process = subprocess.run(
                cmd,
                cwd=str(self.workspace_dir),
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout for searches
            )

            # Parse results
            matches = []
            if process.stdout:
                import json
                for line in process.stdout.strip().split('\n'):
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if entry.get('type') == 'match':
                            data = entry.get('data', {})
                            matches.append({
                                'file': data.get('path', {}).get('text', ''),
                                'line_number': data.get('line_number'),
                                'line': data.get('lines', {}).get('text', '').rstrip(),
                                'match': data.get('submatches', [{}])[0] if data.get('submatches') else {}
                            })
                    except json.JSONDecodeError:
                        continue

            # ripgrep returns exit code 1 when no matches found, which is not an error
            if process.returncode == 0 or process.returncode == 1:
                return {
                    'success': True,
                    'matches': matches,
                    'match_count': len(matches),
                    'pattern': pattern,
                    'file_glob': file_glob
                }
            else:
                return {
                    'success': False,
                    'error': f"ripgrep failed: {process.stderr}",
                    'pattern': pattern
                }

        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': "Search timed out after 30 seconds",
                'pattern': pattern
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Error searching codebase: {str(e)}",
                'pattern': pattern
            }
