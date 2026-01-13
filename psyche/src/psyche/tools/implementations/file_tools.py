"""File operation tools: read_file and write_file."""
import asyncio
from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from psyche.tools.tool_engine import ToolSettings


class PathSafetyError(Exception):
    """Exception raised when a path is outside the workspace."""
    pass


class FileTools:
    """File operation tools with async execution."""

    def __init__(self, workspace_dir: Path, settings: "ToolSettings"):
        """
        Initialize file tools.

        Args:
            workspace_dir: Root workspace directory (all paths must be within this)
            settings: Tool settings object
        """
        self.workspace_dir = workspace_dir
        self.settings = settings

    def _sanitize_path(self, file_path: str) -> Path:
        """
        Sanitize and validate file path.

        Args:
            file_path: Path to sanitize (can be relative or absolute)

        Returns:
            Resolved absolute path

        Raises:
            PathSafetyError: If path is outside workspace or contains unsafe characters
        """
        # Convert to Path and resolve
        if Path(file_path).is_absolute():
            resolved = Path(file_path).resolve()
        else:
            resolved = (self.workspace_dir / file_path).resolve()

        # Ensure path is within workspace
        try:
            resolved.relative_to(self.workspace_dir)
        except ValueError:
            raise PathSafetyError(
                f"Path '{file_path}' is outside workspace '{self.workspace_dir}'"
            )

        return resolved

    async def read_file(self, file_path: str, max_lines: int = 2000) -> Dict[str, Any]:
        """
        Read file contents asynchronously.

        Args:
            file_path: Path to file (relative to workspace or absolute)
            max_lines: Maximum number of lines to read

        Returns:
            Dictionary with success status and either content or error
        """
        return await asyncio.to_thread(self._read_file_sync, file_path, max_lines)

    def _read_file_sync(self, file_path: str, max_lines: int) -> Dict[str, Any]:
        """
        Synchronous file read implementation.

        Args:
            file_path: Path to file
            max_lines: Maximum number of lines to read

        Returns:
            Dictionary with success, content, line_count, and file_path
        """
        try:
            # Sanitize path
            path = self._sanitize_path(file_path)

            # Check if file exists
            if not path.exists():
                return {
                    'success': False,
                    'error': f"File not found: {path}"
                }

            # Check if it's a file (not directory)
            if not path.is_file():
                return {
                    'success': False,
                    'error': f"Path is not a file: {path}"
                }

            # Check file size
            file_size = path.stat().st_size
            if file_size > self.settings.max_file_size:
                return {
                    'success': False,
                    'error': f"File too large: {file_size} bytes (max: {self.settings.max_file_size})"
                }

            # Read file
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    lines.append(line)

            content = ''.join(lines)
            total_lines_in_file = sum(1 for _ in open(path, 'r', encoding='utf-8', errors='replace'))
            truncated = total_lines_in_file > max_lines

            return {
                'success': True,
                'content': content,
                'line_count': len(lines),
                'total_lines': total_lines_in_file,
                'truncated': truncated,
                'file_path': str(path),
                'size_bytes': file_size
            }

        except PathSafetyError as e:
            return {
                'success': False,
                'error': str(e)
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Error reading file: {str(e)}"
            }

    async def write_file(
        self,
        file_path: str,
        content: str,
        create_dirs: bool = True
    ) -> Dict[str, Any]:
        """
        Write file contents asynchronously.

        Args:
            file_path: Path to file (relative to workspace or absolute)
            content: Content to write
            create_dirs: Create parent directories if they don't exist

        Returns:
            Dictionary with success status and either result or error
        """
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
        """
        Synchronous file write implementation.

        Args:
            file_path: Path to file
            content: Content to write
            create_dirs: Create parent directories if needed

        Returns:
            Dictionary with success, file_path, and size_bytes
        """
        try:
            # Sanitize path
            path = self._sanitize_path(file_path)

            # Create parent directories if requested
            if create_dirs and not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)

            # Backup existing file
            backup_path = None
            if path.exists():
                backup_path = path.with_suffix(path.suffix + '.bak')
                # If backup already exists, remove it
                if backup_path.exists():
                    backup_path.unlink()
                # Rename current file to backup
                path.rename(backup_path)

            # Write new content
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)

                size_bytes = len(content.encode('utf-8'))

                result = {
                    'success': True,
                    'file_path': str(path),
                    'size_bytes': size_bytes,
                    'lines_written': content.count('\n') + 1 if content else 0
                }

                if backup_path:
                    result['backup_created'] = str(backup_path)

                return result

            except Exception as write_error:
                # Restore backup if write failed
                if backup_path and backup_path.exists():
                    backup_path.rename(path)
                raise write_error

        except PathSafetyError as e:
            return {
                'success': False,
                'error': str(e)
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Error writing file: {str(e)}"
            }
