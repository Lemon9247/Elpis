"""Directory listing tool."""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from psyche.tools.tool_engine import ToolSettings


class PathSafetyError(Exception):
    """Exception raised when a path is outside the workspace."""
    pass


class DirectoryTool:
    """List directory contents."""

    def __init__(self, workspace_dir: Path, settings: "ToolSettings"):
        """
        Initialize directory tool.

        Args:
            workspace_dir: Root workspace directory
            settings: Tool settings object
        """
        self.workspace_dir = workspace_dir
        self.settings = settings

    def _sanitize_path(self, dir_path: str) -> Path:
        """
        Sanitize and validate directory path.

        Args:
            dir_path: Directory path to sanitize

        Returns:
            Resolved absolute path

        Raises:
            PathSafetyError: If path is outside workspace
        """
        # Convert to Path and resolve
        if Path(dir_path).is_absolute():
            resolved = Path(dir_path).resolve()
        else:
            resolved = (self.workspace_dir / dir_path).resolve()

        # Ensure path is within workspace
        try:
            resolved.relative_to(self.workspace_dir)
        except ValueError:
            raise PathSafetyError(
                f"Path '{dir_path}' is outside workspace '{self.workspace_dir}'"
            )

        return resolved

    async def list_directory(
        self, dir_path: str = ".", recursive: bool = False, pattern: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        List directory contents asynchronously.

        Args:
            dir_path: Directory path (relative to workspace or absolute)
            recursive: List subdirectories recursively
            pattern: Optional glob pattern to filter results

        Returns:
            Dictionary with success, files, directories, and total_items
        """
        return await asyncio.to_thread(self._list_directory_sync, dir_path, recursive, pattern)

    def _list_directory_sync(
        self, dir_path: str, recursive: bool, pattern: Optional[str]
    ) -> Dict[str, Any]:
        """
        Synchronous directory listing implementation.

        Args:
            dir_path: Directory path
            recursive: List recursively
            pattern: Glob pattern filter

        Returns:
            Dictionary with directory listing results
        """
        try:
            # Sanitize path
            path = self._sanitize_path(dir_path)

            # Check if directory exists
            if not path.exists():
                return {"success": False, "error": f"Directory not found: {path}"}

            # Check if it's a directory
            if not path.is_dir():
                return {"success": False, "error": f"Path is not a directory: {path}"}

            files = []
            directories = []

            # List contents
            if recursive:
                # Recursive listing
                if pattern:
                    # Ensure pattern is recursive by adding **/ prefix if not present
                    glob_pattern = f"**/{pattern}" if not pattern.startswith("**/") else pattern
                else:
                    glob_pattern = "**/*"
                entries = path.glob(glob_pattern)
            else:
                # Non-recursive listing
                if pattern:
                    entries = path.glob(pattern)
                else:
                    entries = path.iterdir()

            for entry in sorted(entries):
                # Get relative path from workspace
                try:
                    rel_path = entry.relative_to(self.workspace_dir)
                except ValueError:
                    # Skip entries outside workspace
                    continue

                entry_info = {
                    "name": entry.name,
                    "path": str(rel_path),
                    "absolute_path": str(entry),
                }

                if entry.is_file():
                    entry_info["size"] = entry.stat().st_size
                    entry_info["type"] = "file"
                    files.append(entry_info)
                elif entry.is_dir():
                    entry_info["type"] = "directory"
                    directories.append(entry_info)

            return {
                "success": True,
                "directory": str(path.relative_to(self.workspace_dir)),
                "files": files,
                "directories": directories,
                "total_items": len(files) + len(directories),
                "file_count": len(files),
                "directory_count": len(directories),
                "recursive": recursive,
            }

        except PathSafetyError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            return {"success": False, "error": f"Error listing directory: {str(e)}"}
