"""Bash command execution tool with safety checks."""
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, Any, List

from elpis.config.settings import Settings
from elpis.utils.exceptions import CommandSafetyError, ToolExecutionError


class BashTool:
    """Bash command execution with safety checks."""

    # Dangerous command patterns that are blocked by default
    DANGEROUS_PATTERNS = [
        'rm -rf /',
        'rm -rf ~',
        'rm -rf *',
        'mkfs',
        ':(){:|:&};:',  # Fork bomb
        'dd if=/dev/zero',
        'mv / ',
        'chmod -R 777 /',
        '> /dev/sda',
        'wget http',  # Could be downloading malware
        'curl http',  # Could be downloading malware
    ]

    def __init__(self, workspace_dir: Path, settings: Settings):
        """
        Initialize bash tool.

        Args:
            workspace_dir: Root workspace directory (commands run from here)
            settings: Global settings object
        """
        self.workspace_dir = workspace_dir
        self.settings = settings

    def _check_command_safety(self, command: str) -> None:
        """
        Check if command contains dangerous patterns.

        Args:
            command: Command to check

        Raises:
            CommandSafetyError: If dangerous pattern detected and dangerous commands not enabled
        """
        if self.settings.tools.enable_dangerous_commands:
            return

        command_lower = command.lower()

        for pattern in self.DANGEROUS_PATTERNS:
            if pattern.lower() in command_lower:
                raise CommandSafetyError(
                    f"Dangerous command pattern detected: '{pattern}'. "
                    f"Set enable_dangerous_commands=true to allow."
                )

        # Check for suspicious redirects
        if '>>' in command or '>' in command:
            # Allow redirects to files in workspace, but not to devices or /
            if '/dev/' in command or command.strip().endswith('> /'):
                raise CommandSafetyError(
                    "Suspicious redirect detected. "
                    "Set enable_dangerous_commands=true to allow."
                )

    async def execute_bash(self, command: str) -> Dict[str, Any]:
        """
        Execute bash command asynchronously.

        Args:
            command: Bash command to execute

        Returns:
            Dictionary with success, stdout, stderr, exit_code
        """
        return await asyncio.to_thread(self._execute_bash_sync, command)

    def _execute_bash_sync(self, command: str) -> Dict[str, Any]:
        """
        Synchronous bash command execution.

        Args:
            command: Command to execute

        Returns:
            Dictionary with success, stdout, stderr, exit_code, and command
        """
        try:
            # Check command safety
            self._check_command_safety(command)

            # Execute command
            process = subprocess.run(
                command,
                shell=True,
                cwd=str(self.workspace_dir),
                capture_output=True,
                text=True,
                timeout=self.settings.tools.max_bash_timeout,
                env={**subprocess.os.environ}  # Inherit environment
            )

            return {
                'success': process.returncode == 0,
                'stdout': process.stdout,
                'stderr': process.stderr,
                'exit_code': process.returncode,
                'command': command
            }

        except CommandSafetyError as e:
            return {
                'success': False,
                'error': str(e),
                'command': command
            }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': f"Command timeout after {self.settings.tools.max_bash_timeout} seconds",
                'command': command
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Error executing command: {str(e)}",
                'command': command
            }
