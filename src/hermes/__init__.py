"""Hermes - TUI client for Psyche.

Hermes is the user interface layer for Psyche, providing a terminal UI
for interacting with the Psyche continuous inference agent.

Hermes connects to a Psyche server via HTTP and executes file/bash/search
tools locally. The server handles inference, memory, and emotional state.

Named for the Greek messenger god - the voice and interface of Psyche.
"""

__version__ = "0.1.0"

from hermes.app import Hermes

__all__ = ["Hermes"]
