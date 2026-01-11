"""
Agent orchestrator implementing the ReAct pattern for reasoning and action.

The orchestrator manages the agent's main loop:
1. REASON: Build context and prompt
2. ACT: Generate LLM response (text or tool calls)
3. OBSERVE: Execute tools and collect results
4. REPEAT: Continue until final answer is reached
"""

import asyncio
from typing import Any, Dict, List, Optional

from loguru import logger

from elpis.llm.inference import LlamaInference
from elpis.llm.prompts import build_system_prompt
from elpis.tools.tool_engine import ToolEngine


class AgentOrchestrator:
    """
    Main agent reasoning loop using ReAct pattern.

    The orchestrator coordinates between the LLM and tool execution,
    maintaining conversation history and managing the iterative
    reasoning process.

    Attributes:
        llm: LLM inference engine for generating responses
        tools: Tool execution engine
        settings: Configuration settings
        message_history: Conversation history
    """

    def __init__(
        self,
        llm: LlamaInference,
        tools: ToolEngine,
        settings: Any = None,
    ):
        """
        Initialize the agent orchestrator.

        Args:
            llm: LLM inference instance
            tools: Tool engine instance
            settings: Configuration settings
        """
        self.llm = llm
        self.tools = tools
        self.settings = settings
        self.message_history: List[Dict[str, Any]] = []

    async def process(self, user_input: str) -> str:
        """
        Process user input using the ReAct pattern.

        This is the main entry point for agent interactions. It implements
        the full ReAct loop:
        1. REASON: Build prompt with context and system instructions
        2. ACT: Generate LLM response (text or tool calls)
        3. OBSERVE: Execute tools if needed and collect results
        4. REPEAT: Loop until final answer is generated

        Args:
            user_input: User's message/request

        Returns:
            Agent's final response to the user

        Raises:
            Any exceptions from LLM or tool execution are logged but not raised
        """
        logger.info(f"Processing user input: {user_input[:100]}...")

        # Add user message to history
        self.message_history.append({"role": "user", "content": user_input})

        # ReAct loop with max iterations to prevent infinite loops
        max_iterations = 10
        for iteration in range(max_iterations):
            logger.debug(f"ReAct iteration {iteration + 1}/{max_iterations}")

            try:
                # REASON: Build messages with system prompt and history
                messages = self._build_messages()

                # ACT: Try to get function calls from LLM
                tool_calls = await self.llm.function_call(
                    messages=messages, tools=self.tools.get_tool_schemas()
                )

                if tool_calls:
                    # OBSERVE: Execute tools concurrently
                    logger.info(f"Executing {len(tool_calls)} tool call(s)")
                    results = await self._execute_tools(tool_calls)

                    # Add tool calls to history
                    self.message_history.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": tool_calls,
                        }
                    )

                    # Add tool results to history
                    self.message_history.append(
                        {
                            "role": "tool",
                            "content": self._format_tool_results(results),
                        }
                    )

                    # Continue loop for next reasoning step
                    continue

                else:
                    # No tool calls - get final text response
                    logger.debug("No tool calls, generating final response")
                    response = await self.llm.chat_completion(messages)

                    # Add assistant response to history
                    self.message_history.append(
                        {"role": "assistant", "content": response}
                    )

                    logger.info("Generated final response")
                    return response

            except Exception as e:
                logger.exception(f"Error in ReAct iteration {iteration + 1}: {e}")
                # Try to continue if possible
                error_message = f"I encountered an error: {str(e)}"
                self.message_history.append(
                    {"role": "assistant", "content": error_message}
                )
                return error_message

        # Max iterations reached without final answer
        logger.warning(f"Reached max iterations ({max_iterations}) without completion")
        fallback_message = (
            "I apologize, but I've reached my reasoning limit for this request. "
            "Could you please rephrase your question or break it into smaller parts?"
        )
        self.message_history.append({"role": "assistant", "content": fallback_message})
        return fallback_message

    def _build_messages(self) -> List[Dict[str, str]]:
        """
        Build message list with system prompt.

        Constructs the full context for the LLM by prepending the system
        prompt to the conversation history.

        Returns:
            List of messages ready for LLM consumption
        """
        system_prompt = build_system_prompt(list(self.tools.tools.values()))

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.message_history)

        return messages

    async def _execute_tools(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Execute tool calls concurrently.

        Uses asyncio.gather to run multiple tool executions in parallel
        for improved performance.

        Args:
            tool_calls: List of tool call specifications from LLM

        Returns:
            List of tool execution results
        """
        logger.debug(f"Executing {len(tool_calls)} tool(s) concurrently")

        tasks = [self.tools.execute_tool_call(call) for call in tool_calls]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Tool {i} raised exception: {result}")
                processed_results.append(
                    {
                        "tool_call_id": tool_calls[i].get("id"),
                        "success": False,
                        "result": {"error": str(result)},
                        "duration_ms": 0,
                    }
                )
            else:
                processed_results.append(result)

        return processed_results

    def _format_tool_results(self, results: List[Dict[str, Any]]) -> str:
        """
        Format tool results for LLM consumption.

        Converts structured tool results into a text format that the LLM
        can understand and reason about.

        Args:
            results: List of tool execution results

        Returns:
            Formatted string describing tool results
        """
        formatted_parts = []

        for i, result in enumerate(results):
            tool_id = result.get("tool_call_id", f"tool_{i}")
            success = result.get("success", False)
            tool_result = result.get("result", {})
            duration = result.get("duration_ms", 0)

            if success:
                formatted_parts.append(
                    f"Tool {tool_id} succeeded (took {duration:.2f}ms):\n"
                    f"{tool_result}"
                )
            else:
                error = tool_result.get("error", "Unknown error")
                formatted_parts.append(
                    f"Tool {tool_id} failed (took {duration:.2f}ms):\n"
                    f"Error: {error}"
                )

        return "\n\n".join(formatted_parts)

    def clear_history(self) -> None:
        """
        Clear the conversation history.

        This is useful for starting a fresh conversation or managing
        memory usage in long-running sessions.
        """
        logger.info("Clearing message history")
        self.message_history = []

    def get_history_length(self) -> int:
        """
        Get the current length of message history.

        Returns:
            Number of messages in history
        """
        return len(self.message_history)

    def get_last_message(self) -> Optional[Dict[str, Any]]:
        """
        Get the last message in history.

        Returns:
            Last message dict or None if history is empty
        """
        return self.message_history[-1] if self.message_history else None
