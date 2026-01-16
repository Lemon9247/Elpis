The current implementation of Psyche is "workable" but has limitations. These include:

- Weird UI:
    - Main chat is streaming but dream state is not??
    - There is no hint that you should use /quit to exit, or any hint about / commands in general
    - We should have short aliases for / commands as well, e.g /q, /h etc.

- Interruptions:
    - When the LLM is talking/dreaming, it is impossible to interrupt the LLM's message with a new prompt.
    - It is also impossible to send a message to the LLM while it is using a tool, or interrupt tool use.

- Memory:
    - Compaction and memory storage system seems very flawed. We should determine a better approach
    - Maybe we should look into some kind of "focus" system (See FocusLLM https://arxiv.org/abs/2408.11745)
    - The agent *still* does not seem to actually store anything to memory when Psyche shuts down.

- Tools:
    - The current tool implementations are very unintuitive for the LLM to use. Part of this is due to the
      LLM being very small, but a larger part is that the tools are badly designed. We should improve this.
    - When using a tool, the LLM just dumps the JSON for the tool use into the chat window. This is good for
      debugging, but terrible as actual UI. It would be nicer to take an approach similar to other coding
      agents (e.g. show "Editing foo.txt" rather than the full JSON). We should log tool use (and maybe all
      chat in general) to a json file

- Reasoning:
    - The current chat pattern is very simplistic. User provides new prompt, LLM responds immediately.
      It would be better if the LLM "reasoned" about the prompt first. This reasoning should not be shown to
      the user, but should be possible to be shown/hidden with a keybind or slash command in the same manner
      as is done for other coding agents

- Interopability:
    - Does Psyche actually work well with external MCP servers? Why does Psyche have a tool for long-term memory
      management, when that should be an MCP interface provided by Mnemosyne.
      (Note: Mnemosyne *does* expose some MCP tools for memory interfacing, but the tools seem limited)
    - IS PSYCHE'S TOOL SYSTEM ACTUALLY ANY GOOD??
    - Would Psyche work with a different LLM backend that isn't Elpis? If not, this is bad. We should make sure
      these things are "standardised"

ACTION POINTS:
1) Review the current codebase structure in relation to these issues. Determine what work needs doing, and how
   much work it will take
   
2) Review other coding agents, including: Crush, Opencode and Letta (N.B. we have existing priors on Letta).
   Review *how* they implement our desired approaches to interopability and tool use with various different
   LLM backends (e.g. provider API, ollama, etc.)

3) Review the FocusLLM paper https://arxiv.org/abs/2408.11745 to see how we can improve the memory system,
   and find any other papers which can help improve our current memory workflow. We are still happy with using
   ChromaDB overall, it suits our needs well.

4) Review LLM reasoning workflows, and how the other coding agents implement these. Find publications from
   OpenAI, Anthropic and other AI research institutions which have implemented reasoning models. Synthesise
   this with our existing implementation to determine an actionable reasoning implementation.

5) Synthesise this into an actionable work plan. We should prioritise the codebase to be modular, and should
   not rely on quick hacks. The work plan should start by thinking about the overall *architecture* of the codebase,
   and should extend into looking into the architecture at lower and lower levels. We should include a graph of the
   architecture somewhere, and make sure it is interopable, modular and most importantly: *UNDERSTANDABLE*.
   We should have clear distinction between upper layers (user interface etc), transport layers (compaction, reasoning, etc)
   and lower layers (transformers, llama-cpp, chromadb, etc)