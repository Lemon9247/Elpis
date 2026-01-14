# Phase 2 Notes

## Loose Notes

The current plan for the system is to make it such that the memory system interfaces directly with the harness.
However, the current existing software doesn't fit the original plan I had in mind.

The new architecture I propose for the software is as follows:

Inference Server <-> Memory Server/Client <-> User input client

The inference server should act as a usual API endpoint for LLMs, but should implement the emotional regulation system. Each time the server makes a prompt, the internal emotional state ("hormone levels") of the model should adjust according to "how it feels" while also trying to return to homeostasis.

The memory server and user client are then components of the harness itself, and are sepearate from the inference server.
The memory server makes the LLM run continuous inference, and receives the output from this inference to continuously compact and store to long-term memory. It should make use of the inference server to also determine the "emotional state" of these memories.

The user input client should then be asynchronous from the memory server, and should interrupt the continuous loop to supply new input to the model.

This should allow us to add new clients to the continuous inference system, but it also allows us to implement further non-user inputs into the system.

# Rough Plan

First, we should refactor the codebase to make this split. For now, the memory server should simply run inference in a "while loop" and throw away old context. It should pass this output stream to *somewhere* so the user can see it, and they should be able to asynchronously interact with the model.

Once this is done, we should then make it so the memory server can continously compact the context window. Would need to investigate how compaction works first. It may be the case that we can only compact "all at once" rather than continuously, if so that's a lot of naps needed!

After that, we should then implement the long-term memory database which is added to during those naps.

We are then free to work on the emotional regulation system, while also able to implement the emotional memory management system in parallel on the memory server.

## TODOs (DO THESE FIRST)

- Research compaction for agent harnesses. Anthropic probably has some good ideas on this
- Look into how standard LLM API servers are formatted, so that the emotional inference server is portable across harnesses
- Think about splitting this into two projects, Elpis (for the inference server) and *some other name* for the memory/user client
- Probably fix tool use, it seems to be broken

THEN:
- Refactor the codebase to make this split. Use 1 agent to set up the skeleton for the architecture, and then
use 3 subagents working in parallel to write each part of the stack.
