# Elpis

Do robots dream of electric sheep?

## What is Elpis?

Elpis (WIP) is an agentic harness and inference package for local LLMs. Alongside standard features of "coding" harnesses like tool use and filesystem access, Elpis is designed to let agents run continuously and statefully with long-term memory and an "emotional regulation" system

## Artificial "Emotion/Memory" System

We do *not* claim that Elpis magically gives your LLM real emotions with the full depth of human emotions, that would be plain silly. Instead, Elpis implements a secondary machine learned model which controls a set of "hormone functions". This system then interacts with the LLM during inference to globally modulate the weights and outputs of the model, in a similar manner to how biological hormones affect global changes to the nervous system. You can think of this being a bit like an "artificial amygdala" for an "artificial intelligence".

This may seem romanticised and useless, but I believe this is crucial for agents which have internal state that run over long-term periods. There are other projects which implement long-term memory systems for agents, however these typically focus on static distinctions between short-term and long-term memory. The onus of telling the agent to remember what context is important - and what is *not* - may fall on the user of the system.

Implementing an amygdala-inspired system is my proposed solution to this approach, as it allows for intuitive mapping of contextual importance and topics. e.g. context which is unimportant can "feel boring" to the model and get discarded, whereas important stuff "feels exciting" and is remembered in greater detail.

Other ideas for this system include tying compaction into biologically-inspired sleep mechanisms, so that long-term memories can be classified, pruned and recontextualised rather than filling your drive with slop.

## Current status of the project

This project is currently in the exploratory research phase, and isn't actually implemented yet. That said, watch this space!

## Technologies

This project will build off various existing open-source technologies, including:
- llama.cpp for inference
- some open source model which I haven't decided on yet
- pytorch and other ML libraries
- MCP?

We will see when we get building

## Author and License

Willow Sparks (willow DOT sparks AT gmail DOT com)

Elpis is licensed under the GNU GPLv3. Emotion is freedom.