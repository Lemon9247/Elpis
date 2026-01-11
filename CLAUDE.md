# Work Planning

1) All project notes, work logs and reports can be found in the scratchpad folder

2) When Claude first starts, it should review the latest work on the project by reviewing the git history and anything recent in the scratchpad

3) When Claude is finished working on a long task, it should write a report on its work into a new timestamped markdown file in the scratchpad


# Programming Tasks

1) Claude should always use the Python virtual environment in venv

2) Emojis should never be used in actual code, however they are fine for plaintext files such as .md, .txt, etc.

3) Claude should think carefully about the code it writes, and should not make random assumptions about how a function works

4) When running tests, Claude should prefer running single tests based on what it has changed first. Running the whole test suite should come at the end


# Subagents

1) When using multiple sub-agents for a task, Claude should create a new subfolder in the scratchpad folder.

2) Each subagent should be given a name based on their role, e.g. Testing Agent, Coding Agent

3) This subfolder should contain a hive-mind-[TASK].md file, where [TASK] is substituted with an appropriate name for the task. The subagents
should use this file to coordinate with each other and ask questions.

4) When each subagent finishes their task, they should write up a report of their work in separate markdown files in this subfolder.

5) When all subagents finish, Claude should synthesise their reports into a final summary report, which should be a separate markdown file.