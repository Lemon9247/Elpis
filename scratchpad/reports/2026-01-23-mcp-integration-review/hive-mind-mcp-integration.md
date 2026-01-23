# Hive Mind: MCP Integration Review

## Task Overview
Review how external MCP servers are integrated in Elpis and compare with other harnesses/clients.

## Agents
1. **Codebase Agent** - Reviews Elpis's current MCP integration code
2. **Research Agent** - Investigates how other harnesses integrate external MCP servers

## Coordination Notes
- Both agents should write their findings to separate markdown files in this folder
- Focus on: configuration, server discovery, tool registration, error handling, lifecycle management

## Questions for Discussion
(Agents can add questions here for coordination)

## Status
- [x] Codebase Agent: Complete (report written)
- [x] Research Agent: Complete (report written)
- [x] Final synthesis: Complete

## Reports Generated
1. `codebase-agent-report.md` - Internal architecture analysis
2. `research-agent-report.md` - External patterns research
3. `final-synthesis-report.md` - Combined recommendations

## Key Findings
- Elpis follows MCP best practices for internal servers
- Primary gap: No configuration mechanism for external MCP servers
- Recommendation: Add `~/.psyche.json` with standard `mcpServers` format
- Future consideration: Route external MCP connections through Hermes
