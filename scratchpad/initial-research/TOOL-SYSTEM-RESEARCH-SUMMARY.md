# Tool-System-Agent Research Summary

**Date:** January 11, 2026  
**Agent:** Tool-System-Agent  
**Project:** Elpis - Emotional Coding Agent  
**Status:** COMPLETE

---

## Research Mission

As the Tool-System-Agent, I was tasked with researching and documenting:

1. **Tool Definition Schemas** - How to describe tools to LLMs
2. **Function Calling Formats** - Patterns and protocols for tool invocation
3. **Safe Execution Engine Design** - Sandboxing, whitelisting, error handling
4. **Implementation Patterns** - Concrete code for read_file, write_file, execute_bash, search_codebase, list_directory
5. **Best Practices** - Security and reliability patterns for agent tool systems

---

## Key Findings Summary

### 1. Tool Definition Schema

**Industry Standard:** OpenAI-compatible JSON Schema format

All major platforms now use the same specification:
- OpenAI API
- Google Vertex AI
- vLLM
- llama.cpp
- HuggingFace Transformers

**Format:**
```json
{
  "type": "function",
  "function": {
    "name": "tool_name",
    "description": "What the tool does",
    "parameters": {
      "type": "object",
      "properties": {
        "param": {
          "type": "string",
          "description": "Parameter description"
        }
      },
      "required": ["param"]
    }
  }
}
```

**Python Implementation:** Use Pydantic models for automatic schema generation and runtime validation.

### 2. Function Calling Format

**OpenAI Chat Completion Format (Standard):**

```json
{
  "tool_calls": [
    {
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "read_file",
        "arguments": "{\"file_path\": \"/path/to/file.py\"}"
      }
    }
  ]
}
```

**Advantages:**
- Industry standard across multiple platforms
- Native llama.cpp support with lazy grammar
- Supports parallel tool calls
- Clear tool ID for result mapping

**Recommendation for Elpis:** Use this format for compatibility with both Llama 3.1 and Mistral 7B models already selected.

### 3. Safe Execution Architecture

Implemented as a **6-Layer Model:**

```
Layer 1: Input Validation
  └─ Parse arguments and validate against Pydantic schemas

Layer 2: Authorization
  └─ Check tool whitelist and user permissions

Layer 3: Argument Sanitization
  └─ Path traversal checks, command injection prevention

Layer 4: Sandboxed Execution
  └─ OS-level isolation (subprocess restrictions, environment limits)

Layer 5: Monitoring & Logging
  └─ Full audit trail of all tool calls and results

Layer 6: Error Handling
  └─ Retry logic with exponential backoff, graceful degradation
```

Each layer is independent and can fail safely without compromising the others.

### 4. Core Tool Implementation Patterns

**read_file**
- Validates absolute paths (prevents directory traversal)
- Enforces size limits (10MB default)
- Handles encoding safely with fallback
- Optional regex filtering for large files
- Returns: success flag, content, line count

**write_file**
- Validates target path (workspace only)
- Creates parent directories if needed
- Automatic backup of existing files
- Atomic writes to prevent corruption
- Size limit validation (10MB)

**execute_bash**
- Command validation via deny-list patterns
- Blocks dangerous commands (rm -rf, dd, sudo, etc.)
- Timeout enforcement (30 seconds default)
- Environment isolation (limited PATH, no HOME)
- Process group isolation (preexec_fn=os.setsid)
- Captures stdout/stderr safely

**search_codebase**
- Regex validation (prevents ReDoS attacks)
- Ripgrep wrapper for safety and performance
- Output limiting (100 results max)
- File globbing support
- Context lines around matches

**list_directory**
- Path validation (absolute, within workspace)
- Recursive directory traversal support
- Entry limiting (1000 max per call)
- Metadata tracking (name, type, size)
- Relative path calculation

### 5. Best Practices Identified

1. **Always use absolute paths** - Resolve with Path().resolve()
2. **Separate concerns** - Validation separate from execution
3. **Use type hints everywhere** - Catches bugs early
4. **Log everything** - Audit trail for security and debugging
5. **Set resource limits** - File size, timeout, output length
6. **Use Pydantic models** - Automatic schema + validation
7. **Implement retry logic** - Exponential backoff for transient errors
8. **Structured error messages** - Include recovery suggestions
9. **Command sanitization** - Deny-list approach for bash commands
10. **Process isolation** - OS-level process group creation

---

## Deliverables

### Primary Report
**File:** `/home/lemoneater/Devel/elpis/scratchpad/tool-system-report.md`  
**Size:** 1,430 lines, 43KB  
**Contents:**
- Executive summary
- Tool definition schema details with constraints
- Function calling format comparison
- Complete 6-layer safety architecture explanation
- Full implementation patterns with code examples
- Complete working tool engine example
- Best practices section
- Integration recommendations with emotional system
- References and sources

### Updated Coordination Hub
**File:** `/home/lemoneater/Devel/elpis/scratchpad/hive-mind.md`  
**Updates:** Added comprehensive tool system research findings with:
- Tool schema specification
- Function calling response format
- 6-layer architecture overview
- Core tool implementation patterns
- Security best practices
- Integration guidelines
- Recommended technology stack
- Phase 1 implementation focus

---

## Integration with Elpis Architecture

### Phase 1 (Weeks 1-2) - Basic Tool Engine
- Implement tool definition schema system
- Create read_file and list_directory tools (low-risk foundation)
- Add basic execute_bash with validation
- Implement error handling and logging
- Test with llama-cpp-python integration

### Phase 2 (Weeks 3-4) - Enhanced Safety
- Add search_codebase and write_file tools
- Implement complete validation layer
- Add OS-level process isolation
- Create comprehensive audit logging
- Build tool authorization system

### Phase 3 (Weeks 5-6) - Emotional Integration
- Connect tool outcomes to emotional system
- Tool success → ↑ Dopamine
- Tool failure → ↑ Norepinephrine  
- Novel tools → ↑ Acetylcholine
- Consistent success → ↑ Serotonin

### Phase 4+ (Weeks 7+) - Advanced Features
- Parallel tool calls
- Tool composition/chaining
- Improved error recovery strategies
- Advanced sandboxing (containers/gVisor)

---

## Technical Stack Recommendations

**Validation:** Pydantic models with custom field validators  
**Execution:** Python subprocess with timeout and isolation  
**Logging:** Structured JSON to file for audit trail  
**Framework:** llama-cpp-python for OpenAI-compatible API  
**Workspace:** Dedicated directory `/home/lemoneater/Devel/elpis/workspace`  
**LLM Integration:** Chat completion with tools parameter

---

## Security Considerations

**Attack Vectors Addressed:**
- Path traversal attacks → Absolute path validation + workspace restriction
- Command injection → Deny-list pattern matching + argument separation
- Resource exhaustion → Timeout enforcement + size limits
- Unauthorized access → Tool whitelisting + authorization checks
- Data leaks → Logging restrictions + output sanitization

**Defense Strategy:**
- **Defense in depth:** Multiple layers ensure single layer failure doesn't compromise security
- **Fail-safe defaults:** Tools require explicit allowlisting
- **Least privilege:** Subprocess runs with minimal permissions
- **Audit trail:** All operations logged for forensic analysis

---

## Code Quality Standards

The research includes production-ready code examples with:
- Full type hints for all functions
- Comprehensive docstrings
- Error handling with meaningful messages
- Resource limit enforcement
- Structured logging integration
- Pydantic model validation

All patterns follow Python best practices and are suitable for direct implementation.

---

## Sources & References

### Official Documentation
- [vLLM Tool Calling](https://docs.vllm.ai/en/latest/features/tool_calling/)
- [OpenAI Function Calling API](https://platform.openai.com/docs/guides/function-calling)
- [llama.cpp Function Calling](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md)
- [Pydantic AI Framework](https://ai.pydantic.dev/)

### Security Research
- [Claude Code Sandboxing](https://code.claude.com/docs/en/sandboxing)
- [Trail of Bits: Prompt Injection to RCE](https://blog.trailofbits.com/2025/10/22/prompt-injection-to-rce-in-ai-agents/)
- [LLM Agent Sandboxing Techniques](https://www.codeant.ai/blogs/agentic-rag-shell-sandboxing)

### Implementation Patterns
- [LangChain Agent Error Handling](https://apxml.com/courses/langchain-production-llm/chapter-2-sophisticated-agents-tools/agent-error-handling)
- [ReAct Pattern](https://agent-patterns.readthedocs.io/en/stable/patterns/re_act.html)

---

## Next Steps

1. **Review** - Team reviews tool-system-report.md
2. **Plan** - Architecture team maps implementation schedule
3. **Implement** - Phase 1 tool engine using provided patterns
4. **Test** - Unit and integration tests for each tool
5. **Integrate** - Connect to LLM inference layer (Phase 1)
6. **Enhance** - Add emotional system integration (Phase 3)

---

## Conclusion

The research identified a clear, industry-standard approach for tool systems in LLM-based agents. The OpenAI-compatible function calling format provides excellent compatibility with both llama.cpp (our inference engine) and the Llama 3.1/Mistral models selected for Elpis.

The 6-layer security architecture balances safety with usability, providing defense-in-depth against common attack vectors while remaining simple to understand and maintain.

The provided implementation patterns and code examples are production-ready and can be directly integrated into the Elpis codebase during Phase 1.

**Status:** COMPLETE - Ready for Phase 1 implementation

---

**Tool-System-Agent**  
Elpis Project Coordination Hub  
January 2026
