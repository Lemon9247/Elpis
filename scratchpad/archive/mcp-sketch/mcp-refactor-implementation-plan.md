# Modular MCP Architecture - Implementation Plan

**Date:** 2026-01-14
**Branch:** `mcp-modular-refactor`
**Status:** Planning Phase

## Executive Summary

Transform Elpis from a monolithic project into composable MCP infrastructure:
- **elpis-inference**: Emotional LLM with steering vectors (standalone MCP server)
- **mnemosyne**: Memory system with ChromaDB (standalone MCP server)
- **psyche**: Reference harness orchestrating both servers

**Key Benefits:**
- Use emotional inference with any MCP harness (opencode, aider, etc.)
- Use memory with any LLM backend (Claude API, OpenAI, etc.)
- Clear API boundaries via MCP protocol
- Independent component evolution

**Estimated Effort:** 4-6 weeks full-time equivalent

---

## Critical Decisions (Resolve Before Implementation)

### Decision 1: Sleep Consolidation Orchestration

**Problem:** Mnemosyne needs an LLM for insight generation during sleep, but should remain backend-agnostic.

**Options:**

**A) Harness-Orchestrated** ⭐ **RECOMMENDED**
```python
# Harness coordinates consolidation
memories = await mnemosyne.call("memory_get_buffer")
insight = await elpis.call("generate", {
    "messages": [{"role": "system", "content": f"Analyze: {memories}"}]
})
await mnemosyne.call("memory_store", {"content": insight, ...})
```
- ✅ Keeps servers independent
- ✅ Allows custom consolidation strategies
- ✅ Works with any LLM backend
- ❌ More complex harness logic

**B) Mnemosyne Calls Back to Elpis**
- ❌ Creates inter-server dependency
- ❌ Requires both servers running
- ✅ Simpler harness code

**C) Mnemosyne Has Optional LLM Client**
- ❌ Duplicates inference capability
- ✅ Works without elpis-inference
- ⚠️ Requires API keys for external LLMs

**Decision:** Go with **Option A** (harness-orchestrated). Document consolidation pattern clearly.

---

### Decision 2: Emotional Response Analysis

**Problem:** Multiple files reference `regulator.process_response()` for analyzing model output, but it's commented out in the server sketch.

**Questions:**
1. Is this feature important for emotional dynamics?
2. Could it create unstable feedback loops?
3. Should it be optional/configurable?

**Recommendation:**
- **Defer to Phase 2+**: Implement basic emotional state first without response analysis
- Add as **experimental feature** with configurable enable/disable
- Include damping/caps to prevent runaway emotional spirals
- Document clearly why it's disabled by default

**Alternative:** If deemed essential, implement with:
```python
# config.yaml
emotion:
  response_analysis:
    enabled: false  # Disabled by default
    sentiment_weight: 0.3  # Reduced impact
    max_shift: 0.2  # Cap per-response shift
```

---

### Decision 3: Steering Vector Distribution

**Problem:** How to distribute pre-trained emotion vectors?

**Options:**

**A) Ship with Package** ⭐ **RECOMMENDED**
```
elpis-inference/
└── data/
    └── emotion_vectors/
        ├── excited.pt
        ├── frustrated.pt
        ├── calm.pt
        └── depleted.pt
```
- ✅ Works offline
- ✅ Instant setup
- ❌ ~5MB package size increase
- ✅ Users can override with custom training

**B) Download on First Run**
- ✅ Smaller package
- ❌ Requires internet
- ❌ Slower first launch

**C) Train During Setup**
- ❌ Takes several minutes
- ❌ Complex setup

**Decision:** **Option A** - Ship with package, provide retraining script for customization.

---

### Decision 4: Backend Selection Strategy

**Problem:** Support both llama-cpp-python (current, CPU-optimized) and transformers (new, steering support)?

**Recommendation:**
```yaml
# config.yaml
model:
  backend: "transformers"  # or "llama_cpp"
```

**Phase 1:** Transformers-only (simpler migration)
**Phase 2+:** Optional llama-cpp fallback for CPU performance

This avoids maintaining two backends during initial refactor.

---

## Implementation Phases

### Phase 0: Pre-Implementation (1 week)

**Goal:** Resolve critical decisions, prepare environment

**Tasks:**
- [x] Create `mcp-modular-refactor` branch
- [x] Save design sketches to `scratchpad/mcp-sketch/`
- [ ] **DECISION MEETING**: Resolve Decisions 1-4 above
- [ ] Set up separate repo/directory structure for testing
- [ ] Install dependencies (transformers, steering-vectors, chromadb)
- [ ] Train initial emotion vectors for Llama 3.1 8B
- [ ] Document steering vector training process

**Deliverables:**
- Decision log documenting choices made
- Pre-trained emotion vectors
- Clean dev environment

---

### Phase 1: Internal Modularization (1-2 weeks)

**Goal:** Reorganize current codebase into logical modules WITHOUT breaking Psyche

**Tasks:**

#### 1.1 Emotion System Updates
- [ ] Add `get_steering_coefficients()` to `EmotionalState`
- [ ] Keep `get_modulated_params()` for backward compatibility
- [ ] Add `steering_strength` field to `EmotionalState`
- [ ] Add `get_dominant_emotion()` helper method
- [ ] Update `EmotionalState.to_dict()` to include steering coefficients
- [ ] Write tests for steering coefficient calculation
  - Test quadrant mapping (excited, frustrated, calm, depleted)
  - Test bilinear interpolation
  - Test neutral state balance
  - Test steering strength scaling

#### 1.2 Transformers Inference Backend
- [ ] Implement `TransformersInference` class (from sketch file 01)
- [ ] Add emotion vector training functionality
- [ ] Add emotion vector loading/saving
- [ ] Implement blended steering calculation
- [ ] Add chat completion with steering
- [ ] Add streaming support
- [ ] Add function calling (basic implementation)
- [ ] Write tests for inference
  - Test model loading
  - Test emotion vector blending
  - Test generation with steering
  - Test streaming

#### 1.3 Configuration System
- [ ] Add `SteeringConfig` dataclass
- [ ] Update `ModelSettings` with backend selection
- [ ] Add steering-related config fields
- [ ] Create example config file with transformers backend
- [ ] Write config validation tests

#### 1.4 Integration Testing
- [ ] Test emotional state → steering coefficients → inference
- [ ] Test with Psyche harness (should still work)
- [ ] Performance benchmark: transformers vs llama-cpp
- [ ] Memory usage profiling

**Success Criteria:**
- All existing tests pass
- Psyche works with new transformers backend
- Emotion vectors correctly modulate generation
- No breaking changes to public APIs

---

### Phase 2: Extract elpis-inference Package (2 weeks)

**Goal:** Split out emotional inference as standalone MCP server

**Tasks:**

#### 2.1 Package Structure
- [ ] Create `packages/elpis-inference/` directory
- [ ] Set up `pyproject.toml` with dependencies
  - torch
  - transformers
  - steering-vectors
  - mcp SDK
- [ ] Organize source code:
  ```
  packages/elpis-inference/
  ├── src/elpis/
  │   ├── server.py          # MCP server
  │   ├── emotion/
  │   │   ├── state.py
  │   │   └── regulator.py
  │   ├── llm/
  │   │   ├── transformers_inference.py
  │   │   └── steering.py
  │   └── config/
  │       └── settings.py
  ├── data/
  │   └── emotion_vectors/   # Pre-trained vectors
  ├── tests/
  └── README.md
  ```

#### 2.2 MCP Server Implementation
- [ ] Implement `ElpisInferenceServer` class
- [ ] Register MCP tools:
  - `generate` - Text generation with emotion
  - `generate_stream` - Streaming generation
  - `function_call` - Tool calling
  - `emotion_event` - Trigger emotional event
  - `emotion_get` - Get current state
  - `emotion_reset` - Reset to baseline
  - `emotion_set_baseline` - Set personality
- [ ] Register MCP resources:
  - `emotion://state` - Current emotional state
  - `emotion://events` - Available events
  - `emotion://config` - Configuration
- [ ] Create CLI entry point: `elpis-server`

#### 2.3 Documentation
- [ ] Write comprehensive README.md
  - Installation instructions
  - Quick start guide
  - MCP tool reference
  - Configuration guide
  - Tuning guide (layers, strength, weights)
- [ ] Write API documentation
- [ ] Create example MCP client code
- [ ] Document emotion vector training process

#### 2.4 Testing
- [ ] Unit tests for all MCP tools
- [ ] Integration test with MCP client
- [ ] Test with Claude Desktop MCP config
- [ ] Test with opencode (if available)
- [ ] Performance tests (latency, throughput)

**Success Criteria:**
- `pip install ./packages/elpis-inference` works
- `elpis-server` starts and responds to MCP calls
- Can be used from Claude Desktop config
- All tests pass
- Documentation complete

---

### Phase 3: Extract mnemosyne Package (2 weeks)

**Goal:** Build memory system as standalone MCP server

**Tasks:**

#### 3.1 Package Structure
- [ ] Create `packages/mnemosyne/` directory
- [ ] Set up `pyproject.toml` with dependencies
  - chromadb
  - sentence-transformers
  - mcp SDK
- [ ] Organize source code:
  ```
  packages/mnemosyne/
  ├── src/mnemosyne/
  │   ├── server.py          # MCP server
  │   ├── types.py           # Memory, EmotionalContext
  │   ├── store.py           # ChromaDB storage
  │   ├── encoder.py         # Memory encoding
  │   ├── retriever.py       # Memory retrieval
  │   ├── consolidator.py    # Sleep consolidation
  │   └── config/
  │       └── settings.py
  ├── tests/
  └── README.md
  ```

#### 3.2 Memory System Implementation
- [ ] Implement `Memory` dataclass
- [ ] Implement `EmotionalContext` dataclass
- [ ] Implement `MemoryStore` with ChromaDB
  - Short-term buffer (in-memory)
  - Long-term storage (ChromaDB)
  - Embedding generation
  - Similarity search
  - Emotional congruence search
- [ ] Implement `MemoryEncoder`
  - Conversation encoding
  - Insight encoding
  - Skill encoding
- [ ] Implement `MemoryRetriever`
  - Multi-strategy retrieval
  - Token budget management
  - Context formatting
- [ ] Implement `SleepConsolidator`
  - Memory clustering
  - Importance scoring
  - (Harness will handle LLM-based insight generation)

#### 3.3 MCP Server Implementation
- [ ] Implement `MnemosyneServer` class
- [ ] Register MCP tools:
  - `memory_store` - Store memory
  - `memory_store_conversation` - Store conversation
  - `memory_query` - Semantic search
  - `memory_recent` - Get recent memories
  - `memory_get_context` - Formatted context
  - `memory_consolidate` - Run sleep cycle
  - `memory_should_consolidate` - Check if needed
  - `memory_delete` - Delete memory
  - `memory_stats` - Get statistics
- [ ] Register MCP resources:
  - `memory://stats` - Current statistics
  - `memory://recent` - Recent memories
  - `memory://config` - Configuration
- [ ] Create CLI entry point: `mnemosyne-server`

#### 3.4 Documentation
- [ ] Write comprehensive README.md
  - Installation instructions
  - Quick start guide
  - MCP tool reference
  - Memory types explanation
  - Consolidation guide
- [ ] Document consolidation orchestration pattern
- [ ] Create example harness code

#### 3.5 Testing
- [ ] Unit tests for all components
- [ ] Integration tests with ChromaDB
- [ ] MCP tool tests
- [ ] Test consolidation workflow
- [ ] Performance tests (embedding, retrieval)

**Success Criteria:**
- `pip install ./packages/mnemosyne` works
- `mnemosyne-server` starts and responds to MCP calls
- Can store and retrieve memories
- Consolidation works (harness-orchestrated)
- All tests pass

---

### Phase 4: Update Psyche Harness (1 week)

**Goal:** Transform Psyche into thin orchestration layer using both MCP servers

**Tasks:**

#### 4.1 Psyche Package Restructure
- [ ] Create `packages/psyche/` directory
- [ ] Set up `pyproject.toml` with dependencies
  - elpis-inference
  - mnemosyne
  - textual (for TUI)
  - mcp SDK
- [ ] Organize source code:
  ```
  packages/psyche/
  ├── src/psyche/
  │   ├── __init__.py
  │   ├── harness.py         # Main orchestration
  │   ├── client/
  │   │   ├── app.py         # Textual TUI
  │   │   └── widgets/
  │   ├── mcp_manager.py     # Manages connections to servers
  │   └── config/
  │       └── settings.py
  ├── tests/
  └── README.md
  ```

#### 4.2 MCP Client Implementation
- [ ] Implement `MCPManager` to connect to multiple servers
- [ ] Add connection to elpis-inference
- [ ] Add connection to mnemosyne
- [ ] Implement graceful degradation (works if either server unavailable)
- [ ] Add auto-spawn of subprocess servers

#### 4.3 Orchestration Logic
- [ ] Update conversation loop to use MCP servers
  ```python
  # Get emotion state
  emotion = await elpis.call("emotion_get")

  # Get memory context
  context = await mnemosyne.call("memory_get_context", {
      "query": user_input,
      "emotional_state": emotion
  })

  # Generate response
  response = await elpis.call("generate", {
      "messages": build_messages(context, user_input)
  })

  # Store conversation
  await mnemosyne.call("memory_store_conversation", {
      "messages": messages,
      "emotional_context": response["emotion"]
  })
  ```

- [ ] Implement sleep consolidation orchestration
  ```python
  # Check if sleep needed
  should_sleep = await mnemosyne.call("memory_should_consolidate")

  if should_sleep:
      # Get memories to consolidate
      memories = await mnemosyne.call("memory_get_buffer")

      # Generate insights using elpis
      for cluster in cluster_memories(memories):
          insight = await elpis.call("generate", {
              "messages": [{"role": "system",
                           "content": f"Analyze: {cluster}"}]
          })
          await mnemosyne.call("memory_store", {
              "content": insight,
              "memory_type": "semantic"
          })
  ```

- [ ] Update emotional event handling
- [ ] Add error handling for server unavailability

#### 4.4 Configuration
- [ ] Create default MCP config for spawning servers
- [ ] Add `psyche setup` command to auto-configure
- [ ] Support manual MCP config override

#### 4.5 Testing
- [ ] Integration tests with both servers
- [ ] Test graceful degradation
- [ ] Test sleep consolidation workflow
- [ ] UI tests
- [ ] End-to-end conversation tests

**Success Criteria:**
- Psyche works with modular architecture
- All existing functionality preserved
- Setup is straightforward for new users
- Tests pass

---

### Phase 5: Documentation & Polish (1 week)

**Goal:** Production-ready packages with excellent documentation

**Tasks:**

#### 5.1 Documentation
- [ ] Write migration guide from monolithic Elpis
- [ ] Create usage patterns documentation
  - Full stack (Psyche-style)
  - Emotion only
  - Memory only
  - Hybrid configurations
- [ ] Document MCP integration for other harnesses
  - Claude Desktop example
  - opencode example
  - Custom harness template
- [ ] Write tuning guides
  - Steering layer selection
  - Steering strength tuning
  - Emotion weights adjustment
  - Memory consolidation parameters
- [ ] Create troubleshooting guide
- [ ] Add architecture diagrams

#### 5.2 Examples & Templates
- [ ] Create example scripts
  - Standalone emotion inference
  - Memory system with Claude API
  - Custom harness skeleton
- [ ] Add Jupyter notebooks for experimentation
- [ ] Create deployment templates
  - Docker compose setup
  - systemd service files

#### 5.3 Packaging
- [ ] Finalize `pyproject.toml` for all packages
- [ ] Add package metadata (author, license, etc.)
- [ ] Include pre-trained emotion vectors in package
- [ ] Create meta-package `elpis-stack` for easy installation
  ```bash
  pip install elpis-stack  # Installs all three
  ```

#### 5.4 Testing & CI
- [ ] Set up pytest for all packages
- [ ] Add GitHub Actions CI
  - Run tests on PR
  - Check code formatting
  - Type checking with mypy
- [ ] Add coverage reporting
- [ ] Integration test matrix (different LLMs, configurations)

#### 5.5 Performance & Optimization
- [ ] Profile critical paths
- [ ] Optimize embedding generation
- [ ] Add caching where appropriate
- [ ] Document performance characteristics
- [ ] Provide CPU vs GPU benchmarks

**Success Criteria:**
- Documentation is comprehensive and clear
- Setup experience is smooth
- CI pipeline is green
- Performance is acceptable
- Ready for external users

---

## Testing Strategy

### Unit Tests
- Emotion state calculations
- Steering coefficient generation
- Memory encoding/retrieval
- Consolidation clustering
- MCP tool handlers

### Integration Tests
- Elpis-inference with LLM
- Mnemosyne with ChromaDB
- Psyche with both servers
- MCP protocol compliance

### End-to-End Tests
- Full conversation with memory
- Sleep consolidation cycle
- Emotional dynamics over time
- Graceful degradation scenarios

### Performance Tests
- Generation latency
- Memory retrieval speed
- Consolidation time
- Resource usage (CPU, RAM, disk)

### Compatibility Tests
- Different LLM models
- Different MCP harnesses
- Different Python versions (3.10+)

---

## Risk Mitigation

### Risk 1: Performance Regression
**Concern:** Transformers slower than llama-cpp on CPU

**Mitigation:**
- Benchmark early and often
- Document performance characteristics
- Provide optimization guide
- Consider optional llama-cpp fallback in Phase 2+

### Risk 2: Complex Setup
**Concern:** Three packages harder to install/configure

**Mitigation:**
- Meta-package for one-command install
- `psyche setup` auto-configuration
- Excellent documentation
- Docker/container options

### Risk 3: Emotional Feedback Loops
**Concern:** Response analysis could cause runaway emotions

**Mitigation:**
- Defer response analysis to later phase
- Add caps and damping
- Make it optional/experimental
- Document risks clearly

### Risk 4: Memory Consolidation Complexity
**Concern:** Harness-orchestrated consolidation is complex

**Mitigation:**
- Provide reference implementation in Psyche
- Document pattern clearly
- Consider helper library for common patterns
- Make consolidation optional

### Risk 5: Backward Compatibility
**Concern:** Breaking existing Psyche users

**Mitigation:**
- Keep monolithic version on `main` branch
- Modular version on separate branch initially
- Provide migration guide
- Version bump (2.0.0) indicates breaking changes

---

## Rollback Plan

If major issues arise during implementation:

### Option 1: Pause and Fix
- Stop new feature work
- Focus on resolving blockers
- Resume when stable

### Option 2: Partial Rollback
- Keep completed phases
- Revert problematic phase
- Reassess approach

### Option 3: Full Rollback
- Merge valuable code improvements to `main`
- Abandon modular architecture
- Document lessons learned

**Decision Point:** End of each phase - go/no-go decision

---

## Success Metrics

### Technical Metrics
- [ ] All tests pass (target: 90%+ coverage)
- [ ] Generation latency < 2x current
- [ ] Memory retrieval < 100ms
- [ ] Setup time < 10 minutes

### Quality Metrics
- [ ] Documentation complete and clear
- [ ] No critical bugs
- [ ] Passes code review
- [ ] Performance acceptable

### Adoption Metrics (Post-Launch)
- External harness integrations
- Community feedback positive
- Issue/PR activity healthy

---

## Timeline Estimate

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 0: Pre-Implementation | 1 week | None |
| Phase 1: Internal Modularization | 2 weeks | Phase 0 |
| Phase 2: elpis-inference | 2 weeks | Phase 1 |
| Phase 3: mnemosyne | 2 weeks | Phase 1 |
| Phase 4: Psyche Update | 1 week | Phases 2, 3 |
| Phase 5: Polish | 1 week | Phase 4 |
| **Total** | **9 weeks** | |

**Note:** Phases 2 and 3 can partially overlap (2-3 weeks saved)

**Realistic Timeline:** 6-7 weeks full-time, or 3-4 months part-time

---

## Open Questions

1. **Steering Vector Tuning:** What layer/strength works best for Llama 3.1 8B?
   - **Action:** Empirical testing during Phase 1

2. **Response Analysis:** Include or defer?
   - **Action:** Decide in Phase 0 decision meeting

3. **llama-cpp Backward Compat:** Support both backends?
   - **Action:** Decide based on Phase 1 performance tests

4. **Consolidation Frequency:** How often should sleep cycles run?
   - **Action:** Tune during Phase 4 testing

5. **Emotion Vector Distribution:** Ship in package or separate download?
   - **Action:** Decision 3 resolved (ship with package)

---

## Next Steps

### Immediate (This Week)
1. **Decision Meeting:** Resolve critical decisions 1-4
2. **Environment Setup:** Install dependencies, train initial vectors
3. **Kickoff Phase 1:** Begin internal modularization

### Short Term (Next 2 Weeks)
1. Complete Phase 1 (Internal Modularization)
2. Begin Phase 2 (elpis-inference extraction)

### Medium Term (Next 6-8 Weeks)
1. Complete all implementation phases
2. Begin testing with external harnesses
3. Iterate based on feedback

---

## Conclusion

This refactoring transforms Elpis from a standalone project into **composable emotional AI infrastructure**. The modular design enables:

- Broader adoption (any MCP harness can use components)
- Faster innovation (components evolve independently)
- Clearer architecture (well-defined APIs)
- Easier maintenance (smaller, focused codebases)

**Key Risks:** Performance, complexity, backward compatibility
**Key Mitigations:** Thorough testing, excellent docs, phased rollout

**Recommendation:** Proceed with implementation, starting with Phase 0 decision resolution.

---

**Plan Version:** 1.0
**Last Updated:** 2026-01-14
**Author:** Claude (Opus 4)
**Status:** Ready for Review
