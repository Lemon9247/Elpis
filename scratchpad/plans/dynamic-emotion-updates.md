# Plan: Dynamic Emotion Updates for Elpis

**Branch:** `feature/dynamic-emotion-updates`
**Created:** 2026-01-25

## Overview

The current emotion system uses static event mappings and simple keyword heuristics. This plan introduces dynamic emotion updates through context-aware processing, event compounding, mood inertia, and behavioral monitoring.

## Current Limitations

1. **Keyword heuristics** - `process_response()` looks for words like "successfully", "error" (no real analysis)
2. **Static event effects** - EVENT_MAPPINGS are hardcoded; same effect regardless of context
3. **Uniform decay** - All emotions decay identically toward baseline
4. **No compounding** - Repeated failures don't build frustration
5. **No behavioral monitoring** - Tool failures, retry loops not detected
6. **Early return** - First keyword match triggers, ignores rest of content

## Implementation Phases

### Phase 1: Enhanced Response Analysis (regulation.py)
**Files:** `src/elpis/emotion/regulation.py`

Replace single-keyword matching with multi-factor weighted scoring:
- Score multiple emotion indicators in content (not just first match)
- Weight indicators: success_words, error_words, frustration_words, exploration_words
- Trigger dominant emotion only if above threshold
- Detect frustration patterns ("still", "again", "yet another" + error words)

### Phase 2: Context-Aware Event Intensity (regulation.py)
**Files:** `src/elpis/emotion/regulation.py`, `src/elpis/config/settings.py`

Add `EventHistory` class to track recent events and modify intensity:
- **Event compounding**: Repeated failures increase intensity (1.0 → 1.2 → 1.4, cap 2.0)
- **Success dampening**: Repeated successes decrease intensity (1.0 → 0.8 → 0.6, floor 0.5)
- Track events with timestamps, trim old events (>10 min)

### Phase 3: Mood Inertia (regulation.py)
**Files:** `src/elpis/emotion/regulation.py`

Add resistance to rapid emotional swings based on trajectory:
- Events aligned with current momentum: slight boost (1.1x)
- Events counter to strong momentum: resistance (0.6x-0.8x)
- Uses existing trajectory tracking from `state.get_trajectory()`

### Phase 4: Emotion-Specific Decay Rates (regulation.py, settings.py)
**Files:** `src/elpis/emotion/regulation.py`, `src/elpis/config/settings.py`

Add per-quadrant decay multipliers:
- `decay_multiplier_frustrated: 0.7` (frustration persists longer)
- `decay_multiplier_depleted: 0.8` (depletion persists)
- `decay_multiplier_calm: 1.2` (calm decays slightly faster)
- `decay_multiplier_excited: 1.0` (baseline)

### Phase 5: Behavioral Monitoring (new file + server.py)
**Files:** `src/elpis/emotion/behavioral_monitor.py` (new), `src/elpis/server.py`

New `BehavioralMonitor` class that detects:
- **Retry loops**: Same tool called 3+ times → frustration event
- **Failure streaks**: 2+ consecutive failures → compounding error events
- **Long generations**: >30s generation → mild blocked event
- **Idle periods**: >2min since last response → calming idle event

Hook into server's tool execution and generation lifecycle.

### Phase 6: LLM-Based Emotion Analysis
**Files:** `src/elpis/emotion/sentiment.py` (new)

Two options:
- **A) Local sentiment model**: Load small DistilBERT sentiment classifier at startup (cheap, fast)
- **B) LLM self-analysis**: Use Elpis's own inference for emotion analysis (more expensive, optional)

Only analyze responses >200 chars that pass keyword pre-filter.

## Configuration Additions (settings.py)

```python
# Context-aware intensity
streak_compounding_enabled: bool = True
streak_compounding_factor: float = 0.2  # per-event increase

# Mood inertia
mood_inertia_enabled: bool = True
mood_inertia_resistance: float = 0.4  # max resistance factor

# Quadrant decay multipliers
decay_multiplier_excited: float = 1.0
decay_multiplier_frustrated: float = 0.7
decay_multiplier_calm: float = 1.2
decay_multiplier_depleted: float = 0.8

# Behavioral monitoring
behavioral_monitoring_enabled: bool = True
retry_loop_threshold: int = 3
failure_streak_threshold: int = 2
long_generation_seconds: float = 30.0

# LLM-based emotion analysis
llm_emotion_analysis_enabled: bool = True
llm_analysis_min_length: int = 200  # Min response length to analyze
use_local_sentiment_model: bool = True  # Use lightweight DistilBERT vs full LLM
```

## Critical Files

| File | Changes |
|------|---------|
| `src/elpis/emotion/regulation.py` | EventHistory, inertia, enhanced process_response, quadrant decay |
| `src/elpis/config/settings.py` | New configuration fields |
| `src/elpis/emotion/behavioral_monitor.py` | New file for behavioral monitoring |
| `src/elpis/emotion/sentiment.py` | New file for LLM/sentiment-based emotion analysis |
| `src/elpis/server.py` | BehavioralMonitor integration hooks, sentiment analysis integration |
| `tests/elpis/unit/test_emotion.py` | New tests for all features |

## Implementation Order

1. **Phase 1** (Enhanced response analysis) - Lowest risk, immediate improvement
2. **Phase 2** (Context-aware intensity) - Core dynamic behavior
3. **Phase 3** (Mood inertia) - Complements Phase 2
4. **Phase 4** (Emotion-specific decay) - Simple config-driven change
5. **Phase 5** (Behavioral monitoring) - Larger but valuable addition
6. **Phase 6** (LLM analysis) - Enhanced inference from response content

## Verification

1. **Unit tests** for each component:
   - EventHistory streak calculations
   - Inertia multiplier with various trajectory states
   - Enhanced process_response scoring
   - Quadrant-specific decay rates
   - BehavioralMonitor event detection

2. **Integration tests**:
   - Simulate tool failure sequences → verify frustration compounds
   - Simulate success sequences → verify dampening
   - Verify decay differs by quadrant
   - Test sentiment analysis on varied response content

3. **Manual testing**:
   - Use existing `scripts/emotion_repl.py` or add test commands
   - Observe emotional dynamics in actual Psyche sessions
