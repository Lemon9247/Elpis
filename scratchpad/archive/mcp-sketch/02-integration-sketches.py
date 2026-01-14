"""
Steering Vector Integration Sketches for Elpis
===============================================

This file contains sketches and modifications needed to integrate
steering vector-based emotional modulation into Elpis.

Contents:
1. Updated EmotionalState class with steering coefficient output
2. Updated HomeostasisRegulator with response analysis hooks
3. Updated MCP server tools for the new inference backend
4. Configuration updates
5. Migration guide and notes

These are sketches/templates - adapt as needed for your codebase!
"""

# =============================================================================
# 1. UPDATED EMOTIONAL STATE (src/elpis/emotion/state.py)
# =============================================================================

"""
Changes to EmotionalState:
- Add get_steering_coefficients() method
- Keep get_modulated_params() for backwards compatibility / fallback
- Add optional steering strength scaling
"""

from dataclasses import dataclass, field
from typing import Dict, Any
import time


@dataclass
class EmotionalState:
    """
    2D Emotional state using Valence-Arousal model.

    Now supports both sampling parameter modulation (legacy)
    and steering vector coefficient output (new).
    """

    valence: float = 0.0
    arousal: float = 0.0
    last_update: float = field(default_factory=time.time)
    update_count: int = 0

    baseline_valence: float = 0.0
    baseline_arousal: float = 0.0

    # NEW: Global steering strength (can be adjusted per-personality)
    steering_strength: float = 1.0

    def __post_init__(self) -> None:
        self.valence = max(-1.0, min(1.0, self.valence))
        self.arousal = max(-1.0, min(1.0, self.arousal))

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary representation."""
        return {
            "valence": round(self.valence, 3),
            "arousal": round(self.arousal, 3),
            "quadrant": self.get_quadrant(),
            "last_update": self.last_update,
            "update_count": self.update_count,
            "baseline": {
                "valence": self.baseline_valence,
                "arousal": self.baseline_arousal,
            },
            # NEW: Include steering info
            "steering_coefficients": self.get_steering_coefficients(),
        }

    def get_quadrant(self) -> str:
        """Return named emotional quadrant."""
        if self.arousal >= 0:
            return "excited" if self.valence >= 0 else "frustrated"
        else:
            return "calm" if self.valence >= 0 else "depleted"

    def get_modulated_params(self) -> Dict[str, float]:
        """
        Legacy: Convert emotional state to LLM sampling parameters.

        Keep this for backwards compatibility or as a fallback
        when steering vectors aren't available.
        """
        base_temp = 0.7
        base_top_p = 0.9

        temp_delta = -0.2 * self.arousal
        temperature = max(0.1, min(1.5, base_temp + temp_delta))

        top_p_delta = 0.1 * self.valence
        top_p = max(0.5, min(1.0, base_top_p + top_p_delta))

        return {
            "temperature": round(temperature, 2),
            "top_p": round(top_p, 2),
        }

    # NEW METHOD
    def get_steering_coefficients(self) -> Dict[str, float]:
        """
        Convert emotional state to steering vector blend coefficients.

        Maps the 2D valence-arousal space to weights for each quadrant's
        steering vector. The coefficients sum to 1.0 and represent how
        much of each "pure" emotional state to blend.

        Returns:
            Dictionary mapping emotion names to blend weights (0.0 to 1.0)
        """
        # Normalize valence/arousal from [-1, 1] to [0, 1]
        v = (self.valence + 1.0) / 2.0  # 0 = negative, 1 = positive
        a = (self.arousal + 1.0) / 2.0  # 0 = low, 1 = high

        # Compute quadrant weights using bilinear interpolation
        # This gives smooth blending between adjacent emotional states
        coefficients = {
            "excited": v * a,               # high valence, high arousal
            "frustrated": (1.0 - v) * a,    # low valence, high arousal
            "calm": v * (1.0 - a),          # high valence, low arousal
            "depleted": (1.0 - v) * (1.0 - a),  # low valence, low arousal
        }

        # Apply global steering strength
        # (allows personality-based scaling of emotional expression)
        if self.steering_strength != 1.0:
            coefficients = {
                k: v * self.steering_strength
                for k, v in coefficients.items()
            }

        return coefficients

    # NEW METHOD
    def get_dominant_emotion(self) -> tuple[str, float]:
        """
        Get the strongest emotional component.

        Useful for logging, debugging, or UI display.

        Returns:
            Tuple of (emotion_name, coefficient)
        """
        coefficients = self.get_steering_coefficients()
        return max(coefficients.items(), key=lambda x: x[1])

    def reset(self) -> None:
        self.valence = self.baseline_valence
        self.arousal = self.baseline_arousal
        self.last_update = time.time()

    def shift(self, valence_delta: float, arousal_delta: float) -> None:
        self.valence = max(-1.0, min(1.0, self.valence + valence_delta))
        self.arousal = max(-1.0, min(1.0, self.arousal + arousal_delta))
        self.last_update = time.time()
        self.update_count += 1

    def distance_from_baseline(self) -> float:
        valence_diff = self.valence - self.baseline_valence
        arousal_diff = self.arousal - self.baseline_arousal
        return (valence_diff**2 + arousal_diff**2) ** 0.5


# =============================================================================
# 2. UPDATED MCP SERVER (src/elpis/mcp/server.py or similar)
# =============================================================================

"""
Key changes to MCP server:
- Initialize TransformersInference instead of LlamaInference
- Load emotion vectors on startup
- Pass steering coefficients to inference calls
- Add new tools for steering vector management (optional)
"""

from typing import Any, Dict, List, Optional
# from mcp import Server  # Your MCP server base
# from elpis.emotion.state import EmotionalState
# from elpis.emotion.regulator import HomeostasisRegulator
# from elpis.llm.transformers_inference import TransformersInference, EmotionalSteeringConfig


class ElpisServer:
    """
    MCP Server with steering vector emotional modulation.

    This is a sketch of the key integration points - adapt to your
    actual server structure.
    """

    def __init__(self, config: "ElpisConfig"):
        self.config = config

        # Initialize emotional state and regulator (unchanged)
        self.emotional_state = EmotionalState(
            baseline_valence=config.baseline_valence,
            baseline_arousal=config.baseline_arousal,
            steering_strength=config.steering_strength,
        )
        self.regulator = HomeostasisRegulator(
            state=self.emotional_state,
            decay_rate=config.decay_rate,
        )

        # NEW: Initialize transformers inference with steering support
        self.inference = self._init_inference()

    def _init_inference(self) -> "TransformersInference":
        """Initialize the inference engine with steering vectors."""
        from elpis.llm.transformers_inference import (
            TransformersInference,
            EmotionalSteeringConfig,
        )

        # Configure steering
        steering_config = EmotionalSteeringConfig(
            steering_layer=self.config.steering_layer,
            steering_strength=1.0,  # State handles its own scaling
            emotion_weights={
                "excited": self.config.excited_weight,
                "frustrated": self.config.frustrated_weight,
                "calm": self.config.calm_weight,
                "depleted": self.config.depleted_weight,
            },
        )

        # Create inference engine
        engine = TransformersInference(
            model_name_or_path=self.config.model_path,
            device=self.config.device,
            torch_dtype=self.config.torch_dtype,
            context_length=self.config.context_length,
            steering_config=steering_config,
        )

        # Load pre-trained emotion vectors
        if self.config.emotion_vectors_path:
            engine.load_emotion_vectors(self.config.emotion_vectors_path)
        else:
            # Train on first run (slow - better to pre-train and save)
            from elpis.llm.transformers_inference import EXAMPLE_EMOTION_PROMPTS
            engine.train_emotion_vectors(EXAMPLE_EMOTION_PROMPTS)
            engine.save_emotion_vectors("./data/emotion_vectors")

        return engine

    # -------------------------------------------------------------------------
    # MCP Tool: generate
    # -------------------------------------------------------------------------

    async def tool_generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate text with emotional modulation.

        This replaces your existing generate tool.
        """
        # Apply decay and get current emotional parameters
        self.regulator._apply_decay()

        # Get steering coefficients from emotional state
        steering_coefficients = self.emotional_state.get_steering_coefficients()

        # Also get sampling params (can still use these alongside steering)
        sampling_params = self.emotional_state.get_modulated_params()

        # Log emotional state for debugging
        dominant_emotion, strength = self.emotional_state.get_dominant_emotion()
        # logger.debug(f"Generating with emotion: {dominant_emotion} ({strength:.2f})")

        # Generate with steering
        response = await self.inference.chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=sampling_params["temperature"],
            top_p=sampling_params["top_p"],
            emotion_coefficients=steering_coefficients,
        )

        # Analyze response for emotional feedback
        self.regulator.process_response(response)

        return {
            "content": response,
            "emotion": self.emotional_state.to_dict(),
        }

    # -------------------------------------------------------------------------
    # MCP Tool: generate_stream
    # -------------------------------------------------------------------------

    async def tool_generate_stream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
    ):
        """
        Streaming generation with emotional modulation.
        """
        self.regulator._apply_decay()

        steering_coefficients = self.emotional_state.get_steering_coefficients()
        sampling_params = self.emotional_state.get_modulated_params()

        full_response = []

        async for token in self.inference.chat_completion_stream(
            messages=messages,
            max_tokens=max_tokens,
            temperature=sampling_params["temperature"],
            top_p=sampling_params["top_p"],
            emotion_coefficients=steering_coefficients,
        ):
            full_response.append(token)
            yield token

        # Analyze complete response for emotional feedback
        self.regulator.process_response("".join(full_response))

    # -------------------------------------------------------------------------
    # MCP Tool: function_call (unchanged interface, new backend)
    # -------------------------------------------------------------------------

    async def tool_function_call(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Generate function/tool calls with emotional modulation.
        """
        self.regulator._apply_decay()

        steering_coefficients = self.emotional_state.get_steering_coefficients()
        sampling_params = self.emotional_state.get_modulated_params()

        return await self.inference.function_call(
            messages=messages,
            tools=tools,
            temperature=sampling_params["temperature"],
            emotion_coefficients=steering_coefficients,
        )


# =============================================================================
# 3. CONFIGURATION UPDATES (src/elpis/config/settings.py)
# =============================================================================

"""
New configuration fields for steering vector support.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SteeringConfig:
    """Configuration for emotional steering vectors."""

    # Whether to use steering vectors (vs legacy sampling params only)
    enabled: bool = True

    # Path to pre-trained emotion vectors (None = train on startup)
    vectors_path: Optional[str] = None

    # Which layer to apply steering (model-dependent, tune this!)
    # For Llama 3.1 8B, try layers 12-20
    layer: int = 15

    # Per-emotion weight multipliers (tune for desired expressiveness)
    emotion_weights: Dict[str, float] = field(default_factory=lambda: {
        "excited": 1.0,
        "frustrated": 1.0,
        "calm": 1.0,
        "depleted": 1.0,
    })


@dataclass
class ModelSettings:
    """Extended model settings with steering support."""

    # Existing fields
    path: str = "meta-llama/Llama-3.1-8B-Instruct"
    context_length: int = 8192
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

    # NEW: Backend selection
    # "llama_cpp" = original LlamaInference (faster CPU, no steering)
    # "transformers" = new TransformersInference (steering support)
    backend: str = "transformers"

    # NEW: Device configuration for transformers backend
    device: str = "auto"  # "auto", "cuda", "cpu"
    torch_dtype: str = "auto"  # "auto", "float16", "bfloat16", "float32"

    # NEW: Steering configuration
    steering: SteeringConfig = field(default_factory=SteeringConfig)


@dataclass
class EmotionSettings:
    """Extended emotion settings."""

    # Existing fields
    baseline_valence: float = 0.0
    baseline_arousal: float = 0.0
    decay_rate: float = 0.1
    max_delta: float = 0.5

    # NEW: Global steering strength multiplier
    # 0.0 = no emotional expression
    # 1.0 = normal expression
    # >1.0 = exaggerated expression (use carefully!)
    steering_strength: float = 1.0


# =============================================================================
# 4. EXAMPLE CONFIG FILE (configs/default.yaml)
# =============================================================================

EXAMPLE_CONFIG_YAML = """
# Elpis Configuration with Steering Vector Support

model:
  # Use transformers backend for steering support
  backend: transformers
  path: meta-llama/Llama-3.1-8B-Instruct
  context_length: 8192
  max_tokens: 512

  # Device settings
  device: auto  # auto, cuda, cpu
  torch_dtype: auto  # auto, float16, bfloat16, float32

  # Steering vector settings
  steering:
    enabled: true
    vectors_path: ./data/emotion_vectors  # null to train on startup
    layer: 15  # Tune this! Try 12-20 for Llama 3.1 8B
    emotion_weights:
      excited: 1.0
      frustrated: 1.0
      calm: 1.0
      depleted: 1.0

emotion:
  baseline_valence: 0.0
  baseline_arousal: 0.0
  decay_rate: 0.1
  max_delta: 0.5
  steering_strength: 1.0  # Global multiplier for emotional expression
"""


# =============================================================================
# 5. TRAINING SCRIPT (scripts/train_emotion_vectors.py)
# =============================================================================

"""
Standalone script to train and save emotion vectors.
Run this once before using Elpis with steering.
"""

TRAINING_SCRIPT = '''
#!/usr/bin/env python3
"""Train emotional steering vectors for Elpis."""

import argparse
from pathlib import Path

from elpis.llm.transformers_inference import (
    TransformersInference,
    EmotionalSteeringConfig,
    EXAMPLE_EMOTION_PROMPTS,
)


def main():
    parser = argparse.ArgumentParser(description="Train emotion steering vectors")
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model to train vectors for"
    )
    parser.add_argument(
        "--output",
        default="./data/emotion_vectors",
        help="Output directory for vectors"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=15,
        help="Layer to train steering vectors for"
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use (auto, cuda, cpu)"
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    engine = TransformersInference(
        model_name_or_path=args.model,
        device=args.device,
        steering_config=EmotionalSteeringConfig(steering_layer=args.layer),
    )

    print("Training emotion vectors...")
    engine.train_emotion_vectors(EXAMPLE_EMOTION_PROMPTS)

    print(f"Saving to: {args.output}")
    engine.save_emotion_vectors(args.output)

    print("Done!")


if __name__ == "__main__":
    main()
'''


# =============================================================================
# 6. MIGRATION GUIDE
# =============================================================================

MIGRATION_GUIDE = """
# Migration Guide: llama-cpp-python to Transformers + Steering Vectors

## Overview

This migration replaces the LlamaInference backend with TransformersInference,
enabling emotional modulation via steering vectors instead of just sampling
parameter adjustment.

## Steps

### 1. Install Dependencies

```bash
pip install torch transformers accelerate
pip install steering-vectors
```

### 2. Train Emotion Vectors (One-Time Setup)

```bash
python scripts/train_emotion_vectors.py \\
    --model meta-llama/Llama-3.1-8B-Instruct \\
    --output ./data/emotion_vectors \\
    --layer 15
```

This takes a few minutes and only needs to be done once per model.

### 3. Update Configuration

In your config file, change:

```yaml
model:
  backend: transformers  # was: llama_cpp
  steering:
    enabled: true
    vectors_path: ./data/emotion_vectors
    layer: 15
```

### 4. Update EmotionalState Class

Add the `get_steering_coefficients()` method (see sketch above).

### 5. Update MCP Server

Modify inference calls to pass `emotion_coefficients` parameter.

### 6. Test!

Run your test suite to verify everything works:

```bash
pytest tests/
```

## Tuning Guide

### Steering Layer

The optimal layer varies by model. For Llama 3.1 8B, try:
- Layer 12-15: More subtle effects
- Layer 16-20: Stronger effects
- Too early (< 10): May cause incoherence
- Too late (> 25): May have little effect

### Steering Strength

Start with 1.0 and adjust:
- 0.5: Subtle emotional coloring
- 1.0: Noticeable but natural
- 1.5: Strong emotional expression
- 2.0+: May cause repetition or incoherence

### Emotion Weights

Tune per-emotion weights if some emotions are over/under-expressed:
```yaml
emotion_weights:
  excited: 1.2    # Boost excitement
  frustrated: 0.8  # Tone down frustration
  calm: 1.0
  depleted: 0.7   # Reduce lethargy
```

### Contrastive Prompts

The quality of steering vectors depends heavily on the training prompts.
Improve them by:
- Using more examples (10-20 per emotion)
- Making contrasts clearer
- Including varied phrasings
- Testing and iterating

## Performance Notes

### CPU Performance

Transformers on CPU is slower than llama-cpp-python. Mitigations:
- Use smaller batch sizes
- Consider `torch.compile()` (PyTorch 2.0+)
- Look into `optimum` for Intel CPUs
- Plan for GPU upgrade (even a modest one helps significantly)

### Memory Usage

Transformers + steering may use more RAM:
- Llama 3.1 8B in float32: ~32GB
- Llama 3.1 8B in bfloat16: ~16GB
- Llama 3.1 8B in 8-bit: ~8GB (requires bitsandbytes)

### GPU Acceleration

When you get GPU hardware:
```yaml
model:
  device: cuda
  torch_dtype: bfloat16  # or float16 for older GPUs
```

## Rollback

If issues arise, you can temporarily revert to llama-cpp-python:
```yaml
model:
  backend: llama_cpp
```

The emotional state will fall back to `get_modulated_params()` for
sampling parameter adjustment.
"""


# =============================================================================
# 7. TESTS (tests/test_steering_integration.py)
# =============================================================================

"""
Example tests for steering integration.
"""

import pytest


class TestEmotionalStateSteering:
    """Tests for steering coefficient generation."""

    def test_steering_coefficients_sum(self):
        """Coefficients should sum to approximately 1.0."""
        state = EmotionalState(valence=0.5, arousal=-0.3)
        coeffs = state.get_steering_coefficients()

        total = sum(coeffs.values())
        assert 0.99 <= total <= 1.01, f"Coefficients sum to {total}, expected ~1.0"

    def test_excited_quadrant(self):
        """High valence + high arousal = mostly excited."""
        state = EmotionalState(valence=0.9, arousal=0.9)
        coeffs = state.get_steering_coefficients()

        assert coeffs["excited"] > 0.7
        assert coeffs["excited"] > coeffs["frustrated"]
        assert coeffs["excited"] > coeffs["calm"]
        assert coeffs["excited"] > coeffs["depleted"]

    def test_frustrated_quadrant(self):
        """Low valence + high arousal = mostly frustrated."""
        state = EmotionalState(valence=-0.9, arousal=0.9)
        coeffs = state.get_steering_coefficients()

        assert coeffs["frustrated"] > 0.7

    def test_calm_quadrant(self):
        """High valence + low arousal = mostly calm."""
        state = EmotionalState(valence=0.9, arousal=-0.9)
        coeffs = state.get_steering_coefficients()

        assert coeffs["calm"] > 0.7

    def test_depleted_quadrant(self):
        """Low valence + low arousal = mostly depleted."""
        state = EmotionalState(valence=-0.9, arousal=-0.9)
        coeffs = state.get_steering_coefficients()

        assert coeffs["depleted"] > 0.7

    def test_neutral_state_balanced(self):
        """Neutral state should have roughly equal coefficients."""
        state = EmotionalState(valence=0.0, arousal=0.0)
        coeffs = state.get_steering_coefficients()

        # All should be 0.25 at perfect center
        for emotion, coeff in coeffs.items():
            assert 0.2 <= coeff <= 0.3, f"{emotion} = {coeff}, expected ~0.25"

    def test_steering_strength_scaling(self):
        """Steering strength should scale all coefficients."""
        state = EmotionalState(valence=0.5, arousal=0.5, steering_strength=0.5)
        coeffs = state.get_steering_coefficients()

        # With strength 0.5, total should be ~0.5
        total = sum(coeffs.values())
        assert 0.45 <= total <= 0.55

    def test_dominant_emotion(self):
        """get_dominant_emotion should return highest coefficient."""
        state = EmotionalState(valence=0.8, arousal=0.8)
        emotion, strength = state.get_dominant_emotion()

        assert emotion == "excited"
        assert strength > 0.5


class TestSteeringIntegration:
    """Integration tests for steering with inference."""

    @pytest.fixture
    def mock_inference(self):
        """Create inference engine with mocked model."""
        # This would need actual mocking setup
        pass

    def test_steering_vectors_loaded(self, mock_inference):
        """Verify emotion vectors are loaded correctly."""
        pass

    def test_generation_with_steering(self, mock_inference):
        """Verify steering affects generation."""
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
