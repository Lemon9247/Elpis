#!/usr/bin/env python3
"""Steering vectors example: Using TransformersInference with emotional steering.

This example demonstrates how to use the transformers backend with trained
emotion steering vectors for more nuanced emotional expression.

Prerequisites:
  1. Install transformers: pip install torch transformers
  2. Train emotion vectors: python scripts/train_emotion_vectors.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from elpis.config.settings import ModelSettings
from elpis.emotion.state import EmotionalState


async def main():
    """Run steering vectors example."""
    print("=== Steering Vectors Example ===\n")

    # Check if transformers is available
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("❌ transformers/torch not installed")
        print("\nInstall with: pip install torch transformers")
        return

    # Check if emotion vectors exist
    vectors_dir = Path("./data/emotion_vectors")
    if not vectors_dir.exists():
        print(f"❌ Emotion vectors not found at {vectors_dir}")
        print("\nTrain vectors first:")
        print("  python scripts/train_emotion_vectors.py \\")
        print("    --model meta-llama/Llama-3.1-8B-Instruct \\")
        print("    --layer 15 \\")
        print("    --output ./data/emotion_vectors")
        return

    required_vectors = ["excited.pt", "frustrated.pt", "calm.pt", "depleted.pt"]
    missing = [v for v in required_vectors if not (vectors_dir / v).exists()]
    if missing:
        print(f"❌ Missing emotion vectors: {', '.join(missing)}")
        print(f"\nFound in {vectors_dir}:")
        for f in vectors_dir.glob("*.pt"):
            print(f"  - {f.name}")
        return

    print(f"✅ Found all 4 emotion vectors in {vectors_dir}\n")

    # Configure transformers backend
    settings = ModelSettings(
        backend="transformers",
        path="meta-llama/Llama-3.1-8B-Instruct",  # HuggingFace model ID
        context_length=8192,
        torch_dtype="bfloat16",  # Use bfloat16 for efficiency
        steering_layer=15,  # Layer where steering is applied
        emotion_vectors_dir=str(vectors_dir),
        hardware_backend="auto",  # Use CUDA if available
    )

    print(f"Model: {settings.path}")
    print(f"Backend: {settings.backend}")
    print(f"Steering layer: {settings.steering_layer}")
    print(f"Dtype: {settings.torch_dtype}")
    print()

    # Import and initialize
    from elpis.llm.transformers_inference import TransformersInference

    print("Loading model (this may take a minute)...")
    try:
        llm = TransformersInference(settings)
        print("✅ Model loaded successfully\n")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Test different emotional states
    prompt = "How do you feel about solving complex problems?"

    test_cases = [
        ("Neutral", EmotionalState(valence=0.0, arousal=0.0)),
        ("Very Excited", EmotionalState(valence=0.9, arousal=0.9)),
        ("Calm & Content", EmotionalState(valence=0.7, arousal=-0.6)),
        ("Frustrated", EmotionalState(valence=-0.8, arousal=0.8)),
        ("Depleted", EmotionalState(valence=-0.7, arousal=-0.7)),
    ]

    print(f"Prompt: \"{prompt}\"\n")
    print("=" * 80)

    for name, state in test_cases:
        print(f"\n{name} (valence={state.valence:+.1f}, arousal={state.arousal:+.1f})")
        print("-" * 80)

        # Get steering coefficients from emotional state
        coeffs = state.get_steering_coefficients()
        dominant, strength = state.get_dominant_emotion()

        print(f"Dominant emotion: {dominant} (strength: {strength:.2f})")
        print(f"Steering coefficients:")
        for emotion, coeff in coeffs.items():
            bar = "█" * int(coeff * 30)
            print(f"  {emotion:12s}: {coeff:.3f}  {bar}")

        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt},
        ]

        try:
            # Generate with steering
            print("\nGenerating response with emotional steering...")
            response = await llm.chat_completion(
                messages=messages,
                max_tokens=150,
                temperature=0.7,
                emotion_coefficients=coeffs,
            )

            print(f"\nResponse:\n{response}")

        except Exception as e:
            print(f"❌ Generation failed: {e}")
            continue

    print("\n" + "=" * 80)
    print("\n✅ Steering vector demonstration complete!")
    print("\nNote: Responses should reflect the emotional state through:")
    print("  - Word choice and tone")
    print("  - Energy level and enthusiasm")
    print("  - Perspective and framing")


if __name__ == "__main__":
    asyncio.run(main())
