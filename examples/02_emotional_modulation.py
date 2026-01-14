#!/usr/bin/env python3
"""Emotional modulation example: Using EmotionalState with sampling parameters.

This example demonstrates how emotional state affects sampling parameters
(temperature and top_p) in the llama-cpp backend.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from elpis.config.settings import ModelSettings
from elpis.llm.inference import LlamaInference
from elpis.emotion.state import EmotionalState
from elpis.emotion.regulation import HomeostasisRegulator


async def generate_with_emotion(llm, emotion_state, prompt: str):
    """Generate a response using the given emotional state."""
    # Get modulated parameters based on emotional state
    params = emotion_state.get_modulated_params()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    response = await llm.chat_completion(
        messages=messages,
        max_tokens=100,
        temperature=params["temperature"],
        top_p=params["top_p"],
    )

    return response, params


async def main():
    """Run emotional modulation example."""
    print("=== Emotional Modulation Example ===\n")

    # Load model
    settings = ModelSettings(
        backend="llama-cpp",
        path="./data/models/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf",
        context_length=8192,
        gpu_layers=35,
    )

    print("Loading model...")
    try:
        llm = LlamaInference(settings)
        print("✅ Model loaded\n")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Test prompt
    prompt = "Tell me about the weather today."

    # Test different emotional states
    test_cases = [
        ("Neutral", EmotionalState(valence=0.0, arousal=0.0)),
        ("Excited", EmotionalState(valence=0.8, arousal=0.8)),
        ("Calm", EmotionalState(valence=0.6, arousal=-0.6)),
        ("Frustrated", EmotionalState(valence=-0.7, arousal=0.7)),
        ("Depleted", EmotionalState(valence=-0.5, arousal=-0.8)),
    ]

    print(f"Prompt: \"{prompt}\"\n")
    print("=" * 80)

    for name, state in test_cases:
        print(f"\n{name} State (valence={state.valence:+.1f}, arousal={state.arousal:+.1f})")
        print("-" * 80)

        try:
            response, params = await generate_with_emotion(llm, state, prompt)

            print(f"Quadrant: {state.get_quadrant()}")
            print(f"Temperature: {params['temperature']:.2f}")
            print(f"Top-p: {params['top_p']:.2f}")
            print(f"\nResponse: {response[:200]}...")  # Show first 200 chars

        except Exception as e:
            print(f"❌ Generation failed: {e}")
            continue

    print("\n" + "=" * 80)
    print("\n✅ Comparison complete!")
    print("\nNote: Emotional modulation affects sampling parameters:")
    print("  - Higher arousal → Lower temperature (more focused)")
    print("  - Higher valence → Higher top_p (broader sampling)")


if __name__ == "__main__":
    asyncio.run(main())
