#!/usr/bin/env python3
"""Basic example: Using Elpis with llama-cpp backend.

This example shows the simplest way to use Elpis for inference with
the default llama-cpp backend (GGUF models).
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from elpis.config.settings import Settings, ModelSettings
from elpis.llm.inference import LlamaInference
from elpis.emotion.state import EmotionalState


async def main():
    """Run basic inference example."""
    print("=== Basic Elpis Inference Example ===\n")

    # Configure the model
    settings = ModelSettings(
        backend="llama-cpp",
        path="./data/models/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf",
        context_length=8192,
        gpu_layers=35,
        temperature=0.7,
        top_p=0.9,
    )

    print(f"Loading model: {settings.path}")
    print(f"Backend: {settings.backend}")
    print()

    # Initialize inference engine
    try:
        llm = LlamaInference(settings)
        print("✅ Model loaded successfully\n")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("\nMake sure you have:")
        print("1. Downloaded a GGUF model to ./data/models/")
        print("2. Installed llama-cpp-python: pip install llama-cpp-python")
        return

    # Create some example messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain what emotional intelligence means in 2-3 sentences."},
    ]

    print("Prompt:")
    print(f"  User: {messages[1]['content']}")
    print()

    # Generate response
    print("Generating response...")
    try:
        response = await llm.chat_completion(
            messages=messages,
            max_tokens=150,
            temperature=0.7,
        )

        print("\n" + "=" * 60)
        print("Response:")
        print("=" * 60)
        print(response)
        print("=" * 60)

    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return

    print("\n✅ Example completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
