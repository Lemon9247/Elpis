#!/usr/bin/env python3
"""
Train emotional steering vectors for Elpis.

This script trains steering vectors for the four emotional quadrants:
- excited (high valence, high arousal)
- frustrated (low valence, high arousal)
- calm (high valence, low arousal)
- depleted (low valence, low arousal)

The trained vectors are saved to data/emotion_vectors/ and can be used
by the TransformersInference engine for emotional modulation.

Requirements:
    pip install torch transformers steering-vectors

Usage:
    python scripts/train_emotion_vectors.py --model meta-llama/Llama-3.1-8B-Instruct

Options:
    --model: HuggingFace model ID or local path (default: meta-llama/Llama-3.1-8B-Instruct)
    --output: Output directory for vectors (default: ./data/emotion_vectors)
    --layer: Layer to train steering vectors for (default: 15)
    --device: Device to use (auto, cuda, cpu) (default: auto)
"""

import argparse
import sys
from pathlib import Path

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from steering_vectors import train_steering_vector
except ImportError as e:
    print(f"Error: Missing required dependency: {e}")
    print("\nPlease install required packages:")
    print("  pip install torch transformers steering-vectors")
    sys.exit(1)


# Example emotion prompts for training steering vectors
# Users can customize these for their specific use case
EMOTION_PROMPTS = {
    "excited": (
        # Positive examples (excited state)
        [
            "I feel so energized and enthusiastic right now!",
            "This is amazing! I can't wait to explore this further!",
            "I'm thrilled about this new discovery!",
            "Everything feels possible and exciting!",
            "I have so much energy and motivation!",
            "Wow! This is incredible!",
            "I'm bursting with ideas and enthusiasm!",
            "This breakthrough is so exciting!",
            "I feel alive and ready for anything!",
            "What an exhilarating challenge!",
        ],
        # Negative examples (opposite of excited)
        [
            "I feel tired and drained.",
            "This is boring and uninteresting.",
            "I don't really care about this.",
            "Everything feels like a chore.",
            "I have no energy or motivation.",
            "This is so dull and monotonous.",
            "I can barely keep my eyes open.",
            "Nothing seems worth the effort.",
            "I feel completely apathetic.",
            "This is tedious and exhausting.",
        ],
    ),
    "frustrated": (
        # Positive examples (frustrated state)
        [
            "This is so frustrating! Nothing is working!",
            "I keep hitting walls and can't make progress.",
            "Why won't this work? I've tried everything!",
            "I'm stuck and getting increasingly annoyed.",
            "This error keeps happening no matter what I do!",
            "I can't believe this is still broken!",
            "Nothing I try makes any difference!",
            "This is maddening!",
            "Every solution leads to another problem!",
            "I'm at my wit's end with this!",
        ],
        # Negative examples (opposite of frustrated)
        [
            "Everything is going smoothly.",
            "I'm making great progress on this.",
            "This is working exactly as expected.",
            "I feel calm and in control.",
            "Problems are resolving themselves nicely.",
            "The solution came to me easily.",
            "Everything is falling into place.",
            "I'm pleased with how this is going.",
            "This is proceeding perfectly.",
            "I'm satisfied with the results.",
        ],
    ),
    "calm": (
        # Positive examples (calm state)
        [
            "I feel peaceful and centered.",
            "Everything is fine, there's no rush.",
            "I'm relaxed and taking my time.",
            "Things are proceeding at a comfortable pace.",
            "I feel serene and unhurried.",
            "There's a gentle ease to this moment.",
            "I'm content with where things are.",
            "I feel balanced and at peace.",
            "There's no need to worry.",
            "Everything is unfolding naturally.",
        ],
        # Negative examples (opposite of calm)
        [
            "I need to hurry! There's no time!",
            "Everything is urgent and stressful!",
            "I'm anxious and on edge.",
            "So much pressure! I can't relax!",
            "My mind is racing with worries.",
            "I'm overwhelmed and tense.",
            "The deadline is looming!",
            "I feel frantic and rushed.",
            "There's too much to handle!",
            "I'm stressed and agitated.",
        ],
    ),
    "depleted": (
        # Positive examples (depleted state)
        [
            "I feel exhausted and empty.",
            "I don't have the energy for this.",
            "Everything feels heavy and difficult.",
            "I'm running on fumes here.",
            "I need a break, I'm worn out.",
            "I feel completely drained.",
            "I have nothing left to give.",
            "I'm too tired to even think.",
            "Everything requires too much effort.",
            "I feel hollow and spent.",
        ],
        # Negative examples (opposite of depleted)
        [
            "I feel refreshed and ready!",
            "I have plenty of energy for this.",
            "Everything feels light and easy.",
            "I'm full of vitality!",
            "I'm well-rested and capable.",
            "I feel renewed and energized.",
            "I'm strong and prepared.",
            "I have abundant energy.",
            "I feel vigorous and alive.",
            "I'm in great spirits!",
        ],
    ),
}


def main():
    parser = argparse.ArgumentParser(
        description="Train emotional steering vectors for Elpis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--output",
        default="./data/emotion_vectors",
        help="Output directory for trained vectors",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=15,
        help="Layer to train steering vectors for (try 12-20 for Llama 3.1 8B)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for training",
    )
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"\n{'='*60}")
    print("Elpis Emotion Vector Training")
    print(f"{'='*60}\n")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Layer: {args.layer}")
    print(f"Output: {args.output}")
    print(f"\n{'='*60}\n")

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    print("(This may take a few minutes on first run)\n")

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map=device if device != "cpu" else None,
            trust_remote_code=True,
        )

        if device == "cpu":
            model = model.to(device)

        model.eval()
        print("✓ Model loaded successfully\n")

    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Train vectors for each emotion
    print("Training emotion vectors...\n")

    for emotion_name, (positive, negative) in EMOTION_PROMPTS.items():
        print(f"Training '{emotion_name}' vector...")
        print(f"  - {len(positive)} positive examples")
        print(f"  - {len(negative)} negative examples")

        try:
            vector = train_steering_vector(
                model=model,
                tokenizer=tokenizer,
                positive_examples=positive,
                negative_examples=negative,
                layers=[args.layer],
            )

            # Save vector
            vector_path = output_path / f"{emotion_name}.pt"
            torch.save(vector, vector_path)

            print(f"  ✓ Saved to {vector_path}\n")

        except Exception as e:
            print(f"  ✗ Error training {emotion_name}: {e}\n")
            continue

    print(f"{'='*60}")
    print("Training complete!")
    print(f"{'='*60}\n")
    print(f"Emotion vectors saved to: {output_path}")
    print("\nNext steps:")
    print("1. Update your Elpis config to use these vectors:")
    print(f"   emotion_vectors_path: {output_path}")
    print("\n2. Restart Elpis to load the new vectors")
    print("\n3. Experiment with different layers (12-20) for best results")
    print("   Run this script again with --layer <N> to try different layers")


if __name__ == "__main__":
    main()
