"""LLM-based and local sentiment analysis for emotional inference.

Provides two analysis modes:
1. Local sentiment model: Fast, lightweight DistilBERT-based analysis
2. LLM self-analysis: Uses the inference engine for deeper emotion detection

Both modes are optional enhancements to the keyword-based analysis in regulation.py.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple
from loguru import logger


@dataclass
class SentimentResult:
    """Result from sentiment analysis."""

    # Core sentiment (-1 to 1, negative to positive)
    sentiment_score: float

    # Confidence in the analysis (0 to 1)
    confidence: float

    # Detected emotions with scores
    emotions: Dict[str, float]

    # Raw analysis source
    source: str  # "local_model", "llm", or "keyword"


class SentimentAnalyzer:
    """
    Analyzes text content for emotional content.

    Supports both local transformer models and LLM-based analysis.
    Falls back gracefully if models aren't available.
    """

    def __init__(
        self,
        use_local_model: bool = True,
        min_length: int = 200,
        llm_analyze_fn: Optional[Callable[[str], str]] = None,
    ):
        """
        Initialize the sentiment analyzer.

        Args:
            use_local_model: Whether to use local DistilBERT model
            min_length: Minimum text length to analyze
            llm_analyze_fn: Optional callback for LLM-based analysis
        """
        self.use_local_model = use_local_model
        self.min_length = min_length
        self.llm_analyze_fn = llm_analyze_fn

        self._local_model = None
        self._local_tokenizer = None
        self._model_loaded = False
        self._model_load_attempted = False

    def _load_local_model(self) -> bool:
        """
        Attempt to load the local sentiment model.

        Returns:
            True if model loaded successfully, False otherwise
        """
        if self._model_load_attempted:
            return self._model_loaded

        self._model_load_attempted = True

        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )

            model_name = "distilbert-base-uncased-finetuned-sst-2-english"
            logger.info(f"Loading local sentiment model: {model_name}")

            self._local_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._local_model = AutoModelForSequenceClassification.from_pretrained(
                model_name
            )
            self._model_loaded = True
            logger.info("Local sentiment model loaded successfully")
            return True

        except ImportError:
            logger.warning(
                "transformers library not available for local sentiment analysis"
            )
            return False
        except Exception as e:
            logger.warning(f"Failed to load local sentiment model: {e}")
            return False

    def analyze(self, content: str) -> Optional[SentimentResult]:
        """
        Analyze content for emotional sentiment.

        Args:
            content: Text content to analyze

        Returns:
            SentimentResult if analysis succeeded, None if skipped
        """
        # Skip short content
        if len(content) < self.min_length:
            return None

        # Try local model first if enabled
        if self.use_local_model:
            result = self._analyze_local(content)
            if result is not None:
                return result

        # Fall back to LLM analysis if available
        if self.llm_analyze_fn is not None:
            result = self._analyze_llm(content)
            if result is not None:
                return result

        return None

    def _analyze_local(self, content: str) -> Optional[SentimentResult]:
        """
        Analyze content using local DistilBERT model.

        Args:
            content: Text content to analyze

        Returns:
            SentimentResult or None if model not available
        """
        if not self._load_local_model():
            return None

        try:
            import torch

            # Tokenize (truncate to max model length)
            inputs = self._local_tokenizer(
                content,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )

            # Run inference
            with torch.no_grad():
                outputs = self._local_model(**inputs)
                logits = outputs.logits

            # Convert to probabilities
            probs = torch.softmax(logits, dim=-1)[0]
            negative_prob = probs[0].item()
            positive_prob = probs[1].item()

            # Convert to sentiment score (-1 to 1)
            sentiment_score = positive_prob - negative_prob

            # Confidence is how far from 0.5 the max prob is
            confidence = abs(max(positive_prob, negative_prob) - 0.5) * 2

            # Map to emotion categories based on sentiment
            emotions = self._map_sentiment_to_emotions(sentiment_score, confidence)

            return SentimentResult(
                sentiment_score=sentiment_score,
                confidence=confidence,
                emotions=emotions,
                source="local_model",
            )

        except Exception as e:
            logger.warning(f"Local sentiment analysis failed: {e}")
            return None

    def _analyze_llm(self, content: str) -> Optional[SentimentResult]:
        """
        Analyze content using LLM self-analysis.

        Args:
            content: Text content to analyze

        Returns:
            SentimentResult or None if analysis failed
        """
        if self.llm_analyze_fn is None:
            return None

        try:
            # Construct prompt for emotion analysis
            prompt = f"""Analyze the emotional tone of the following text.
Rate the overall sentiment from -1 (very negative) to 1 (very positive).
Also identify any strong emotions present.

Text: {content[:500]}...

Respond with ONLY a JSON object like:
{{"sentiment": 0.3, "confidence": 0.8, "emotions": {{"curiosity": 0.6, "frustration": 0.2}}}}
"""

            response = self.llm_analyze_fn(prompt)

            # Parse JSON response
            import json
            # Find JSON in response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
                return SentimentResult(
                    sentiment_score=float(data.get("sentiment", 0)),
                    confidence=float(data.get("confidence", 0.5)),
                    emotions=data.get("emotions", {}),
                    source="llm",
                )

        except Exception as e:
            logger.warning(f"LLM sentiment analysis failed: {e}")

        return None

    def _map_sentiment_to_emotions(
        self, sentiment_score: float, confidence: float
    ) -> Dict[str, float]:
        """
        Map sentiment score to emotion categories.

        Args:
            sentiment_score: Score from -1 to 1
            confidence: Analysis confidence

        Returns:
            Dictionary of emotion names to scores
        """
        emotions: Dict[str, float] = {}

        # Scale emotions by confidence
        scale = confidence

        if sentiment_score > 0.3:
            emotions["satisfaction"] = min(1.0, sentiment_score * scale)
            if sentiment_score > 0.6:
                emotions["excitement"] = min(1.0, (sentiment_score - 0.3) * scale)
        elif sentiment_score < -0.3:
            emotions["frustration"] = min(1.0, abs(sentiment_score) * scale)
            if sentiment_score < -0.6:
                emotions["distress"] = min(1.0, (abs(sentiment_score) - 0.3) * scale)
        else:
            emotions["neutral"] = scale

        return emotions

    def get_emotional_event(
        self, result: SentimentResult
    ) -> Optional[Tuple[str, float]]:
        """
        Convert sentiment result to an emotional event.

        Args:
            result: SentimentResult from analysis

        Returns:
            Tuple of (event_type, intensity) or None if no strong emotion
        """
        # Need minimum confidence to trigger event
        if result.confidence < 0.5:
            return None

        # Map sentiment score to event
        score = result.sentiment_score

        if score > 0.5:
            return ("success", 0.3 + (score - 0.5) * 0.6)
        elif score > 0.2:
            return ("insight", 0.3)
        elif score < -0.5:
            return ("frustration", 0.3 + (abs(score) - 0.5) * 0.6)
        elif score < -0.2:
            return ("error", 0.3)

        return None
