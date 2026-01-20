"""Unit tests for PsycheHTTPServer."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from psyche.server.http import PsycheHTTPServer, HTTPServerConfig


class TestEmotionEndpoint:
    """Tests for the /v1/psyche/emotion endpoint."""

    @pytest.fixture
    def mock_core(self):
        """Create a mock PsycheCore."""
        core = MagicMock()
        core.get_emotion = AsyncMock(return_value={
            "valence": 0.5,
            "arousal": 0.3,
            "quadrant": "content",
        })
        return core

    @pytest.fixture
    def server(self, mock_core):
        """Create HTTP server with mock core."""
        return PsycheHTTPServer(core=mock_core)

    @pytest.fixture
    def client(self, server):
        """Create test client."""
        return TestClient(server.app)

    def test_get_emotion_returns_current_state(self, client, mock_core):
        """GET /v1/psyche/emotion should return current emotional state."""
        response = client.get("/v1/psyche/emotion")

        assert response.status_code == 200
        data = response.json()
        assert data["valence"] == 0.5
        assert data["arousal"] == 0.3
        assert data["quadrant"] == "content"

    def test_get_emotion_calls_core(self, client, mock_core):
        """GET /v1/psyche/emotion should call core.get_emotion()."""
        client.get("/v1/psyche/emotion")

        mock_core.get_emotion.assert_called_once()

    def test_get_emotion_with_neutral_state(self, mock_core):
        """GET /v1/psyche/emotion should handle neutral state."""
        mock_core.get_emotion = AsyncMock(return_value={
            "valence": 0.0,
            "arousal": 0.0,
            "quadrant": "neutral",
        })
        server = PsycheHTTPServer(core=mock_core)
        client = TestClient(server.app)

        response = client.get("/v1/psyche/emotion")

        assert response.status_code == 200
        data = response.json()
        assert data["valence"] == 0.0
        assert data["arousal"] == 0.0
        assert data["quadrant"] == "neutral"

    def test_get_emotion_with_extreme_values(self, mock_core):
        """GET /v1/psyche/emotion should handle extreme emotional values."""
        mock_core.get_emotion = AsyncMock(return_value={
            "valence": 1.0,
            "arousal": -0.8,
            "quadrant": "calm",
        })
        server = PsycheHTTPServer(core=mock_core)
        client = TestClient(server.app)

        response = client.get("/v1/psyche/emotion")

        assert response.status_code == 200
        data = response.json()
        assert data["valence"] == 1.0
        assert data["arousal"] == -0.8
        assert data["quadrant"] == "calm"


class TestEmotionUpdateEndpoint:
    """Tests for the POST /v1/psyche/emotion endpoint."""

    @pytest.fixture
    def mock_core(self):
        """Create a mock PsycheCore."""
        core = MagicMock()
        core.update_emotion = AsyncMock(return_value={
            "valence": 0.7,
            "arousal": 0.5,
            "quadrant": "excited",
        })
        return core

    @pytest.fixture
    def server(self, mock_core):
        """Create HTTP server with mock core."""
        return PsycheHTTPServer(core=mock_core)

    @pytest.fixture
    def client(self, server):
        """Create test client."""
        return TestClient(server.app)

    def test_update_emotion_returns_new_state(self, client, mock_core):
        """POST /v1/psyche/emotion should return updated emotional state."""
        response = client.post(
            "/v1/psyche/emotion",
            json={"event_type": "success", "intensity": 0.8},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["valence"] == 0.7
        assert data["arousal"] == 0.5
        assert data["quadrant"] == "excited"

    def test_update_emotion_calls_core_with_params(self, client, mock_core):
        """POST /v1/psyche/emotion should call core.update_emotion() with correct params."""
        client.post(
            "/v1/psyche/emotion",
            json={"event_type": "frustration", "intensity": 0.5},
        )

        mock_core.update_emotion.assert_called_once_with(
            event_type="frustration",
            intensity=0.5,
        )

    def test_update_emotion_default_intensity(self, client, mock_core):
        """POST /v1/psyche/emotion should use default intensity of 1.0."""
        client.post(
            "/v1/psyche/emotion",
            json={"event_type": "engagement"},
        )

        mock_core.update_emotion.assert_called_once_with(
            event_type="engagement",
            intensity=1.0,
        )

    def test_update_emotion_various_events(self, mock_core):
        """POST /v1/psyche/emotion should handle various event types."""
        server = PsycheHTTPServer(core=mock_core)
        client = TestClient(server.app)

        event_types = ["success", "failure", "frustration", "engagement", "boredom"]
        for event_type in event_types:
            mock_core.update_emotion.reset_mock()
            response = client.post(
                "/v1/psyche/emotion",
                json={"event_type": event_type, "intensity": 0.3},
            )
            assert response.status_code == 200
            mock_core.update_emotion.assert_called_once()


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    @pytest.fixture
    def mock_core(self):
        """Create a mock PsycheCore."""
        return MagicMock()

    @pytest.fixture
    def server(self, mock_core):
        """Create HTTP server with mock core."""
        return PsycheHTTPServer(core=mock_core)

    @pytest.fixture
    def client(self, server):
        """Create test client."""
        return TestClient(server.app)

    def test_health_returns_ok(self, client):
        """GET /health should return status ok."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "connections" in data
        assert "model" in data


class TestModelsEndpoint:
    """Tests for the /v1/models endpoint."""

    @pytest.fixture
    def mock_core(self):
        """Create a mock PsycheCore."""
        return MagicMock()

    @pytest.fixture
    def server(self, mock_core):
        """Create HTTP server with mock core."""
        config = HTTPServerConfig(model_name="test-psyche")
        return PsycheHTTPServer(core=mock_core, config=config)

    @pytest.fixture
    def client(self, server):
        """Create test client."""
        return TestClient(server.app)

    def test_models_lists_psyche_model(self, client):
        """GET /v1/models should list the psyche model."""
        response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "test-psyche"
        assert data["data"][0]["owned_by"] == "psyche"
