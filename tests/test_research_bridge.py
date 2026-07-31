"""
Unit tests for dashboard_core/research_bridge.py -- the UI-free logic
behind the Research Bridge dashboard page (ui_pages/research_bridge.py).
"""
from unittest.mock import MagicMock, patch

import pytest

from dashboard_core.research_bridge import (
    build_context_block, build_search_query, call_custom_api, new_log_entry,
)


class TestBuildContextBlock:

    def test_includes_hypothesis(self):
        block = build_context_block("does X affect Y?")
        assert "does X affect Y?" in block

    def test_includes_real_data_when_given(self):
        block = build_context_block("hypothesis", real_data="sigma=+4.43, p=0.0033")
        assert "sigma=+4.43, p=0.0033" in block

    def test_omits_real_data_section_when_empty(self):
        block = build_context_block("hypothesis", real_data="")
        assert "DATI REALI" not in block

    def test_includes_notes_when_given(self):
        block = build_context_block("hypothesis", notes="tested on 4 qubits")
        assert "tested on 4 qubits" in block

    def test_never_fabricates_data(self):
        # only what's explicitly passed in should appear -- nothing invented
        block = build_context_block("hypothesis")
        assert "DATI REALI" not in block
        assert "NOTE:" not in block


class TestBuildSearchQuery:

    def test_short_hypothesis_passes_through(self):
        assert build_search_query("does X affect Y?") == "does X affect Y?"

    def test_newlines_collapsed_to_spaces(self):
        assert "\n" not in build_search_query("line one\nline two")

    def test_long_hypothesis_truncated(self):
        long_text = "word " * 100
        query = build_search_query(long_text, max_len=50)
        assert len(query) <= 53  # max_len + "..."
        assert query.endswith("...")


class TestCallCustomApi:

    def test_returns_message_content_on_success(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "yes, this matches known physics"}}]
        }
        mock_response.raise_for_status.return_value = None
        with patch("dashboard_core.research_bridge.requests.post", return_value=mock_response) as mock_post:
            result = call_custom_api("context", "https://api.example.com/v1/chat", "sk-test")
        assert result == "yes, this matches known physics"
        mock_post.assert_called_once()

    def test_sends_bearer_auth_header(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        mock_response.raise_for_status.return_value = None
        with patch("dashboard_core.research_bridge.requests.post", return_value=mock_response) as mock_post:
            call_custom_api("context", "https://api.example.com", "sk-secret")
        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer sk-secret"

    def test_raises_on_http_error(self):
        import requests
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
        with patch("dashboard_core.research_bridge.requests.post", return_value=mock_response):
            with pytest.raises(requests.HTTPError):
                call_custom_api("context", "https://api.example.com", "sk-test")

    def test_default_model_used_when_not_specified(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        mock_response.raise_for_status.return_value = None
        with patch("dashboard_core.research_bridge.requests.post", return_value=mock_response) as mock_post:
            call_custom_api("context", "https://api.example.com", "sk-test")
        assert mock_post.call_args.kwargs["json"]["model"] == "gpt-4o-mini"


class TestNewLogEntry:

    def test_contains_all_fields(self):
        entry = new_log_entry("Google (AI Overview)", "hypothesis text", "context block", "the response")
        assert entry["source"] == "Google (AI Overview)"
        assert entry["hypothesis"] == "hypothesis text"
        assert entry["response"] == "the response"
        assert "timestamp" in entry

    def test_model_label_defaults_empty(self):
        entry = new_log_entry("source", "h", "c")
        assert entry["model"] == ""
