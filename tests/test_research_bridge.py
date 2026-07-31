"""
Unit tests for dashboard_core/research_bridge.py -- the UI-free logic
behind the Research Bridge dashboard page (ui_pages/research_bridge.py).
"""
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from dashboard_core.research_bridge import (
    build_context_block, build_search_query, call_custom_api, new_log_entry,
    build_next_search_query, call_local_cli,
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

    def test_auto_detects_anthropic_from_url(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"content": [{"type": "text", "text": "yes, known physics"}]}
        mock_response.raise_for_status.return_value = None
        with patch("dashboard_core.research_bridge.requests.post", return_value=mock_response) as mock_post:
            result = call_custom_api("context", "https://api.anthropic.com/v1/messages", "sk-ant-test")
        assert result == "yes, known physics"
        headers = mock_post.call_args.kwargs["headers"]
        assert headers["x-api-key"] == "sk-ant-test"
        assert "Authorization" not in headers
        assert mock_post.call_args.kwargs["json"]["max_tokens"] == 1024

    def test_anthropic_default_model(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"content": [{"type": "text", "text": "ok"}]}
        mock_response.raise_for_status.return_value = None
        with patch("dashboard_core.research_bridge.requests.post", return_value=mock_response) as mock_post:
            call_custom_api("context", "https://api.anthropic.com/v1/messages", "sk-ant-test")
        assert mock_post.call_args.kwargs["json"]["model"] == "claude-haiku-4-5"

    def test_explicit_provider_overrides_url_detection(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        mock_response.raise_for_status.return_value = None
        with patch("dashboard_core.research_bridge.requests.post", return_value=mock_response) as mock_post:
            call_custom_api("context", "https://my-proxy.example.com/anthropic.com/relay", "sk-test",
                             provider="openai")
        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer sk-test"

    def test_explicit_anthropic_provider_without_url_hint(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"content": [{"type": "text", "text": "ok"}]}
        mock_response.raise_for_status.return_value = None
        with patch("dashboard_core.research_bridge.requests.post", return_value=mock_response) as mock_post:
            call_custom_api("context", "https://my-self-hosted-proxy.example.com", "sk-ant-test",
                             provider="anthropic")
        headers = mock_post.call_args.kwargs["headers"]
        assert headers["x-api-key"] == "sk-ant-test"


class TestCallLocalCli:
    """Uses real subprocess calls (python itself, always on PATH in this
    test environment) rather than mocking subprocess -- the whole point
    of this function is real process/pipe plumbing, which a mock can't
    catch bugs in (this is exactly how the posix=False -> posix=True
    shlex bug was caught during manual testing before these tests
    existed)."""

    def test_pipes_context_via_stdin_and_captures_stdout(self):
        result = call_local_cli(
            "hello from the bridge",
            'python -c "import sys; print(sys.stdin.read().upper())"',
        )
        assert result == "HELLO FROM THE BRIDGE"

    def test_empty_command_raises(self):
        with pytest.raises(ValueError):
            call_local_cli("context", "")
        with pytest.raises(ValueError):
            call_local_cli("context", "   ")

    def test_nonexistent_command_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            call_local_cli("context", "this-command-does-not-exist-anywhere-xyz123")

    def test_nonzero_exit_raises_called_process_error(self):
        with pytest.raises(subprocess.CalledProcessError):
            call_local_cli("context", 'python -c "import sys; sys.exit(1)"')

    def test_timeout_raises(self):
        with pytest.raises(subprocess.TimeoutExpired):
            call_local_cli(
                "context", 'python -c "import time; time.sleep(5)"', timeout=0.5,
            )

    def test_quoted_argument_with_spaces_preserved(self):
        # a follow-up to the posix=False bug: an argument containing
        # spaces, correctly quoted in the command string, must survive
        # as ONE argument, not be split apart.
        result = call_local_cli(
            "ignored", 'python -c "import sys; print(sys.argv[1])" "two words"',
        )
        assert result == "two words"


class TestBuildNextSearchQuery:

    def test_returns_none_without_credentials(self):
        assert build_next_search_query("hypothesis", [], api_url="", api_key="") is None
        assert build_next_search_query("hypothesis", [], api_url="https://x", api_key="") is None
        assert build_next_search_query("hypothesis", [], api_url="", api_key="sk-x") is None

    def test_parses_search_response(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "SEARCH: does amplitude damping favor low Hamming weight"}}]
        }
        mock_response.raise_for_status.return_value = None
        with patch("dashboard_core.research_bridge.requests.post", return_value=mock_response):
            result = build_next_search_query("hyp", [], "https://api.example.com", "sk-test")
        assert result['done'] is False
        assert result['query'] == "does amplitude damping favor low Hamming weight"

    def test_parses_done_response(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "DONE: the hypothesis is already known physics"}}]
        }
        mock_response.raise_for_status.return_value = None
        with patch("dashboard_core.research_bridge.requests.post", return_value=mock_response):
            result = build_next_search_query("hyp", [], "https://api.example.com", "sk-test")
        assert result['done'] is True
        assert result['query'] is None
        assert "already known physics" in result['reasoning']

    def test_unrecognized_format_surfaces_raw_reply(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "I'm not sure what to do here"}}]
        }
        mock_response.raise_for_status.return_value = None
        with patch("dashboard_core.research_bridge.requests.post", return_value=mock_response):
            result = build_next_search_query("hyp", [], "https://api.example.com", "sk-test")
        assert result['done'] is False
        assert result['query'] is None
        assert result['reasoning'] == "I'm not sure what to do here"

    def test_includes_chain_history_in_prompt(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "DONE: ok"}}]}
        mock_response.raise_for_status.return_value = None
        history = [{'query': 'first query', 'result': 'first result text'}]
        with patch("dashboard_core.research_bridge.requests.post", return_value=mock_response) as mock_post:
            build_next_search_query("hyp", history, "https://api.example.com", "sk-test")
        sent_prompt = mock_post.call_args.kwargs["json"]["messages"][0]["content"]
        assert "first query" in sent_prompt
        assert "first result text" in sent_prompt


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
