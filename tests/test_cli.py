"""
Tests for dense_evolution/cli.py -- the `dense-evolution` console script.

_cmd_serve() and _cmd_offline_composer()'s own side effects (starting a
real server, hitting the real internet) are mocked out at the call
boundary so these tests stay fast and offline; the dispatch logic in
main() and _cmd_offline_composer()'s own HTML-parsing/path logic are
exercised for real.
"""
import builtins
import io
import urllib.request
from urllib.error import URLError

import pytest

from dense_evolution import cli


def test_main_with_no_args_prints_usage_to_stdout_and_exits_0(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli.main([])
    assert exc_info.value.code == 0
    out = capsys.readouterr()
    assert "usage: dense-evolution" in out.out
    assert out.err == ""


def test_main_with_unknown_command_prints_usage_to_stderr_and_exits_1(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["not-a-real-command"])
    assert exc_info.value.code == 1
    out = capsys.readouterr()
    assert "usage: dense-evolution" in out.err
    assert out.out == ""


def test_main_serve_dispatches_to_cmd_serve(monkeypatch):
    called = []
    monkeypatch.setattr(cli, "_cmd_serve", lambda: called.append(True))
    cli.main(["serve"])
    assert called == [True]


def test_main_offline_composer_uses_default_dest(monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "_cmd_offline_composer", lambda dest: seen.append(dest))
    cli.main(["offline-composer"])
    assert seen == ["composer-offline"]


def test_main_offline_composer_uses_given_dest(monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "_cmd_offline_composer", lambda dest: seen.append(dest))
    cli.main(["offline-composer", "my-folder"])
    assert seen == ["my-folder"]


def test_cmd_serve_calls_local_site_server_main(monkeypatch):
    monkeypatch.setattr(cli, "_require_composer_extra", lambda: None)
    called = []
    import local_site.app.server as server_module
    monkeypatch.setattr(server_module, "main", lambda: called.append(True))
    cli._cmd_serve()
    assert called == [True]


def test_require_composer_extra_passes_when_deps_are_installed():
    # fastapi/uvicorn/pydantic are installed in this dev environment
    # (the `composer` extra) -- must not raise or exit.
    cli._require_composer_extra()


def test_require_composer_extra_exits_when_a_dep_is_missing(monkeypatch, capsys):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "fastapi":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(SystemExit) as exc_info:
        cli._require_composer_extra()
    assert exc_info.value.code == 1
    assert "pip install dense-evolution[composer]" in capsys.readouterr().err


_FAKE_PAGE_HTML = """<!doctype html><html><head>
<link rel="stylesheet" href="../assets/stylesheets/main.css">
<link rel="icon" href="../assets/favicon.svg">
<link rel="canonical" href="https://tatopenn-cell.github.io/Dense-Evolution/composer/">
<link rel="prev" href="../getting-started/">
<script src="../assets/javascripts/bundle.js"></script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
</head><body>
<img src="../assets/logo.png">
</body></html>"""


class _FakeResponse:
    def __init__(self, data: bytes):
        self._data = data

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_cmd_offline_composer_mirrors_the_page_and_same_origin_assets(tmp_path, monkeypatch):
    requested = []

    def fake_urlopen(url, timeout=30):
        requested.append(url)
        if url == cli.COMPOSER_PAGE_URL:
            return _FakeResponse(_FAKE_PAGE_HTML.encode("utf-8"))
        return _FakeResponse(b"fake-asset-bytes")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    dest = str(tmp_path / "offline")
    page_local_path = cli._cmd_offline_composer(dest)

    # The page itself is saved at the same relative depth it has on the
    # real site (composer/index.html), not at dest/index.html directly.
    assert page_local_path.replace("\\", "/").endswith("offline/composer/index.html")
    with open(page_local_path, encoding="utf-8") as f:
        assert f.read() == _FAKE_PAGE_HTML

    # stylesheet/icon/script/img same-origin assets ARE mirrored.
    assert (tmp_path / "offline" / "assets" / "stylesheets" / "main.css").exists()
    assert (tmp_path / "offline" / "assets" / "favicon.svg").exists()
    assert (tmp_path / "offline" / "assets" / "javascripts" / "bundle.js").exists()
    assert (tmp_path / "offline" / "assets" / "logo.png").exists()

    # canonical/prev <link> tags are not assets and must never be fetched.
    assert "https://tatopenn-cell.github.io/Dense-Evolution/getting-started/" not in requested

    # a genuinely external CDN script is a different origin -- must not be downloaded.
    assert not (tmp_path / "offline" / "tex-mml-chtml.js").exists()
    assert "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" not in requested


def test_cmd_offline_composer_skips_an_asset_that_fails_to_download(tmp_path, monkeypatch, capsys):
    def fake_urlopen(url, timeout=30):
        if url == cli.COMPOSER_PAGE_URL:
            return _FakeResponse(_FAKE_PAGE_HTML.encode("utf-8"))
        raise URLError("boom")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    dest = str(tmp_path / "offline")
    page_local_path = cli._cmd_offline_composer(dest)

    # A failed asset download must not abort the whole mirror -- the page
    # itself is still saved.
    assert io.open(page_local_path, encoding="utf-8").read() == _FAKE_PAGE_HTML
    assert "saltato (non essenziale)" in capsys.readouterr().out
