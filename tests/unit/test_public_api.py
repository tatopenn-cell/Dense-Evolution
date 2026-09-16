"""
Unit tests for dense_evolution/__init__.py's __all__ -- the explicit
public API surface added in Phase 3 of the architectural refactor (see
prog.txt). Nothing here tests the individual functions themselves (each
has its own dedicated test file); this only guards the __all__ list
itself against typos/staleness as the package evolves.
"""
import pathlib
import re

import dense_evolution as de


def test_all_names_resolve_and_are_not_none():
    assert len(de.__all__) > 0
    for name in de.__all__:
        assert hasattr(de, name), f"__all__ lists {name!r}, but dense_evolution has no such attribute"
        assert getattr(de, name) is not None, f"dense_evolution.{name} is None"


def test_all_has_no_duplicates():
    assert len(de.__all__) == len(set(de.__all__))


def test_star_import_only_binds_all_names():
    # `from dense_evolution import *` in a fresh namespace must bind
    # exactly __all__, not every module-level name (submodules like
    # `chunk`, `backends`, `circuits`, ... would otherwise leak in too).
    ns = {}
    exec("from dense_evolution import *", ns)
    bound = {k for k in ns if k != "__builtins__"}
    assert bound == set(de.__all__)


def _extract_pyproject_version(pyproject_text):
    """Split out of test_version_matches_pyproject (prog.txt test-suite
    audit, issue #269 point 5) so the regex itself is directly testable
    against synthetic input, not just the repo's own current
    pyproject.toml. Not switched to tomllib -- stdlib only from Python
    3.11, but CI's own matrix still tests 3.10 -- widened the regex to
    accept single quotes too, and added a specific check for PEP 621's
    `dynamic = ["version"]` instead of the previous generic "couldn't
    find a version line" message either way."""
    match = re.search(r'''^version\s*=\s*['"]([^'"]+)['"]''', pyproject_text, re.MULTILINE)
    if match is None:
        if re.search(r'^dynamic\s*=.*version', pyproject_text, re.MULTILINE):
            raise AssertionError(
                "pyproject.toml declares version as dynamic (PEP 621) -- this "
                "regex only handles a literal `version = \"...\"` line and needs "
                "updating for whatever mechanism (setuptools_scm, etc.) now "
                "supplies the version."
            )
        raise AssertionError("couldn't find a version line in pyproject.toml")
    return match.group(1)


def test_version_matches_pyproject():
    """The 8.1.74 wheel published to PyPI had this exact drift: pyproject.toml
    was bumped, __init__.py's hardcoded __version__ wasn't, in the window
    between those two commits -- caught only by downloading the published
    wheel and checking it by hand. This regression test catches the same
    drift at test time, before anything ships."""
    pyproject_text = (pathlib.Path(__file__).resolve().parents[2] / "pyproject.toml").read_text()
    version = _extract_pyproject_version(pyproject_text)
    assert de.__version__ == version, (
        f"dense_evolution.__version__ ({de.__version__!r}) doesn't match "
        f"pyproject.toml's version ({version!r})"
    )


def test_extract_pyproject_version_accepts_single_quotes():
    assert _extract_pyproject_version("[project]\nversion = '1.2.3'\n") == "1.2.3"


def test_extract_pyproject_version_accepts_double_quotes():
    assert _extract_pyproject_version('[project]\nversion = "1.2.3"\n') == "1.2.3"


def test_extract_pyproject_version_gives_specific_error_for_dynamic_version():
    import pytest
    with pytest.raises(AssertionError, match="dynamic"):
        _extract_pyproject_version('[project]\ndynamic = ["version"]\n')


def test_extract_pyproject_version_gives_generic_error_when_truly_absent():
    import pytest
    with pytest.raises(AssertionError, match="couldn't find a version line"):
        _extract_pyproject_version('[project]\nname = "foo"\n')
