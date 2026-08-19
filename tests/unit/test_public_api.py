"""
Unit tests for dense_evolution/__init__.py's __all__ -- the explicit
public API surface added in Phase 3 of the architectural refactor (see
prog.txt). Nothing here tests the individual functions themselves (each
has its own dedicated test file); this only guards the __all__ list
itself against typos/staleness as the package evolves.
"""
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
