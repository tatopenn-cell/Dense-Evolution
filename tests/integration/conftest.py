"""Fixtures shared by tests/integration/ only.

dashboard_core used to enable float64 (jax_enable_x64) as a bare
module-level side effect in its own __init__.py, so any test that merely
imported a dashboard_core submodule got it for free. That side effect
moved into an explicit dashboard_core.enable_dashboard_precision(),
called by the real entry point (tools/dashboard/app.py) -- see
dashboard_core/__init__.py's own docstring for why (prog.txt,
dashboard_core audit point 2). This fixture is the test-suite's
equivalent entry point: every test under tests/integration/ (where all
of dashboard_core's own tests live) still gets float64, just via an
explicit call instead of an import-time side effect.
"""
import pytest


@pytest.fixture(autouse=True, scope="session")
def _dashboard_precision():
    import dashboard_core as dc
    dc.enable_dashboard_precision()
