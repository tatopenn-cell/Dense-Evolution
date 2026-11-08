"""docs/assets/composer/browser_memory.js, run with Node.

The qubit limit of the page comes from the memory of the browser and from the three heap sizes that
were measured in Chromium on Kaggle. The expected values are computed here from the same three
numbers, written again, and from the rules of the engine: floor 16, ceiling 24, 15% kept free.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

MEMORY = Path(__file__).resolve().parents[2] / "docs" / "assets" / "composer" / "browser_memory.js"
MEASURED = {20: 214.8125, 22: 534.625, 24: 1545.9375}

_SCRIPT = (
    "global.window = {};"
    "eval(require('fs').readFileSync(process.argv[1], 'utf8'));"
    "const a = JSON.parse(process.argv[2]);"
    "const M = window.DEBrowserMemory;"
    "process.stdout.write(JSON.stringify(a.map((g) => g === null ? M.browserQubitLimit(undefined) : M.browserQubitLimit(g))"
    ".concat([[20, 22, 24].map((n) => M.predictedHeapMb(n))])));"
)


def _node():
    exe = shutil.which("node")
    if exe:
        return exe
    try:
        from playwright._impl._driver import compute_driver_executable

        return str(compute_driver_executable()[0])
    except (ImportError, AttributeError):
        return None


pytestmark = pytest.mark.skipif(_node() is None, reason="Node is not available")

GB = [None, 0.25, 0.5, 1, 2, 4, 8, 32]


@pytest.fixture(scope="module")
def answers():
    done = subprocess.run(
        [_node(), "-e", _SCRIPT, str(MEMORY), json.dumps(GB)], capture_output=True, text=True, check=True
    )
    out = json.loads(done.stdout)
    return dict(zip(map(str, GB), out[:-1])), out[-1]


def _expected(gb):
    if gb is None:
        return 16
    slope = (MEASURED[24] - MEASURED[22]) / (2**24 - 2**22)
    base = MEASURED[22] - slope * 2**22
    usable = min(gb * 1024 * 0.85, 4096)
    n = 16
    while n < 24 and base + slope * 2 ** (n + 1) <= usable:
        n += 1
    return n


@pytest.mark.parametrize("gb", GB)
def test_the_limit_is_the_one_computed_from_the_measured_heap(gb, answers):
    assert answers[0][str(gb)]["limit"] == _expected(gb)


def test_the_limit_never_falls_under_16_nor_goes_over_24_and_grows_with_the_memory(answers):
    limits = [answers[0][str(g)]["limit"] for g in GB]
    assert min(limits) == 16 and max(limits) == 24
    known = [answers[0][str(g)]["limit"] for g in GB if g is not None]
    assert known == sorted(known)


def test_the_model_is_exact_at_the_two_points_it_is_drawn_through_and_not_under_the_third(answers):
    at20, at22, at24 = answers[1]
    assert at22 == pytest.approx(MEASURED[22], abs=1e-6)
    assert at24 == pytest.approx(MEASURED[24], abs=1e-6)
    assert at20 >= MEASURED[20]


def test_a_browser_that_does_not_tell_its_memory_gets_the_floor_and_says_why(answers):
    assert answers[0]["None"]["usableMb"] is None
    assert "non rilevabile" in answers[0]["None"]["basis"]
