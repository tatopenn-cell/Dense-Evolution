"""Regression tests for test_docs_examples.py::_extract_section_code
(prog.txt test-suite audit, issue #269 point 5) -- the heading-not-found
error message and the section-boundary fix, using synthetic markdown
files rather than the real docs (whose content is covered by
test_docs_examples.py itself)."""
import pytest

from test_docs_examples import _extract_section_code


def test_missing_heading_lists_available_headings(tmp_path, monkeypatch):
    import test_docs_examples
    monkeypatch.setattr(test_docs_examples, 'REPO_ROOT', tmp_path)
    (tmp_path / "doc.md").write_text("## Real Heading\n```python\nx = 1\n```\n", encoding='utf-8')
    with pytest.raises(AssertionError, match=r"Real Heading"):
        _extract_section_code("doc.md", "## Nonexistent Heading")


def test_only_the_named_sections_fence_is_returned_not_a_later_ones(tmp_path, monkeypatch):
    # The bug this guards against: searching the whole remainder of the
    # file (not just up to the next heading) would return the SECOND
    # fence here for "## First", since re.search finds the first match
    # in the unbounded remainder -- which happens to also be the first
    # fence overall, so this specifically needs a non-code block placed
    # between the two fences under "## First" itself to prove the
    # boundary is respected, not just "some fence was found".
    import test_docs_examples
    monkeypatch.setattr(test_docs_examples, 'REPO_ROOT', tmp_path)
    (tmp_path / "doc.md").write_text(
        "## First\n"
        "some text, no fence here yet\n"
        "## Second\n"
        "```python\nwrong_block = True\n```\n",
        encoding='utf-8',
    )
    with pytest.raises(AssertionError, match=r"no python fence found under '## First'"):
        _extract_section_code("doc.md", "## First")


def test_extracts_the_correct_sections_own_fence(tmp_path, monkeypatch):
    import test_docs_examples
    monkeypatch.setattr(test_docs_examples, 'REPO_ROOT', tmp_path)
    (tmp_path / "doc.md").write_text(
        "## First\n```python\ncorrect_block = True\n```\n"
        "## Second\n```python\nwrong_block = True\n```\n",
        encoding='utf-8',
    )
    code = _extract_section_code("doc.md", "## First")
    assert "correct_block" in code
    assert "wrong_block" not in code
