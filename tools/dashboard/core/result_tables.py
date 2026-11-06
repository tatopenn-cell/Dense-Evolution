"""Tables of the result of a run, for the dashboard."""

from typing import Dict, List

__all__ = ["outcome_rows"]


def outcome_rows(counts: Dict[str, int]) -> List[dict]:
    """Rows of the outcomes of a run, most frequent first.

    Args:
        counts: shot count of each outcome, as a bitstring.

    Returns:
        A list of ``{"outcome": bitstring, "shots": n, "probability": n / total}``
        sorted by ``shots`` descending and, for equal ``shots``, by ``outcome``.
        An empty list if ``counts`` is empty or has no shot.

    Examples:
        >>> outcome_rows({"11": 497, "00": 503})[0]
        {'outcome': '00', 'shots': 503, 'probability': 0.503}
    """
    total = sum(counts.values())
    if total <= 0:
        return []
    rows = [
        {"outcome": key, "shots": int(n), "probability": n / total}
        for key, n in counts.items()
    ]
    rows.sort(key=lambda row: (-row["shots"], row["outcome"]))
    return rows
