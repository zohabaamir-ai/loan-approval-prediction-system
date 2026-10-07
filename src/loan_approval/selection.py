"""Pick the final model from the cross-validation decisions (no test data involved)."""

from __future__ import annotations

import json
from pathlib import Path

from loan_approval.candidates import FALLBACK_NAME
from loan_approval.data import REPO_ROOT

CANDIDATES_PATH = REPO_ROOT / "reports" / "candidates_cv.json"


def choose_final_name(path: str | Path = CANDIDATES_PATH) -> str:
    """The first challenger that passed the adoption rule, otherwise the simple fallback."""
    payload = json.loads(Path(path).read_text())
    winners = [name for name, d in payload["decisions"].items() if d["wins"]]
    return winners[0] if winners else FALLBACK_NAME
