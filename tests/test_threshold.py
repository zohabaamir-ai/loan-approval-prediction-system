import json

import numpy as np

from loan_approval.candidates import FALLBACK_NAME
from loan_approval.selection import choose_final_name
from loan_approval.threshold import best_threshold, metrics_at, threshold_table


def test_metrics_at_hand_checked_example():
    y = [1, 1, 0, 0]
    p = [0.9, 0.4, 0.6, 0.1]
    m = metrics_at(y, p, 0.5)  # predicts approve for rows 0 and 2
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5


def test_higher_false_approval_cost_gives_higher_threshold():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = np.clip(0.5 * y + rng.normal(0.25, 0.25, 2000), 0, 1)
    equal = best_threshold(y, p, 1, 1)["threshold"]
    strict = best_threshold(y, p, 5, 1)["threshold"]
    assert strict > equal


def test_threshold_table_has_one_row_per_threshold():
    table = threshold_table([0, 1, 1], [0.2, 0.7, 0.9], grid=[0.3, 0.5, 0.8])
    assert len(table) == 3
    assert table["recall"].tolist() == [1.0, 1.0, 0.5]


def test_choose_final_name(tmp_path):
    win = tmp_path / "win.json"
    win.write_text(json.dumps({"decisions": {"challenger_a": {"wins": True}}}))
    lose = tmp_path / "lose.json"
    lose.write_text(json.dumps({"decisions": {"challenger_a": {"wins": False}}}))
    assert choose_final_name(win) == "challenger_a"
    assert choose_final_name(lose) == FALLBACK_NAME
