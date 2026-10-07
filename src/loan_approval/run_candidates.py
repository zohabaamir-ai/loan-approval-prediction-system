"""Cross-validate the stronger candidates on TRAINING data only and apply the adoption rule.

Run with:  python -m loan_approval.run_candidates

Needs reports/baseline_cv.json first (python -m loan_approval.run_baseline).
The held-out test split is never touched.
"""

from __future__ import annotations

import json
import time

from loan_approval.candidates import (
    BASELINE_NAME,
    CHALLENGER_NAMES,
    FALLBACK_NAME,
    beats_baseline,
    build_candidate,
)
from loan_approval.data import REPO_ROOT, SEED, clean_data, load_data, split_data
from loan_approval.evaluate import cross_validate_model, format_table

BASELINE_PATH = REPO_ROOT / "reports" / "baseline_cv.json"
OUTPUT_PATH = REPO_ROOT / "reports" / "candidates_cv.json"


def main() -> None:
    if not BASELINE_PATH.exists():
        raise SystemExit("Run  python -m loan_approval.run_baseline  first.")
    baseline = json.loads(BASELINE_PATH.read_text())["results"][BASELINE_NAME]

    df = clean_data(load_data())
    X_train, _unused_test, y_train, _unused_test_y = split_data(df)  # test is NOT touched

    results: dict = {}
    for name in (FALLBACK_NAME, *CHALLENGER_NAMES):
        start = time.time()
        results[name] = cross_validate_model(build_candidate(name), X_train, y_train)
        print(f"done {name:<34} {time.time() - start:6.1f}s")

    print()
    print(format_table({BASELINE_NAME + " (baseline)": baseline, **results}))

    decisions = {name: beats_baseline(baseline, results[name]) for name in CHALLENGER_NAMES}
    print("\nAdoption rule: challenger mean ROC-AUC must exceed baseline mean + baseline std.")
    for name, d in decisions.items():
        verdict = "ADOPT" if d["wins"] else f"KEEP {FALLBACK_NAME}"
        print(
            f"  {name}: needs > {d['required_above']:.4f}, got {d['candidate_mean']:.4f} "
            f"-> {verdict}"
        )

    payload = {
        "description": "5-fold stratified CV on the training split only. Test split not used.",
        "seed": SEED,
        "rule": "adopt a challenger only if mean ROC-AUC > baseline mean + baseline std",
        "baseline": BASELINE_NAME,
        "results": results,
        "decisions": decisions,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
