"""Honest baseline: the original 4 models in a Pipeline, cross-validated on TRAINING data only.

Run with:  python -m loan_approval.run_baseline

The held-out test split is created by split_data() but deliberately never used here.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import pandas as pd
import sklearn

from loan_approval.data import REPO_ROOT, SEED, clean_data, load_data, split_data
from loan_approval.evaluate import cross_validate_model, format_table
from loan_approval.pipeline import MODEL_NAMES, REFERENCE_NAMES, build_pipeline, build_reference

OUTPUT_PATH = REPO_ROOT / "reports" / "baseline_cv.json"


def main() -> None:
    df = clean_data(load_data())
    X_train, _unused_test, y_train, _unused_test_y = split_data(df)  # test is NOT touched
    print(f"Training rows: {len(X_train)}   approved share: {y_train.mean():.3f}")

    results: dict = {}
    for name in REFERENCE_NAMES:
        start = time.time()
        results[name] = cross_validate_model(build_reference(name), X_train, y_train)
        print(f"done {name:<22} {time.time() - start:6.1f}s")
    for name in MODEL_NAMES:
        start = time.time()
        results[name] = cross_validate_model(build_pipeline(name), X_train, y_train)
        print(f"done {name:<22} {time.time() - start:6.1f}s")

    print()
    print(format_table(results))

    payload = {
        "description": "5-fold stratified CV on the training split only. Test split not used.",
        "seed": SEED,
        "n_train": int(len(X_train)),
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "results": results,
    }
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
