"""Choose the decision threshold on TRAINING data only (out-of-fold predictions).

Run with:  python -m loan_approval.run_threshold

Needs reports/candidates_cv.json first. The held-out test split is never touched.
Assumption (stated in the README): a false approval costs 3 times a false rejection.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from loan_approval.candidates import build_candidate  # noqa: E402
from loan_approval.data import REPO_ROOT, SEED, clean_data, load_data, split_data  # noqa: E402
from loan_approval.selection import choose_final_name  # noqa: E402
from loan_approval.threshold import (  # noqa: E402
    best_threshold,
    metrics_at,
    oof_probabilities,
    threshold_table,
)

OUTPUT_PATH = REPO_ROOT / "reports" / "threshold.json"
FIGURE_PATH = REPO_ROOT / "reports" / "figures" / "threshold_tradeoff.png"
PRIMARY_FP_COST = 3.0  # false approval costs 3x a false rejection (an assumption, see README)
FN_COST = 1.0
SENSITIVITY_FP_COSTS = (1.0, 2.0, 3.0, 5.0, 10.0)


def main() -> None:
    name = choose_final_name()
    print(f"Final model (from the adoption rule): {name}")

    df = clean_data(load_data())
    X_train, _unused_test, y_train, _unused_test_y = split_data(df)  # test is NOT touched
    proba = oof_probabilities(build_candidate(name), X_train, y_train)

    chosen = best_threshold(y_train, proba, PRIMARY_FP_COST, FN_COST)
    default = metrics_at(y_train, proba, 0.5)
    sensitivity = {
        f"{c:g}:1": best_threshold(y_train, proba, c, FN_COST)["threshold"]
        for c in SENSITIVITY_FP_COSTS
    }

    def show(label: str, m: dict) -> None:
        print(
            f"{label:<28} thr={m['threshold']:.2f}  precision={m['precision']:.3f}  "
            f"recall={m['recall']:.3f}  f1={m['f1']:.3f}  false approvals={m['fp']}  "
            f"false rejections={m['fn']}"
        )

    print()
    show("default threshold 0.50", default)
    show(f"chosen (cost {PRIMARY_FP_COST:g}:1)", chosen)
    print(
        "\nThreshold that is cost-optimal for other cost ratios (false approval : false rejection):"
    )
    for ratio, thr in sensitivity.items():
        print(f"  {ratio:>5} -> {thr:.2f}")

    table = threshold_table(y_train, proba)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(table["threshold"], table["precision"], label="precision")
    ax.plot(table["threshold"], table["recall"], label="recall")
    ax.plot(table["threshold"], table["f1"], label="F1", linestyle=":")
    ax.axvline(chosen["threshold"], color="grey", linestyle="--", label="chosen threshold")
    ax.set_xlabel("decision threshold (approve if probability >= threshold)")
    ax.set_ylabel("score (out-of-fold, training data)")
    ax.set_title("Precision / recall trade-off")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=120)

    payload = {
        "description": "Threshold chosen on out-of-fold predictions of the training split only.",
        "seed": SEED,
        "model": name,
        "assumption": f"false approval costs {PRIMARY_FP_COST:g}x a false rejection",
        "chosen": chosen,
        "default_0_5": default,
        "cost_optimal_threshold_by_cost_ratio": sensitivity,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved {OUTPUT_PATH} and {FIGURE_PATH}")


if __name__ == "__main__":
    main()
