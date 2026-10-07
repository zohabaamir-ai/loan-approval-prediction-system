"""The ONE scoring of the held-out test split, for the final model only.

Run with:  python -m loan_approval.run_final

Everything is frozen before this runs: the model (reports/candidates_cv.json) and the decision
threshold (reports/threshold.json). The script refuses to score the test split a second time,
because a re-run after looking at the result would no longer be an honest estimate.
To rebuild only the figure and tables, use:
    python -m loan_approval.run_final --render-only
"""

from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from loan_approval.candidates import BASELINE_NAME, build_candidate  # noqa: E402
from loan_approval.data import REPO_ROOT, SEED, clean_data, load_data, split_data  # noqa: E402
from loan_approval.fairness import age_band, group_report, max_gap  # noqa: E402
from loan_approval.final import bootstrap_ci, score_at_threshold  # noqa: E402
from loan_approval.selection import choose_final_name  # noqa: E402

REPORTS = REPO_ROOT / "reports"
FINAL_PATH = REPORTS / "final_test.json"
TABLE_PATH = REPORTS / "results_table.md"
FIGURE_PATH = REPORTS / "figures" / "confusion_matrix_test.png"
METRIC_COLUMNS = ("roc_auc", "f1", "precision", "recall", "accuracy")


def score_test_once() -> dict:
    name = choose_final_name()
    threshold = json.loads((REPORTS / "threshold.json").read_text(encoding="utf-8"))["chosen"][
        "threshold"
    ]
    df = clean_data(load_data())
    X_train, X_test, y_train, y_test = split_data(df)

    model = build_candidate(name).fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]

    test = score_at_threshold(y_test, proba, threshold)
    ci = bootstrap_ci(y_test, proba, threshold)
    fairness = {
        "by_gender": group_report(y_test, proba, X_test["person_gender"], threshold),
        "by_age_band": group_report(y_test, proba, age_band(X_test["person_age"]), threshold),
    }
    for section in fairness.values():
        section["_gaps"] = {
            "model_approval_rate_gap": max_gap(section, "model_approval_rate"),
            "recall_gap": max_gap(section, "recall"),
        }
    return {
        "description": "Held-out test split, scored once. Model and threshold were frozen before.",
        "seed": SEED,
        "model": name,
        "threshold": threshold,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test": test,
        "bootstrap_95ci": ci,
        "fairness": fairness,
    }


def plot_confusion(final: dict) -> None:
    t = final["test"]
    counts = [[t["tn"], t["fp"]], [t["fn"], t["tp"]]]
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(counts, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{counts[i][j]:,}", ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1], ["rejected", "approved"])
    ax.set_yticks([0, 1], ["rejected", "approved"])
    ax.set_xlabel("model decision")
    ax.set_ylabel("historical decision")
    ax.set_title(f"Held-out test, threshold {final['threshold']:.2f}")
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=120)


def build_markdown(final: dict) -> str:
    baseline = json.loads((REPORTS / "baseline_cv.json").read_text(encoding="utf-8"))["results"]
    cand = json.loads((REPORTS / "candidates_cv.json").read_text(encoding="utf-8"))["results"]

    def cv_row(label: str, res: dict, data: str) -> str:
        cells = " | ".join(f"{res[m]['mean']:.3f} ± {res[m]['std']:.3f}" for m in METRIC_COLUMNS)
        return f"| {label} | {data} | {cells} |"

    t, ci = final["test"], final["bootstrap_95ci"]
    cells = " | ".join(
        f"**{t[m]:.3f}** [{ci[m]['low']:.3f}, {ci[m]['high']:.3f}]" for m in METRIC_COLUMNS
    )
    cv = "5-fold CV on training data, mean ± std"
    lines = [
        "| Model | Evaluated on | ROC-AUC | F1 | Precision | Recall | Accuracy |",
        "|---|---|---|---|---|---|---|",
        cv_row('Predict "rejected" for everyone (reference)', baseline["predict_rejected"], cv),
        cv_row(
            "Rule: reject if previous default (reference)", baseline["previous_default_rule"], cv
        ),
        cv_row(
            "Random Forest, all features, threshold 0.50 (baseline)", baseline[BASELINE_NAME], cv
        ),
        cv_row("HistGradientBoosting, no gender, threshold 0.50", cand[final["model"]], cv),
        f"| **HistGradientBoosting, no gender, threshold {final['threshold']:.2f} (final)** "
        f"| **held-out test, 95% bootstrap CI** | {cells} |",
        "",
        f"Test rows: {final['n_test']:,}. Training rows: {final['n_train']:,}.",
        "",
    ]
    for title, key in (("gender", "by_gender"), ("age band", "by_age_band")):
        section = final["fairness"][key]
        lines += [
            f"Audit by {title} (held-out test):",
            "",
            f"| {title.capitalize()} | Rows | Approved in data | Approved by model "
            "| Recall | False approval rate |",
            "|---|---|---|---|---|---|",
        ]
        for group, r in section.items():
            if group == "_gaps":
                continue
            lines.append(
                f"| {group} | {r['n']:,} | {r['true_approval_rate']:.1%} "
                f"| {r['model_approval_rate']:.1%} | {r['recall']:.1%} "
                f"| {r['false_approval_rate']:.1%} |"
            )
        gaps = section["_gaps"]
        lines += [
            "",
            f"Largest gap in model approval rate: {gaps['model_approval_rate_gap']:.1%}. "
            f"Largest gap in recall: {gaps['recall_gap']:.1%}.",
            "",
        ]
    return "\n".join(lines)


def render(final: dict) -> None:
    plot_confusion(final)
    TABLE_PATH.write_text(build_markdown(final), encoding="utf-8")


def main() -> None:
    if "--render-only" in sys.argv:
        render(json.loads(FINAL_PATH.read_text(encoding="utf-8")))
        print(f"Rebuilt {TABLE_PATH} and {FIGURE_PATH} from the saved result.")
        return
    if FINAL_PATH.exists():
        raise SystemExit(
            "The test split has already been scored once (reports/final_test.json exists).\n"
            "Scoring it again would make the number a re-roll, not an honest estimate.\n"
            "To rebuild only the figure and tables, run:\n"
            "    python -m loan_approval.run_final --render-only"
        )
    final = score_test_once()
    FINAL_PATH.write_text(json.dumps(final, indent=2), encoding="utf-8")  # saved BEFORE rendering
    t, ci = final["test"], final["bootstrap_95ci"]
    print(f"Final model: {final['model']}   threshold: {final['threshold']:.2f}")
    print(f"Held-out test rows: {final['n_test']}\n")
    for m in METRIC_COLUMNS:
        print(f"  {m:<10} {t[m]:.3f}   95% CI [{ci[m]['low']:.3f}, {ci[m]['high']:.3f}]")
    print(f"\nConfusion counts: tn={t['tn']} fp={t['fp']} fn={t['fn']} tp={t['tp']}")
    for key, title in (("by_gender", "gender"), ("by_age_band", "age band")):
        print(f"\nBy {title}:")
        for group, r in final["fairness"][key].items():
            if group != "_gaps":
                print(
                    f"  {group:<14} n={r['n']:<5} data approved={r['true_approval_rate']:.3f}  "
                    f"model approved={r['model_approval_rate']:.3f}  recall={r['recall']:.3f}"
                )
    render(final)
    print(f"\nSaved {FINAL_PATH}, {TABLE_PATH} and {FIGURE_PATH}")


if __name__ == "__main__":
    main()
