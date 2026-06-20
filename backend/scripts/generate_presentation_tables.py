"""
PNG tables of BASELINE (UNTUNED) training parameters only — as used in code when tune=False.
Source: maagap/config.py + maagap/models.py

Run: python generate_presentation_tables.py
Outputs: outputs/presentation_tables/untuned_*.png
"""

import os

import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "presentation_tables")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "DejaVu Sans", "Helvetica"]
TITLE_FS = 14
CELL_FS = 10
HEADER_FS = 10


def _save_table(title: str, col_labels: list, rows: list, filename: str, figsize=(11, 3.8), col_widths=None):
    nrows = len(rows)
    fig_h = max(2.0, 0.40 * (nrows + 2) + 1.0)
    fig, ax = plt.subplots(figsize=(figsize[0], fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=TITLE_FS, fontweight="bold", pad=16, loc="left")

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="upper center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(CELL_FS)
    table.scale(1, 1.85)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor("#1a5276")
            cell.set_text_props(color="white", fontweight="bold", fontsize=HEADER_FS)
            cell.set_height(0.12)
        else:
            cell.set_facecolor("#f8f9fa" if row % 2 == 1 else "#ffffff")
            cell.set_text_props(fontsize=CELL_FS)

    if col_widths:
        for j, w in enumerate(col_widths):
            for i in range(nrows + 1):
                table[(i, j)].set_width(w)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print("Saved:", path)


def main():
    # --- Global (applies to all runs) ---
    _save_table(
        "MAAGAP — Dataset split & reproducibility",
        ["Setting", "Value"],
        [
            ["Training set", "70%"],
            ["Validation set", "15%"],
            ["Test set", "15%"],
            ["Random seed (NumPy / sklearn / TF where applicable)", "42"],
        ],
        "untuned_00_data_split.png",
        figsize=(10, 2.8),
        col_widths=[0.55, 0.45],
    )

    # RF: train_random_forest(..., tune=False) → RandomForestClassifier in models.py
    _save_table(
        "Random Forest — untuned (baseline) parameters",
        ["Parameter", "Value"],
        [
            ["n_estimators", "300"],
            ["max_depth", "15"],
            ["min_samples_split", "2 (sklearn default)"],
            ["min_samples_leaf", "1 (sklearn default)"],
            ["max_features", "sqrt (sklearn default)"],
            ["class_weight", "balanced"],
            ["random_state", "42"],
            ["n_jobs", "-1 (all cores)"],
        ],
        "untuned_01_random_forest.png",
        figsize=(10, 4.2),
        col_widths=[0.42, 0.58],
    )

    # XGB: train_xgboost(..., tune=False)
    _save_table(
        "XGBoost — untuned (baseline) parameters",
        ["Parameter", "Value"],
        [
            ["n_estimators", "300"],
            ["max_depth", "10"],
            ["learning_rate", "0.08"],
            ["objective (binary delay)", "binary:logistic"],
            ["eval_metric", "logloss"],
            ["scale_pos_weight", "neg_count / pos_count (from training labels)"],
            ["use_label_encoder", "False"],
            ["random_state", "42"],
            ["n_jobs", "-1 (all cores)"],
        ],
        "untuned_02_xgboost.png",
        figsize=(10, 4.5),
        col_widths=[0.42, 0.58],
    )

    # LSTM: train_lstm(..., tune=False) — default_params + _build_lstm + fit
    _save_table(
        "LSTM — untuned (baseline) architecture & training",
        ["Parameter", "Value"],
        [
            ["Input timesteps (quarters)", "4"],
            ["LSTM units (layer 1 → layer 2)", "64 → 32"],
            ["Dropout (after each LSTM)", "0.35"],
            ["Dense hidden units (ReLU)", "32"],
            ["Output activation (binary)", "sigmoid"],
            ["Optimizer", "Adam"],
            ["Learning rate", "0.001"],
            ["Batch size", "32"],
            ["Loss", "binary_crossentropy"],
            ["Max epochs", "60"],
            ["Early stopping", "monitor val_loss, patience 8, restore_best_weights=True"],
            ["Class weights", "balanced (sklearn compute_class_weight)"],
        ],
        "untuned_03_lstm.png",
        figsize=(11, 6.0),
        col_widths=[0.40, 0.60],
    )

    # Meta: train_meta_ensemble — same whether bases are tuned or not; meta-learner itself is "untuned" logistic
    _save_table(
        "Meta-ensemble — stacking classifier (training)",
        ["Parameter", "Value"],
        [
            ["Algorithm", "Logistic regression (sklearn)"],
            ["Input features", "3 probabilities: RF, XGBoost, LSTM"],
            ["max_iter", "500"],
            ["random_state", "42"],
            ["Trained on", "Validation-set stacked probabilities vs. binary delay label"],
        ],
        "untuned_04_meta_ensemble.png",
        figsize=(10, 3.4),
        col_widths=[0.38, 0.62],
    )

    print("\nAll UNTUNED-only tables written to:", OUT_DIR)
    print("Files: untuned_00_data_split.png … untuned_04_meta_ensemble.png")


if __name__ == "__main__":
    main()
