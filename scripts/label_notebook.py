import json

def add_label(cell, label_text):
    label_block = (
        "# " + "="*60 + "\n" +
        f"# {label_text}\n" +
        "# " + "="*60 + "\n\n"
    )
    if not cell['source'][0].startswith("# ==="):
        cell['source'].insert(0, label_block)

with open('MAAGAP_Objective1.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Hardcoded indices based on notebook structure mapping
labels = {
    3: "STEP 1: LOAD & CLEAN REAL DATA (PPDO & FUND TRANSFER)",
    7: "STEP 2: GENERATE MULTI-YEAR SYNTHETIC DATASET",
    11: "STEP 3: FEATURE ENGINEERING (STATIC & TEMPORAL)",
    13: "STEP 4: DATA SPLIT (70% TRAIN / 15% VAL / 15% TEST)",
    17: "STEP 5A: TRAIN RANDOM FOREST",
    19: "STEP 5B: TRAIN XGBOOST",
    21: "STEP 5C: TRAIN RISK CATEGORISATION MODELS",
    23: "STEP 5D: TRAIN LSTM (TEMPORAL SEQUENCE MODEL)",
    27: "STEP 5E: TRAIN META-ENSEMBLE (LOGISTIC STACKING)",
    29: "OBJECTIVE 3: DYNAMIC RISK SCORING ENGINE",
    32: "OBJECTIVE 4: LP RESOURCE ALLOCATION OPTIMIZATION",
    34: "OBJECTIVE 4: MONTE CARLO ROBUSTNESS SIMULATION",
    40: "STEP 6: FULL PIPELINE EVALUATION (METRICS)"
}

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and i in labels:
        add_label(cell, labels[i])

with open('MAAGAP_Objective1.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook code cells successfully labeled.")
