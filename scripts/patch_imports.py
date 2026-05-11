"""Patch notebook: add missing sklearn imports back to cell [01]."""
import json

NB_PATH = "MAAGAP_Objective1.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Cell [01] is the imports cell — find it and add missing sklearn import
imports_cell = nb["cells"][1]
src = "".join(imports_cell["source"])

# Check what's already there
missing = []
if "from sklearn.metrics import roc_curve" not in src:
    missing.append("from sklearn.metrics import roc_curve")

if missing:
    # Append missing imports before the first `from maagap` line
    lines = imports_cell["source"]
    insert_at = next(
        (i for i, l in enumerate(lines) if "from maagap" in l),
        len(lines)
    )
    for imp in reversed(missing):
        lines.insert(insert_at, imp + "\n")
    imports_cell["source"] = lines
    print("Added missing imports:", missing)
else:
    print("All imports already present — no change needed.")

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Notebook saved.")
