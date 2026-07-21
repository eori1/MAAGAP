# Manuscript Alignment

Related: [[00-Overview]] · [[01-Architecture]] · [[06-Current-State-and-Next-Steps]]

Source: `UG-CICT-THESIS-MANUSCRIPT.md` (repo root). Objectives as renumbered after deduping the original 4/5 duplicate (see [[02-Decisions-Log]]) — current manuscript has 5 objectives.

| # | Objective | Status | Where |
|---|---|---|---|
| 1 | Multi-stage predictive framework: RF + XGBoost + LSTM, static + temporal features | ✅ Done | `maagap/models.py`, `maagap/feature_engineering.py` |
| 2 | Model evaluation: Accuracy/Precision/Recall/F1/AUC-ROC + MAE | ✅ Done | `maagap/evaluation.py`; MAE added via `RegressionModelTrainer` (delay days, cost overrun pct) |
| 3 | Dynamic risk scoring engine, 4 tiers (Low/Medium/High/Critical), logic-consistency testing | ✅ Done | `maagap/risk_scoring.py` |
| 4 | Inspector deployment optimization (LP), ≥15% improvement over manual baseline | ✅ Done | `maagap/optimization.py::InspectorAssignmentOptimizer`. Achieved ~230-250% improvement over manual round-robin in test runs — far exceeds the 15% target |
| 5 | ISO/IEC 25010 software quality evaluation (Functional Suitability, Usability, Reliability) | ❌ **Not started** | No evidence of this anywhere in the codebase or prior sessions. This is a real, undiscussed gap — likely needs a UAT/survey instrument, not code |

## Manuscript-adjacent features (not numbered objectives, but described in the design chapters)

| Feature | Status | Notes |
|---|---|---|
| Explainable AI (SHAP) | ✅ Done | `maagap/explainability.py`, per-prediction attributions exported |
| ERD relational schema | ✅ Done, upgraded | Originally CSV exports mirroring the ERD; now real Postgres tables in Supabase (`backend/supabase/schema.sql`) — a stronger implementation than the manuscript's own CSV-table description |
| Risk-tier escalation alerts | ✅ Done | `risk_alerts` table, diffed across pipeline runs |
| Automated PDF reports | ✅ Done (lightweight) | Browser print-based export (`data-print-hide`/`data-print-area` convention), not a server-generated PDF library. Functionally satisfies "automated reports" but worth knowing it's `window.print()`, not e.g. Puppeteer/WeasyPrint |
| AuthService / role-based access (Use Case Diagram: Manager, Inspector, System Administrator) | ✅ Done | Supabase Auth, 3 roles, RLS-backed profiles table. See [[01-Architecture#Roles]] |
| PostgreSQL database (Figure 4 ERD) | ✅ Done | Supabase-hosted Postgres, not local — see [[02-Decisions-Log]] for why |
| Next.js dashboard (manuscript originally said Streamlit/Flask) | ✅ Done, manuscript corrected | Manuscript text updated to describe the actual Next.js stack built |

## Known remaining gaps (as of this vault's creation)

1. **ISO/IEC 25010 evaluation (Objective 5)** — entirely unaddressed. Would need a UAT plan/survey with real PPDO stakeholders or faculty, not just code.
2. **Password self-service** — accounts are currently Admin-managed only (create + set initial password). No "forgot password" email flow or self-service "change my password" UI. Discussed as a candidate next task, not yet built.
3. **`/api/projects` scope note** — Dashboard/Projects/Forecast Engine show the ~450-project *monitored cohort*, not the full 3000-row historical registry. This is an intentional, documented choice (see [[02-Decisions-Log]]), not a bug, but worth remembering if someone asks "why don't I see project X" for a project outside the current pipeline run's test split.
