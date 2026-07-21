# MAAGAP — Project Overview

**MAAGAP** (Machine Analytics for Allocation, Governance and Assessment of Projects) is an undergraduate thesis system for the College of Information and Communications Technology, West Visayas State University. Authors: Jullian A. Bilan, Clarence Anthony G. Bolivar, Kirk Henrich C. Gamo, Jan Floyd J. Vallota. Adviser: John Cristopher Mateo.

## The problem it solves

The Iloilo Provincial Planning and Development Office (PPDO) tracks a backlog of government infrastructure and non-infrastructure Programs, Projects, and Activities (PPAs). Current tracking is reactive (spreadsheets, manual inspection scheduling). MAAGAP adds:

1. **Predictive risk assessment** — ML models forecast which projects will be delayed or run over budget, before it happens.
2. **Optimized resource allocation** — a linear program assigns PPDO's 5-6 field inspectors to the highest-risk projects under real capacity constraints, instead of ad hoc/round-robin visits.

See [[05-Manuscript-Alignment]] for how each thesis objective maps to what's actually built.

## Three-tier architecture

- **Backend** (`backend/`) — Python ML pipeline: data preprocessing, synthetic data generation, ensemble models (RF/XGBoost/LSTM/meta-ensemble), risk scoring, the inspector-assignment LP, SHAP explainability. Orchestrated by `backend/main.py`. See [[01-Architecture#Backend pipeline]].
- **Database** — Supabase (hosted Postgres). Source of truth for everything the frontend displays. See [[01-Architecture#Supabase schema]].
- **Frontend** (`frontend/`) — Next.js 16 App Router dashboard (7 pages) with Supabase Auth and three roles (Manager/Inspector/Admin). See [[01-Architecture#Frontend]].

## Who's using this knowledge base

This vault exists so a new Claude Code session (after `/compact` or a fresh start) can reconstruct full context without re-deriving it from scratch or guessing. Read order for a cold start:

1. This file (00-Overview) — what the project is
2. [[06-Current-State-and-Next-Steps]] — what's done, what's pending, right now
3. [[04-Workflows-and-Gotchas]] — things that will bite you if you don't know them
4. [[01-Architecture]] — how the pieces fit together
5. [[02-Decisions-Log]] and [[03-Progress-Log]] — why things are the way they are, when needed

## Branch and remote state

All work described in this vault happened on branch `fix/inspector-assignment-alignment`, off `main`. Check `git log --oneline -10` and `git status` at the start of a session — this file is not a substitute for that, only for the *why* behind what git shows.
