"use client";

import { useState } from "react";
import { PROJECT_TYPES, IMPLEMENTING_AGENCIES, FUNDING_SOURCES, MUNICIPALITIES } from "@/lib/ppaOptions";
import styles from "./AddPpaModal.module.css";

interface Props {
  onClose: () => void;
  onCreated: () => void;
}

// Manager/Admin only. A manually-added PPA cannot get an immediate risk
// score -- the ML pipeline is a batch process, not on-demand -- so it's
// created as "Pending Assessment" until the next full pipeline run scores it.
export default function AddPpaModal({ onClose, onCreated }: Props) {
  const [projectName, setProjectName] = useState("");
  const [description, setDescription] = useState("");
  const [projectType, setProjectType] = useState<string>(PROJECT_TYPES[0].value);
  const [category, setCategory] = useState(IMPLEMENTING_AGENCIES[0]);
  const [location, setLocation] = useState(MUNICIPALITIES[0]);
  const [budget, setBudget] = useState("");
  const [startDate, setStartDate] = useState("");
  const [fundingSource, setFundingSource] = useState(FUNDING_SOURCES[0]);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);

    const budgetNum = Number(budget);
    if (!projectName.trim()) { setError("Project name is required."); return; }
    if (!startDate) { setError("Start date is required."); return; }
    if (!budgetNum || budgetNum <= 0) { setError("Budget must be greater than 0."); return; }

    setSubmitting(true);
    try {
      const res = await fetch("/api/projects", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          projectName,
          description,
          projectType,
          category,
          location,
          budgetAllocated: budgetNum,
          startDate,
          fundingSource,
        }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error ?? "Failed to add PPA");
      }
      onCreated();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to add PPA");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
        <h2 className={styles.title}>Add New PPA</h2>
        <p className={styles.subtitle}>New projects show as &quot;Pending Assessment&quot; until the next pipeline run scores them.</p>

        {error && <div className={styles.error}>{error}</div>}

        <form onSubmit={handleSubmit}>
          <div className={styles.field}>
            <label className={styles.label}>Project Name</label>
            <input
              className={styles.input}
              value={projectName}
              onChange={(e) => setProjectName(e.target.value)}
              placeholder="e.g. Farm-to-Market Road Improvement, Brgy. Rizal"
            />
          </div>

          <div className={styles.field}>
            <label className={styles.label}>Description (optional)</label>
            <textarea
              className={styles.input}
              style={{ resize: "vertical", minHeight: 60 }}
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="What is this project's scope and purpose?"
            />
          </div>

          <div className={styles.row}>
            <div className={styles.field}>
              <label className={styles.label}>Project Type</label>
              <select className={styles.select} value={projectType} onChange={(e) => setProjectType(e.target.value)}>
                {PROJECT_TYPES.map((t) => <option key={t.value} value={t.value}>{t.label} ({t.durationMonths}mo)</option>)}
              </select>
            </div>
            <div className={styles.field}>
              <label className={styles.label}>Municipality</label>
              <select className={styles.select} value={location} onChange={(e) => setLocation(e.target.value)}>
                {MUNICIPALITIES.map((m) => <option key={m} value={m}>{m}</option>)}
              </select>
            </div>
          </div>

          <div className={styles.field}>
            <label className={styles.label}>Implementing Agency</label>
            <select className={styles.select} value={category} onChange={(e) => setCategory(e.target.value)}>
              {IMPLEMENTING_AGENCIES.map((a) => <option key={a} value={a}>{a}</option>)}
            </select>
          </div>

          <div className={styles.row}>
            <div className={styles.field}>
              <label className={styles.label}>Approved Budget (₱)</label>
              <input
                type="number" min={0} step="1000"
                className={styles.input}
                value={budget}
                onChange={(e) => setBudget(e.target.value)}
                placeholder="e.g. 5000000"
              />
            </div>
            <div className={styles.field}>
              <label className={styles.label}>Start Date</label>
              <input
                type="date"
                className={styles.input}
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
              />
            </div>
          </div>

          <div className={styles.field}>
            <label className={styles.label}>Funding Source</label>
            <select className={styles.select} value={fundingSource} onChange={(e) => setFundingSource(e.target.value)}>
              {FUNDING_SOURCES.map((f) => <option key={f} value={f}>{f}</option>)}
            </select>
          </div>

          <div className={styles.actions}>
            <button type="button" className={styles.cancelBtn} onClick={onClose}>Cancel</button>
            <button type="submit" className={styles.submitBtn} disabled={submitting}>
              {submitting ? "Adding..." : "Add PPA"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
