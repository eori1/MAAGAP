"use client";

import { useState } from "react";
import styles from "./SubmitReportModal.module.css";

interface Props {
  reportId: string;
  projectName: string;
  onClose: () => void;
  onReviewed: () => void;
}

// Manager/Admin-only: request revision on a submitted report. Approve is a
// single click handled inline on the Reports page (no comment needed), so
// this modal only covers the "needs revision" path, which requires one.
export default function ReportReviewModal({ reportId, projectName, onClose, onReviewed }: Props) {
  const [comment, setComment] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);

    if (!comment.trim()) {
      setError("A comment is required so the inspector knows what to fix.");
      return;
    }

    setSubmitting(true);
    try {
      const res = await fetch(`/api/reports/${reportId}/review`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "request_revision", comment }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error ?? "Failed to request revision");
      }
      onReviewed();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to request revision");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
        <h2 className={styles.title}>Request Revision</h2>
        <p className={styles.subtitle}>{projectName}</p>

        {error && <div className={styles.error}>{error}</div>}

        <form onSubmit={handleSubmit}>
          <div className={styles.field}>
            <label className={styles.label}>What needs to be fixed?</label>
            <textarea
              className={styles.textarea}
              value={comment}
              onChange={(e) => setComment(e.target.value)}
              placeholder="e.g. missing financial accomplishment %, photos don't match the described issue..."
              autoFocus
            />
          </div>

          <div className={styles.actions}>
            <button type="button" className={styles.cancelBtn} onClick={onClose}>Cancel</button>
            <button type="submit" className={styles.submitBtn} disabled={submitting}>
              {submitting ? "Sending..." : "Send"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
