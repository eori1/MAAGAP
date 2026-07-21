"use client";

import { useState } from "react";
import styles from "./ReportDetailModal.module.css";

export interface ReportDetail {
  reportId: string;
  projectId: string;
  inspectorName: string;
  date: string;
  actualProgress: number | null;
  financialAccomplishmentPct: number | null;
  slippage: number | null;
  issuesSummary: string;
  notes: string;
  photoUrls: string[];
  reviewStatus: "pending" | "approved" | "needs_revision" | null;
  reviewComment: string | null;
}

const REVIEW_STYLE: Record<string, { bg: string; color: string; label: string }> = {
  pending:        { bg: "#f1f5f9", color: "#7a8fa6", label: "Awaiting Review" },
  approved:       { bg: "#d4efdf", color: "#1e8449", label: "Approved" },
  needs_revision: { bg: "#fde2e2", color: "#c0392b", label: "Needs Revision" },
};

interface Props {
  report: ReportDetail;
  canReview: boolean;
  onClose: () => void;
  onReviewed: () => void;
}

// Full-detail, read-first view of a single field report: the notes, issues,
// and full-size photos a Manager/Admin actually needs before deciding to
// approve it or send it back -- the Reports table row only has room for
// small thumbnails and truncated text, not enough to review from.
export default function ReportDetailModal({ report, canReview, onClose, onReviewed }: Props) {
  const [requestingRevision, setRequestingRevision] = useState(false);
  const [comment, setComment] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleApprove() {
    setError(null);
    setSubmitting(true);
    try {
      const res = await fetch(`/api/reports/${report.reportId}/review`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "approve" }),
      });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error ?? "Failed to approve");
      onReviewed();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to approve");
    } finally {
      setSubmitting(false);
    }
  }

  async function handleSendRevision() {
    setError(null);
    if (!comment.trim()) {
      setError("A comment is required so the inspector knows what to fix.");
      return;
    }
    setSubmitting(true);
    try {
      const res = await fetch(`/api/reports/${report.reportId}/review`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "request_revision", comment }),
      });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error ?? "Failed to request revision");
      onReviewed();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to request revision");
    } finally {
      setSubmitting(false);
    }
  }

  const review = report.reviewStatus ? REVIEW_STYLE[report.reviewStatus] : null;
  const canAct = canReview && report.reviewStatus === "pending";

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
        <div className={styles.header}>
          <div>
            <h2 className={styles.title}>{report.projectId}</h2>
            <p className={styles.subtitle}>{report.inspectorName} · submitted {report.date}</p>
          </div>
          <button className={styles.closeBtn} onClick={onClose} aria-label="Close">×</button>
        </div>

        {review && (
          <span className={styles.reviewBadge} style={{ background: review.bg, color: review.color }}>
            {review.label}
          </span>
        )}

        {error && <div className={styles.error}>{error}</div>}

        <div className={styles.statGrid}>
          <div className={styles.statBox}>
            <div className={styles.statLabel}>Physical</div>
            <div className={styles.statValue}>{report.actualProgress !== null ? `${report.actualProgress.toFixed(0)}%` : "—"}</div>
          </div>
          <div className={styles.statBox}>
            <div className={styles.statLabel}>Financial</div>
            <div className={styles.statValue}>{report.financialAccomplishmentPct !== null ? `${report.financialAccomplishmentPct.toFixed(0)}%` : "—"}</div>
          </div>
          <div className={styles.statBox}>
            <div className={styles.statLabel}>Slippage</div>
            <div className={styles.statValue}>{report.slippage !== null ? `${report.slippage > 0 ? "-" : "+"}${Math.abs(report.slippage).toFixed(1)} pts` : "—"}</div>
          </div>
        </div>

        <div className={styles.section}>
          <div className={styles.sectionLabel}>Issues noted</div>
          <div className={styles.sectionBody}>{report.issuesSummary || "None"}</div>
        </div>

        {report.notes && (
          <div className={styles.section}>
            <div className={styles.sectionLabel}>Additional notes</div>
            <div className={styles.sectionBody}>{report.notes}</div>
          </div>
        )}

        {report.reviewComment && (
          <div className={styles.section}>
            <div className={styles.sectionLabel}>Reviewer comment</div>
            <div className={styles.reviewComment}>{report.reviewComment}</div>
          </div>
        )}

        {report.photoUrls.length > 0 && (
          <div className={styles.section}>
            <div className={styles.sectionLabel}>Site photos</div>
            <div className={styles.photoGrid}>
              {report.photoUrls.map((url, i) => (
                <a key={i} href={url} target="_blank" rel="noopener noreferrer">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={url} alt={`Site photo ${i + 1}`} className={styles.photo} />
                </a>
              ))}
            </div>
          </div>
        )}

        {canAct && !requestingRevision && (
          <div className={styles.actions}>
            <button className={styles.cancelBtn} onClick={onClose}>Close</button>
            <button className={styles.revisionBtn} disabled={submitting} onClick={() => setRequestingRevision(true)}>
              Request Revision
            </button>
            <button className={styles.approveBtn} disabled={submitting} onClick={handleApprove}>
              {submitting ? "Approving..." : "Approve"}
            </button>
          </div>
        )}

        {canAct && requestingRevision && (
          <div className={styles.section} style={{ marginTop: "1rem" }}>
            <div className={styles.sectionLabel}>What needs to be fixed?</div>
            <textarea
              className={styles.textarea}
              value={comment}
              onChange={(e) => setComment(e.target.value)}
              placeholder="e.g. missing financial accomplishment %, photos don't match the described issue..."
              autoFocus
            />
            <div className={styles.actions}>
              <button className={styles.cancelBtn} onClick={() => setRequestingRevision(false)}>Back</button>
              <button className={styles.approveBtn} disabled={submitting} onClick={handleSendRevision}>
                {submitting ? "Sending..." : "Send"}
              </button>
            </div>
          </div>
        )}

        {!canAct && (
          <div className={styles.actions}>
            <button className={styles.cancelBtn} onClick={onClose} style={{ flex: 1 }}>Close</button>
          </div>
        )}
      </div>
    </div>
  );
}
