"use client";

import { useState } from "react";
import { getSupabaseBrowserClient } from "@/lib/supabaseBrowserClient";
import styles from "./SubmitReportModal.module.css";

interface Props {
  assignmentId: string;
  projectName: string;
  onClose: () => void;
  onSubmitted: () => void;
}

export default function SubmitReportModal({ assignmentId, projectName, onClose, onSubmitted }: Props) {
  const [physicalPct, setPhysicalPct] = useState("");
  const [financialPct, setFinancialPct] = useState("");
  const [issuesNoted, setIssuesNoted] = useState("");
  const [notes, setNotes] = useState("");
  const [photos, setPhotos] = useState<File[]>([]);
  const [previews, setPreviews] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  function handlePhotoChange(e: React.ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files ?? []);
    setPhotos((prev) => [...prev, ...files]);
    setPreviews((prev) => [...prev, ...files.map((f) => URL.createObjectURL(f))]);
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setSubmitting(true);

    try {
      const supabase = getSupabaseBrowserClient();
      const photoUrls: string[] = [];

      for (const file of photos) {
        const path = `${assignmentId}/${Date.now()}-${file.name}`;
        const { error: uploadError } = await supabase.storage
          .from("inspection-photos")
          .upload(path, file, { upsert: false });
        if (uploadError) throw new Error(`Photo upload failed: ${uploadError.message}`);

        const { data: urlData } = supabase.storage.from("inspection-photos").getPublicUrl(path);
        photoUrls.push(urlData.publicUrl);
      }

      const res = await fetch("/api/reports/submit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          assignmentId,
          physicalAccomplishmentPct: physicalPct ? Number(physicalPct) : null,
          financialAccomplishmentPct: financialPct ? Number(financialPct) : null,
          issuesNoted,
          notes,
          photoUrls,
        }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error ?? "Failed to submit report");
      }

      onSubmitted();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to submit report");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
        <h2 className={styles.title}>Submit Inspection Report</h2>
        <p className={styles.subtitle}>{projectName}</p>

        {error && <div className={styles.error}>{error}</div>}

        <form onSubmit={handleSubmit}>
          <div className={styles.row}>
            <div className={styles.field}>
              <label className={styles.label}>Physical accomplishment (%)</label>
              <input
                type="number" min={0} max={100} step="0.1"
                className={styles.input}
                value={physicalPct}
                onChange={(e) => setPhysicalPct(e.target.value)}
              />
            </div>
            <div className={styles.field}>
              <label className={styles.label}>Financial accomplishment (%)</label>
              <input
                type="number" min={0} max={100} step="0.1"
                className={styles.input}
                value={financialPct}
                onChange={(e) => setFinancialPct(e.target.value)}
              />
            </div>
          </div>

          <div className={styles.field}>
            <label className={styles.label}>Issues noted</label>
            <textarea
              className={styles.textarea}
              value={issuesNoted}
              onChange={(e) => setIssuesNoted(e.target.value)}
              placeholder="e.g. right-of-way dispute, weather delay, material shortage..."
            />
          </div>

          <div className={styles.field}>
            <label className={styles.label}>Additional notes</label>
            <textarea
              className={styles.textarea}
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
            />
          </div>

          <div className={styles.field}>
            <label className={styles.label}>Site photos</label>
            <input
              type="file" accept="image/*" multiple
              className={styles.fileInput}
              onChange={handlePhotoChange}
            />
            {previews.length > 0 && (
              <div className={styles.photoPreviewRow}>
                {previews.map((src, i) => (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img key={i} src={src} alt={`Site photo ${i + 1}`} className={styles.photoThumb} />
                ))}
              </div>
            )}
          </div>

          <div className={styles.actions}>
            <button type="button" className={styles.cancelBtn} onClick={onClose}>Cancel</button>
            <button type="submit" className={styles.submitBtn} disabled={submitting}>
              {submitting ? "Submitting..." : "Submit Report"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
