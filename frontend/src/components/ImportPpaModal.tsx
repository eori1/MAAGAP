"use client";

import { useRef, useState } from "react";
import { parseCsv, toCsv } from "@/lib/csv";
import { validatePpaRow, type PpaInput } from "@/lib/ppaValidation";
import shared from "./AddPpaModal.module.css";
import styles from "./ImportPpaModal.module.css";

interface Props {
  onClose: () => void;
  onCreated: () => void;
}

interface PreviewRow {
  raw: Record<string, string>;
  valid: boolean;
  error?: string;
  normalized?: PpaInput;
}

// Column header -> PpaInput field, matched case-insensitively (see
// headerKey below) so a reordered or Excel-edited file still works.
const HEADER_MAP: Record<string, keyof PpaInput> = {
  "name": "projectName",
  "project name": "projectName",
  "description": "description",
  "project type": "projectType",
  "implementing agency": "category",
  "municipality": "location",
  "budget": "budgetAllocated",
  "start date": "startDate",
  "funding source": "fundingSource",
};

function headerKey(h: string): string {
  return h.trim().toLowerCase();
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

const TEMPLATE_HEADERS = ["Name", "Description", "Project Type", "Implementing Agency", "Municipality", "Budget", "Start Date", "Funding Source"];
const TEMPLATE_EXAMPLE = [
  "Farm-to-Market Road Improvement, Brgy. Rizal",
  "Concreting of a 2km farm-to-market road to improve produce transport.",
  "Infrastructure",
  "Provincial Engineering Office",
  "Oton",
  "5000000",
  "2026-09-01",
  "General Fund",
];

export default function ImportPpaModal({ onClose, onCreated }: Props) {
  const [rows, setRows] = useState<PreviewRow[]>([]);
  const [fileInfo, setFileInfo] = useState<{ name: string; size: number } | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState<{ created: number } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  function downloadTemplate() {
    const csv = toCsv([TEMPLATE_HEADERS, TEMPLATE_EXAMPLE]);
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "maagap-ppa-import-template.csv";
    a.click();
    URL.revokeObjectURL(url);
  }

  function processFile(file: File) {
    setError(null);
    setResult(null);
    setFileInfo({ name: file.name, size: file.size });

    const reader = new FileReader();
    reader.onload = () => {
      const text = String(reader.result ?? "");
      const parsed = parseCsv(text);
      if (parsed.length < 2) {
        setError("CSV has no data rows.");
        setRows([]);
        return;
      }

      const [headerRow, ...dataRows] = parsed;
      const fieldByColumn = headerRow.map((h) => HEADER_MAP[headerKey(h)] ?? null);

      const preview: PreviewRow[] = dataRows.map((cols) => {
        const raw: Record<string, string> = {};
        const input: Partial<Record<keyof PpaInput, string>> = {};
        cols.forEach((value, i) => {
          const field = fieldByColumn[i];
          if (field) input[field] = value;
          raw[headerRow[i] ?? `col${i}`] = value;
        });
        const validation = validatePpaRow(input);
        return { raw, valid: validation.valid, error: validation.error, normalized: validation.normalized };
      });

      setRows(preview);
    };
    reader.readAsText(file);
  }

  function handleFileInput(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) processFile(file);
  }

  function handleDrop(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragActive(false);
    const file = e.dataTransfer.files?.[0];
    if (file) processFile(file);
  }

  function clearFile() {
    setFileInfo(null);
    setRows([]);
    setResult(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  const validCount = rows.filter((r) => r.valid).length;

  async function handleImport() {
    setError(null);
    setSubmitting(true);
    try {
      const res = await fetch("/api/projects/bulk", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rows: rows.filter((r) => r.valid).map((r) => r.normalized) }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error ?? "Failed to import");
      setResult({ created: data.created ?? 0 });
      onCreated();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to import");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className={shared.overlay} onClick={onClose}>
      <div className={shared.modal} style={{ maxWidth: 640 }} onClick={(e) => e.stopPropagation()}>
        <h2 className={shared.title}>Import Data</h2>
        <p className={shared.subtitle}>Bulk-add PPAs from a CSV. Imported projects show as &quot;Pending Assessment&quot; until the next pipeline run scores them.</p>

        {error && <div className={shared.error}>{error}</div>}
        {result && <div className={shared.successBanner}>Imported {result.created} project{result.created === 1 ? "" : "s"} successfully.</div>}

        <div className={styles.step}>
          <div className={styles.stepLabel}>Step 1 — Get the template</div>
          <button type="button" className={styles.templateBtn} onClick={downloadTemplate}>
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="7 10 12 15 17 10" /><line x1="12" y1="15" x2="12" y2="3" />
            </svg>
            Download CSV template
          </button>
        </div>

        <div className={styles.step}>
          <div className={styles.stepLabel}>Step 2 — Upload your completed CSV</div>

          {!fileInfo ? (
            <div
              className={`${styles.dropzone} ${dragActive ? styles.dropzoneActive : ""}`}
              onClick={() => fileInputRef.current?.click()}
              onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
              onDragLeave={() => setDragActive(false)}
              onDrop={handleDrop}
            >
              <svg className={styles.dropzoneIcon} width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="17 8 12 3 7 8" /><line x1="12" y1="3" x2="12" y2="15" />
              </svg>
              <div className={styles.dropzoneText}>Drag and drop your CSV here, or click to browse</div>
              <div className={styles.dropzoneHint}>.csv files only</div>
            </div>
          ) : (
            <div className={styles.fileChip}>
              <div className={styles.fileChipInfo}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--accent-600)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><polyline points="14 2 14 8 20 8" />
                </svg>
                <span className={styles.fileChipName}>{fileInfo.name}</span>
                <span className={styles.fileChipSize}>{formatFileSize(fileInfo.size)}</span>
              </div>
              <button type="button" className={styles.changeFileBtn} onClick={clearFile}>Change file</button>
            </div>
          )}
          <input ref={fileInputRef} type="file" accept=".csv" className={styles.hiddenInput} onChange={handleFileInput} />
        </div>

        {rows.length > 0 && (
          <div className={styles.step}>
            <div className={styles.stepLabel}>Step 3 — Review &amp; confirm</div>
            <div className={shared.previewSummary}>
              <strong>{validCount}</strong> of {rows.length} rows valid
              {validCount < rows.length ? " — invalid rows will be skipped." : "."}
            </div>
            <div className={shared.previewWrap}>
              <table className={shared.previewTable}>
                <thead>
                  <tr>
                    <th className={shared.previewTh}>#</th>
                    <th className={shared.previewTh}>Name</th>
                    <th className={shared.previewTh}>Municipality</th>
                    <th className={shared.previewTh}>Budget</th>
                    <th className={shared.previewTh}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r, i) => (
                    <tr key={i}>
                      <td className={shared.previewTd}>{i + 1}</td>
                      <td className={shared.previewTd}>{r.normalized?.projectName ?? r.raw["Name"] ?? "—"}</td>
                      <td className={shared.previewTd}>{r.normalized?.location ?? r.raw["Municipality"] ?? "—"}</td>
                      <td className={shared.previewTd}>{r.normalized?.budgetAllocated?.toLocaleString() ?? r.raw["Budget"] ?? "—"}</td>
                      <td className={shared.previewTd}>
                        {r.valid ? <span className={shared.statusValid}>✓ Valid</span> : <span className={shared.statusError}>✗ {r.error}</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        <div className={shared.actions}>
          <button type="button" className={shared.cancelBtn} onClick={onClose}>Close</button>
          <button
            type="button"
            className={shared.submitBtn}
            disabled={submitting || validCount === 0}
            onClick={handleImport}
          >
            {submitting ? "Importing..." : validCount === 0 ? "No valid rows to import" : `Import ${validCount} Valid Row${validCount === 1 ? "" : "s"}`}
          </button>
        </div>
      </div>
    </div>
  );
}
