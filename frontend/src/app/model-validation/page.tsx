"use client";

import { useEffect, useMemo, useState } from "react";
import Sidebar from "@/components/Sidebar";
import TopRight from "@/components/TopRight";
import Skeleton from "@/components/ui/Skeleton";
import EmptyState from "@/components/ui/EmptyState";
import Badge from "@/components/ui/Badge";
import StatCard from "@/components/ui/StatCard";
import { RISK_TONE } from "@/lib/riskTone";
import styles from "./page.module.css";

interface ValidationRow {
  projectId: string;
  riskTier: "Low" | "Medium" | "High" | "Critical";
  delayProbability: number | null;
  predictedDelayDays: number | null;
  expectedProgress: number | null;
  actualProgress: number | null;
  financialAccomplishmentPct: number | null;
  slippage: number | null;
  agreement: "confirmed" | "contradicted" | "inconclusive";
  reportDate: string;
  inspectorName: string;
}

type AgreementFilter = "all" | ValidationRow["agreement"];

const AGREEMENT_LABEL: Record<ValidationRow["agreement"], string> = {
  confirmed: "Confirmed",
  contradicted: "Contradicted",
  inconclusive: "Inconclusive",
};

const AGREEMENT_TONE: Record<ValidationRow["agreement"], "good" | "critical" | "neutral"> = {
  confirmed: "good",
  contradicted: "critical",
  inconclusive: "neutral",
};

function FolderIcon() {
  return (
    <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
      <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
    </svg>
  );
}
function CheckIcon() {
  return (
    <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
      <rect x="3" y="4" width="18" height="18" rx="2" /><polyline points="9 12 11 14 15 10" />
    </svg>
  );
}
function AlertIcon() {
  return (
    <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
      <line x1="12" y1="9" x2="12" y2="13" /><line x1="12" y1="17" x2="12.01" y2="17" />
    </svg>
  );
}
function QuestionIcon() {
  return (
    <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
      <circle cx="12" cy="12" r="9" /><path d="M9.5 9a2.5 2.5 0 0 1 5 0c0 1.5-2.5 2-2.5 3.5" /><line x1="12" y1="17" x2="12.01" y2="17" />
    </svg>
  );
}

export default function ModelValidationPage() {
  const [rows, setRows] = useState<ValidationRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [agreementFilter, setAgreementFilter] = useState<AgreementFilter>("all");

  useEffect(() => {
    fetch("/api/model-validation")
      .then((res) => {
        if (!res.ok) throw new Error("no data");
        return res.json();
      })
      .then((data) => { setRows(data); setLoading(false); })
      .catch(() => {
        setLoadError("No field reports submitted yet — validation will appear here once inspectors submit reports.");
        setLoading(false);
      });
  }, []);

  const counts = useMemo(() => ({
    total: rows.length,
    confirmed: rows.filter((r) => r.agreement === "confirmed").length,
    contradicted: rows.filter((r) => r.agreement === "contradicted").length,
    inconclusive: rows.filter((r) => r.agreement === "inconclusive").length,
  }), [rows]);

  const filteredRows = useMemo(() => {
    if (agreementFilter === "all") return rows;
    return rows.filter((r) => r.agreement === agreementFilter);
  }, [rows, agreementFilter]);

  function toggleFilter(f: AgreementFilter) {
    setAgreementFilter((prev) => (prev === f ? "all" : f));
  }

  return (
    <div className={styles.shell}>
      <Sidebar />
      <div className={styles.main}>

        <div className={styles.topbar}>
          <div className={styles.breadcrumb}>
            <span>Model Validation</span>
            <span className={styles.sep}>/</span>
            <span className={styles.breadActive}>Prediction vs. Reality</span>
          </div>
          <TopRight />
        </div>

        <div className={styles.headCard}>
          <h1 className={styles.headTitle}>Model Validation</h1>
          <p className={styles.headSub}>How the model&apos;s risk predictions compare against what inspectors actually reported in the field</p>
        </div>

        <div className={styles.statsRow}>
          <div className={styles.statTile}>
            <StatCard label="Total Validated" value={counts.total} tone="accent" loading={loading} icon={<FolderIcon />} />
          </div>
          <div
            className={`${styles.statTile} ${agreementFilter === "confirmed" ? styles.statTileActive : ""}`}
            onClick={() => toggleFilter("confirmed")}
          >
            <StatCard label="Confirmed" value={counts.confirmed} tone="good" loading={loading} icon={<CheckIcon />} />
          </div>
          <div
            className={`${styles.statTile} ${agreementFilter === "contradicted" ? styles.statTileActive : ""}`}
            onClick={() => toggleFilter("contradicted")}
          >
            <StatCard label="Contradicted" value={counts.contradicted} tone="critical" loading={loading} icon={<AlertIcon />} />
          </div>
          <div
            className={`${styles.statTile} ${agreementFilter === "inconclusive" ? styles.statTileActive : ""}`}
            onClick={() => toggleFilter("inconclusive")}
          >
            <StatCard label="Inconclusive" value={counts.inconclusive} tone="neutral" loading={loading} icon={<QuestionIcon />} />
          </div>
        </div>

        <div className={styles.card}>
          <div className={styles.cardHeader}>
            <h2 className={styles.cardTitle}>Predicted <span style={{ color: "var(--accent-600)" }}>vs.</span> Reported</h2>
            <p className={styles.cardSub}>Only lists projects with at least one real inspection report — nothing to validate against otherwise.</p>
          </div>
          <div className={styles.divider} />

          <div className={styles.filterRow}>
            {(["all", "confirmed", "contradicted", "inconclusive"] as AgreementFilter[]).map((f) => (
              <button
                key={f}
                className={`${styles.chip} ${agreementFilter === f ? styles.chipActive : ""}`}
                onClick={() => setAgreementFilter(f)}
              >
                {f === "all" ? "All" : AGREEMENT_LABEL[f]}
              </button>
            ))}
          </div>

          <div className={styles.tableWrap}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th className={styles.th}>Project</th>
                  <th className={styles.th} style={{ textAlign: "center" }}>Predicted Risk</th>
                  <th className={styles.th} style={{ textAlign: "center" }}>Delay Probability</th>
                  <th className={styles.th} style={{ textAlign: "center" }}>Expected Progress</th>
                  <th className={styles.th} style={{ textAlign: "center" }}>Actual Progress</th>
                  <th className={styles.th} style={{ textAlign: "center" }}>Slippage</th>
                  <th className={styles.th} style={{ textAlign: "center" }}>Agreement</th>
                  <th className={styles.th}>Inspector</th>
                  <th className={styles.th} style={{ textAlign: "center" }}>Report Date</th>
                </tr>
              </thead>
              <tbody>
                {loading && Array.from({ length: 5 }).map((_, i) => (
                  <tr key={i} className={styles.row}>
                    {Array.from({ length: 9 }).map((__, j) => (
                      <td key={j} className={styles.td}><Skeleton height="1rem" /></td>
                    ))}
                  </tr>
                ))}

                {!loading && !loadError && filteredRows.map((r) => (
                  <tr key={r.projectId} className={styles.row}>
                    <td className={styles.td}><div className={styles.projectName}>{r.projectId}</div></td>
                    <td className={`${styles.td} ${styles.tdCenter}`}>
                      <Badge tone={RISK_TONE[r.riskTier] ?? "neutral"}>{r.riskTier}</Badge>
                    </td>
                    <td className={`${styles.td} ${styles.tdCenter}`}>{r.delayProbability !== null ? `${(r.delayProbability * 100).toFixed(0)}%` : "—"}</td>
                    <td className={`${styles.td} ${styles.tdCenter}`}>{r.expectedProgress !== null ? `${r.expectedProgress.toFixed(0)}%` : "—"}</td>
                    <td className={`${styles.td} ${styles.tdCenter}`}>{r.actualProgress !== null ? `${r.actualProgress.toFixed(0)}%` : "Not reported"}</td>
                    <td className={`${styles.td} ${styles.tdCenter}`} style={r.slippage !== null ? { color: r.slippage > 5 ? "var(--status-critical-text)" : "var(--status-good-text)", fontWeight: 700 } : { color: "var(--ink-500)" }}>
                      {r.slippage !== null ? `${r.slippage > 0 ? "-" : "+"}${Math.abs(r.slippage).toFixed(1)} pts` : "—"}
                    </td>
                    <td className={`${styles.td} ${styles.tdCenter}`}>
                      <Badge tone={AGREEMENT_TONE[r.agreement]}>{AGREEMENT_LABEL[r.agreement]}</Badge>
                    </td>
                    <td className={styles.td}>{r.inspectorName}</td>
                    <td className={`${styles.td} ${styles.tdCenter}`}>{r.reportDate}</td>
                  </tr>
                ))}
              </tbody>
            </table>

            {loadError && <EmptyState title="No validation data yet" message={loadError} />}
            {!loading && !loadError && filteredRows.length === 0 && (
              <EmptyState title="No rows match this filter" message="Try a different agreement filter." />
            )}
          </div>
        </div>

      </div>
    </div>
  );
}
