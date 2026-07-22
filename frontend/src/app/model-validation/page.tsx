"use client";

import { useEffect, useMemo, useState } from "react";
import Sidebar from "@/components/Sidebar";
import TopRight from "@/components/TopRight";
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

const AGREEMENT_STYLE: Record<ValidationRow["agreement"], { bg: string; color: string; label: string }> = {
  confirmed:     { bg: "#d4efdf", color: "#1e8449", label: "Confirmed" },
  contradicted:  { bg: "#fde2e2", color: "#c0392b", label: "Contradicted" },
  inconclusive:  { bg: "#f1f5f9", color: "#7a8fa6", label: "Inconclusive" },
};

const RISK_TIER_STYLE: Record<string, { bg: string; color: string }> = {
  Low:      { bg: "#d4efdf", color: "#1e8449" },
  Medium:   { bg: "#fff3cd", color: "#8a6d00" },
  High:     { bg: "#fde2e2", color: "#c0392b" },
  Critical: { bg: "#c0392b", color: "#fff" },
};

export default function ModelValidationPage() {
  const [rows, setRows] = useState<ValidationRow[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/model-validation")
      .then(res => {
        if (!res.ok) throw new Error("no data");
        return res.json();
      })
      .then(setRows)
      .catch(() => setLoadError("No field reports submitted yet — validation will appear here once inspectors submit reports."));
  }, []);

  const counts = useMemo(() => ({
    total: rows.length,
    confirmed: rows.filter(r => r.agreement === "confirmed").length,
    contradicted: rows.filter(r => r.agreement === "contradicted").length,
    inconclusive: rows.filter(r => r.agreement === "inconclusive").length,
  }), [rows]);

  return (
    <div className={styles.shell}>
      <Sidebar />
      <div className={styles.main}>

        {/* Top bar */}
        <div className={styles.topbar}>
          <div className={styles.breadcrumb}>
            <span>Model Validation</span>
            <span className={styles.sep}>/</span>
            <span className={styles.breadActive}>Prediction vs. Reality</span>
          </div>
          <TopRight />
        </div>

        {/* Page heading */}
        <div className={styles.headCard}>
          <h1 className={styles.headTitle}>Model Validation</h1>
          <p className={styles.headSub}>How the model&apos;s risk predictions compare against what inspectors actually reported in the field</p>
        </div>

        {/* Stats row */}
        <div className={styles.statsRow}>
          {[
            { label: "Projects Validated", value: counts.total, color: "#2756c5" },
            { label: "Confirmed",          value: counts.confirmed, color: "#1e8449" },
            { label: "Contradicted",       value: counts.contradicted, color: "#c0392b" },
            { label: "Inconclusive",       value: counts.inconclusive, color: "#7a8fa6" },
          ].map(({ label, value, color }) => (
            <div key={label} className={styles.statCard}>
              <div className={styles.statLabel}>{label}</div>
              <div className={styles.statValue} style={{ color }}>{value}</div>
            </div>
          ))}
        </div>

        {/* Table card */}
        <div className={styles.card}>
          <div className={styles.cardHeader}>
            <h2 className={styles.cardTitle}>Predicted <span style={{ color: "#2756c5" }}>vs.</span> Reported</h2>
            <p className={styles.cardSub}>Only lists projects with at least one real inspection report — nothing to validate against otherwise.</p>
          </div>
          <div className={styles.divider} />

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
                {loadError && (
                  <tr><td colSpan={9} className={styles.emptyRow}>{loadError}</td></tr>
                )}

                {!loadError && rows.map((r) => {
                  const tier = RISK_TIER_STYLE[r.riskTier] ?? { bg: "#94a3b8", color: "#fff" };
                  const ag = AGREEMENT_STYLE[r.agreement];
                  return (
                    <tr key={r.projectId} className={styles.row}>
                      <td className={styles.td}><div className={styles.projectName}>{r.projectId}</div></td>
                      <td className={`${styles.td} ${styles.tdCenter}`}>
                        <span className={styles.statusBadge} style={{ background: tier.bg, color: tier.color }}>{r.riskTier}</span>
                      </td>
                      <td className={`${styles.td} ${styles.tdCenter}`}>{r.delayProbability !== null ? `${(r.delayProbability * 100).toFixed(0)}%` : "—"}</td>
                      <td className={`${styles.td} ${styles.tdCenter}`}>{r.expectedProgress !== null ? `${r.expectedProgress.toFixed(0)}%` : "—"}</td>
                      <td className={`${styles.td} ${styles.tdCenter}`}>{r.actualProgress !== null ? `${r.actualProgress.toFixed(0)}%` : "Not reported"}</td>
                      <td className={`${styles.td} ${styles.tdCenter}`} style={r.slippage !== null ? { color: r.slippage > 5 ? "#e74c3c" : "#27ae60", fontWeight: 700 } : { color: "#94a3b8" }}>
                        {r.slippage !== null ? `${r.slippage > 0 ? "-" : "+"}${Math.abs(r.slippage).toFixed(1)} pts` : "—"}
                      </td>
                      <td className={`${styles.td} ${styles.tdCenter}`}>
                        <span className={styles.statusBadge} style={{ background: ag.bg, color: ag.color }}>{ag.label}</span>
                      </td>
                      <td className={styles.td}>{r.inspectorName}</td>
                      <td className={`${styles.td} ${styles.tdCenter}`}>{r.reportDate}</td>
                    </tr>
                  );
                })}

                {!loadError && rows.length === 0 && (
                  <tr>
                    <td colSpan={9} className={styles.emptyRow}>No field reports submitted yet — validation will appear here once inspectors submit reports.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

      </div>
    </div>
  );
}
