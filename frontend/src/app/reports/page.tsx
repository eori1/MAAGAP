"use client";

import { useState, useMemo, useEffect } from "react";
import Sidebar from "@/components/Sidebar";
import TopRight from "@/components/TopRight";
import styles from "./page.module.css";

/* ─── Types (mirror backend reports.json payload) ─────── */
interface InspectionReport {
  projectId: string;
  quarter: number;
  totalQuarters: number;
  plannedProgress: number;
  actualProgress: number;
  slippage: number;
  issues: number;
  date: string;
  status: "Validated" | "Pending Review" | "Flagged";
  inspectorId: string;
  inspectorName: string;
  riskTier: "Low" | "Medium" | "High" | "Critical";
}

const STATUS_STYLE: Record<string, { bg: string; color: string }> = {
  Validated:        { bg: "#27ae60", color: "#fff" },
  "Pending Review":  { bg: "#f59e0b", color: "#fff" },
  Flagged:          { bg: "#e74c3c", color: "#fff" },
};

const QUARTER_OPTIONS = ["All Quarters", "Q1", "Q2", "Q3", "Q4"];

/* ─── Page Component ──────────────────────────────────── */
export default function ReportsPage() {
  const [reports, setReports] = useState<InspectionReport[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [search,  setSearch]  = useState("");
  const [quarter, setQuarter] = useState("All Quarters");
  const [status,  setStatus]  = useState("All Status");
  const [sortBy,  setSortBy]  = useState("Most Recent");
  const [orderBy, setOrderBy] = useState("Descending");

  useEffect(() => {
    fetch("/api/reports")
      .then(res => {
        if (!res.ok) throw new Error("no data");
        return res.json();
      })
      .then(setReports)
      .catch(() => setLoadError("No inspection reports found. Run the backend pipeline (python main.py) to generate them."));
  }, []);

  const filtered = useMemo(() => {
    let list = [...reports];
    if (search)                     list = list.filter(r => r.projectId.toLowerCase().includes(search.toLowerCase()) || r.inspectorName.toLowerCase().includes(search.toLowerCase()));
    if (quarter !== "All Quarters") list = list.filter(r => `Q${r.quarter}` === quarter);
    if (status  !== "All Status")   list = list.filter(r => r.status === status);

    const dir = orderBy === "Ascending" ? 1 : -1;
    list.sort((a, b) => {
      if (sortBy === "Progress")  return (a.actualProgress - b.actualProgress) * dir;
      if (sortBy === "Quarter")   return (a.quarter - b.quarter) * dir;
      if (sortBy === "Name")      return a.projectId.localeCompare(b.projectId) * dir;
      return (new Date(a.date).getTime() - new Date(b.date).getTime()) * dir;
    });
    return list;
  }, [reports, search, quarter, status, sortBy, orderBy]);

  const clearFilters = () => { setSearch(""); setQuarter("All Quarters"); setStatus("All Status"); };
  const exportPdf = () => window.print();

  return (
    <div className={styles.shell}>
      <div data-print-hide><Sidebar /></div>
      <div className={styles.main}>

        {/* ── Top bar ── */}
        <div className={styles.topbar} data-print-hide>
          <div className={styles.breadcrumb}>
            <span>Reports</span>
            <span className={styles.sep}>/</span>
            <span className={styles.breadActive}>Project Reports</span>
          </div>
          <TopRight />
        </div>

        {/* ── Main white card ── */}
        <div className={styles.card} data-print-area>

          {/* Card header */}
          <div className={styles.cardHeader}>
            <div>
              <h1 className={styles.cardTitle}>
                Project <span className={styles.accent}>Reports</span>
              </h1>
              <p className={styles.cardSub}>Latest quarterly inspection log per project ({filtered.length} of {reports.length})</p>
            </div>
            <button className={styles.exportBtn} onClick={exportPdf} data-print-hide>
              Export Report (PDF)
            </button>
          </div>
          <div className={styles.divider} />

          {/* Filters */}
          <div className={styles.filtersRow} data-print-hide>
            <div className={styles.filterGroup}>
              <label className={styles.filterLabel}>Search</label>
              <div className={styles.searchWrap}>
                <svg className={styles.searchIcon} width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
                <input
                  className={styles.searchInput}
                  placeholder="Search Reports ..."
                  value={search}
                  onChange={e => setSearch(e.target.value)}
                />
              </div>
            </div>

            <div className={styles.filterGroup}>
              <label className={styles.filterLabel}>Quarter</label>
              <div className={styles.selectWrap}>
                <select className={styles.select} value={quarter} onChange={e => setQuarter(e.target.value)}>
                  {QUARTER_OPTIONS.map(q => <option key={q}>{q}</option>)}
                </select>
                <svg className={styles.selectChevron} width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><polyline points="6 9 12 15 18 9"/></svg>
              </div>
            </div>

            <div className={styles.filterGroup}>
              <label className={styles.filterLabel}>Status</label>
              <div className={styles.selectWrap}>
                <select className={styles.select} value={status} onChange={e => setStatus(e.target.value)}>
                  <option>All Status</option>
                  <option>Validated</option><option>Pending Review</option><option>Flagged</option>
                </select>
                <svg className={styles.selectChevron} width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><polyline points="6 9 12 15 18 9"/></svg>
              </div>
            </div>

            <button className={styles.clearBtn} onClick={clearFilters}>Clear Filters</button>
          </div>

          {/* Sort + count row */}
          <div className={styles.sortRow} data-print-hide>
            <div className={styles.sortControls}>
              <span className={styles.sortLabel}>Sort by:</span>
              <select className={styles.sortSelect} value={sortBy} onChange={e => setSortBy(e.target.value)}>
                <option>Most Recent</option><option>Name</option><option>Progress</option><option>Quarter</option>
              </select>
              <span className={styles.sortLabel} style={{ marginLeft:"1rem" }}>Order by:</span>
              <select className={styles.sortSelect} value={orderBy} onChange={e => setOrderBy(e.target.value)}>
                <option>Ascending</option><option>Descending</option>
              </select>
            </div>
            <div className={styles.countBadge}>{filtered.length} Reports Found</div>
          </div>

          {/* Table */}
          <div className={styles.tableWrap}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th className={styles.th}>Project</th>
                  <th className={styles.th} style={{ textAlign:"center" }}>Quarter</th>
                  <th className={styles.th} style={{ textAlign:"center" }}>Progress</th>
                  <th className={styles.th} style={{ textAlign:"center" }}>Slippage</th>
                  <th className={styles.th} style={{ textAlign:"center" }}>Date</th>
                  <th className={styles.th} style={{ textAlign:"center" }}>Status</th>
                  <th className={styles.th}>Inspector</th>
                </tr>
              </thead>
              <tbody>
                {loadError && (
                  <tr><td colSpan={7} className={styles.emptyRow}>{loadError}</td></tr>
                )}

                {!loadError && filtered.map((r) => {
                  const st = STATUS_STYLE[r.status] ?? { bg:"#94a3b8", color:"#fff" };
                  return (
                    <tr key={`${r.projectId}-${r.quarter}`} className={styles.row}>
                      <td className={styles.td}>
                        <div className={styles.projectName}>{r.projectId}</div>
                        <div className={styles.projectId}>{r.riskTier} risk · {r.issues} issue{r.issues === 1 ? "" : "s"} noted</div>
                        <div className={styles.progressBarBg}>
                          <div className={styles.progressBar} style={{ width:`${r.actualProgress}%` }} />
                        </div>
                      </td>
                      <td className={`${styles.td} ${styles.tdCenter}`}>Q{r.quarter}/{r.totalQuarters}</td>
                      <td className={`${styles.td} ${styles.tdCenter}`}>{r.actualProgress.toFixed(0)}%</td>
                      <td className={`${styles.td} ${styles.tdCenter}`} style={{ color: r.slippage > 5 ? "#e74c3c" : "#27ae60", fontWeight: 700 }}>
                        {r.slippage > 0 ? "-" : "+"}{Math.abs(r.slippage).toFixed(1)} pts
                      </td>
                      <td className={`${styles.td} ${styles.tdCenter}`}>{r.date}</td>
                      <td className={`${styles.td} ${styles.tdCenter}`}>
                        <span className={styles.statusBadge} style={{ background: st.bg, color: st.color }}>
                          {r.status}
                        </span>
                      </td>
                      <td className={styles.td}>{r.inspectorName}</td>
                    </tr>
                  );
                })}

                {!loadError && filtered.length === 0 && (
                  <tr>
                    <td colSpan={7} className={styles.emptyRow}>No reports found matching your filters.</td>
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
