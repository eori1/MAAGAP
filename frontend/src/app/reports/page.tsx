"use client";

import { useState, useMemo, useEffect } from "react";
import Sidebar from "@/components/Sidebar";
import TopRight from "@/components/TopRight";
import ReportDetailModal, { ReportDetail } from "@/components/ReportDetailModal";
import styles from "./page.module.css";

/* ─── Types (mirror /api/reports payload) ─────── */
interface InspectionReport {
  projectId: string;
  source: "inspector" | "pipeline";
  quarter: number | null;
  totalQuarters: number | null;
  plannedProgress: number | null;
  actualProgress: number | null;
  slippage: number | null;
  issuesSummary: string;
  notes: string;
  photoUrls: string[];
  date: string;
  status: "Validated" | "Pending Review" | "Flagged" | "Submitted";
  inspectorId: string | null;
  inspectorName: string;
  riskTier: "Low" | "Medium" | "High" | "Critical";
  reportId: string | null;
  reviewStatus: "pending" | "approved" | "needs_revision" | null;
  reviewComment: string | null;
  financialAccomplishmentPct: number | null;
}

const REVIEW_STYLE: Record<string, { bg: string; color: string; label: string }> = {
  pending:        { bg: "#f1f5f9", color: "#7a8fa6", label: "Awaiting Review" },
  approved:       { bg: "#d4efdf", color: "#1e8449", label: "Approved" },
  needs_revision: { bg: "#fde2e2", color: "#c0392b", label: "Needs Revision" },
};

const STATUS_STYLE: Record<string, { bg: string; color: string }> = {
  Validated:        { bg: "#27ae60", color: "#fff" },
  "Pending Review":  { bg: "#f59e0b", color: "#fff" },
  Flagged:          { bg: "#e74c3c", color: "#fff" },
  Submitted:        { bg: "#2756c5", color: "#fff" },
};

const SOURCE_STYLE: Record<string, { bg: string; color: string; label: string }> = {
  inspector: { bg: "#e0f0ff", color: "#1a6ed8", label: "Field Report" },
  pipeline:  { bg: "#f1f5f9", color: "#64748b", label: "Pipeline Estimate" },
};

const QUARTER_OPTIONS = ["All Quarters", "Q1", "Q2", "Q3", "Q4"];
const SOURCE_OPTIONS = ["All Sources", "Field Report", "Pipeline Estimate"];

/* ─── Page Component ──────────────────────────────────── */
export default function ReportsPage() {
  const [reports, setReports] = useState<InspectionReport[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [search,  setSearch]  = useState("");
  const [quarter, setQuarter] = useState("All Quarters");
  const [status,  setStatus]  = useState("All Status");
  const [source,  setSource]  = useState("All Sources");
  const [sortBy,  setSortBy]  = useState("Most Recent");
  const [orderBy, setOrderBy] = useState("Descending");
  const [viewerRole, setViewerRole] = useState<"manager" | "inspector" | "admin" | null>(null);
  const [detailTarget, setDetailTarget] = useState<InspectionReport | null>(null);

  const loadReports = () => {
    fetch("/api/reports")
      .then(res => {
        if (!res.ok) throw new Error("no data");
        return res.json();
      })
      .then(setReports)
      .catch(() => setLoadError("No inspection reports found. Run the backend pipeline (python main.py) to generate them."));
  };

  useEffect(() => {
    loadReports();
    fetch("/api/me")
      .then(res => (res.ok ? res.json() : null))
      .then(profile => setViewerRole(profile?.role ?? null))
      .catch(() => setViewerRole(null));
  }, []);

  const canReview = viewerRole === "manager" || viewerRole === "admin";

  function handleReviewed() {
    setDetailTarget(null);
    loadReports();
  }

  const filtered = useMemo(() => {
    let list = [...reports];
    if (search)                     list = list.filter(r => r.projectId.toLowerCase().includes(search.toLowerCase()) || r.inspectorName.toLowerCase().includes(search.toLowerCase()));
    if (quarter !== "All Quarters") list = list.filter(r => `Q${r.quarter}` === quarter);
    if (status  !== "All Status")   list = list.filter(r => r.status === status);
    if (source  !== "All Sources")  list = list.filter(r => SOURCE_STYLE[r.source].label === source);

    const dir = orderBy === "Ascending" ? 1 : -1;
    list.sort((a, b) => {
      if (sortBy === "Progress")  return ((a.actualProgress ?? -1) - (b.actualProgress ?? -1)) * dir;
      if (sortBy === "Quarter")   return ((a.quarter ?? 0) - (b.quarter ?? 0)) * dir;
      if (sortBy === "Name")      return a.projectId.localeCompare(b.projectId) * dir;
      return (new Date(a.date).getTime() - new Date(b.date).getTime()) * dir;
    });
    return list;
  }, [reports, search, quarter, status, source, sortBy, orderBy]);

  const clearFilters = () => { setSearch(""); setQuarter("All Quarters"); setStatus("All Status"); setSource("All Sources"); };
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
              <p className={styles.cardSub}>Latest inspection per project — real field reports where submitted, pipeline estimate otherwise ({filtered.length} of {reports.length})</p>
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
                  <option>Validated</option><option>Pending Review</option><option>Flagged</option><option>Submitted</option>
                </select>
                <svg className={styles.selectChevron} width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><polyline points="6 9 12 15 18 9"/></svg>
              </div>
            </div>

            <div className={styles.filterGroup}>
              <label className={styles.filterLabel}>Source</label>
              <div className={styles.selectWrap}>
                <select className={styles.select} value={source} onChange={e => setSource(e.target.value)}>
                  {SOURCE_OPTIONS.map(s => <option key={s}>{s}</option>)}
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
                  <th className={styles.th} style={{ textAlign:"center" }}>Source</th>
                  <th className={styles.th} style={{ textAlign:"center" }}>Progress</th>
                  <th className={styles.th} style={{ textAlign:"center" }}>Slippage</th>
                  <th className={styles.th} style={{ textAlign:"center" }}>Date</th>
                  <th className={styles.th} style={{ textAlign:"center" }}>Status</th>
                  <th className={styles.th} style={{ textAlign:"center" }}>Review Status</th>
                  <th className={styles.th}>Inspector</th>
                </tr>
              </thead>
              <tbody>
                {loadError && (
                  <tr><td colSpan={8} className={styles.emptyRow}>{loadError}</td></tr>
                )}

                {!loadError && filtered.map((r) => {
                  const st = STATUS_STYLE[r.status] ?? { bg:"#94a3b8", color:"#fff" };
                  const src = SOURCE_STYLE[r.source];
                  return (
                    <tr key={`${r.projectId}-${r.date}`} className={styles.row}>
                      <td className={styles.td}>
                        <div className={styles.projectName}>{r.projectId}</div>
                        <div className={styles.projectId}>{r.riskTier} risk · {r.issuesSummary}</div>
                        {r.notes && <div className={styles.projectId} style={{ fontStyle: "italic" }}>{r.notes}</div>}
                        {r.photoUrls.length > 0 && (
                          <div style={{ display: "flex", gap: 4, marginTop: 4 }}>
                            {r.photoUrls.map((url, i) => (
                              // eslint-disable-next-line @next/next/no-img-element
                              <a key={i} href={url} target="_blank" rel="noopener noreferrer">
                                <img src={url} alt={`Site photo ${i + 1}`} style={{ width: 32, height: 32, borderRadius: 6, objectFit: "cover", border: "1px solid #e0eaf5" }} />
                              </a>
                            ))}
                          </div>
                        )}
                        <div className={styles.progressBarBg}>
                          <div className={styles.progressBar} style={{ width:`${r.actualProgress ?? 0}%` }} />
                        </div>
                      </td>
                      <td className={`${styles.td} ${styles.tdCenter}`}>
                        <span className={styles.statusBadge} style={{ background: src.bg, color: src.color }}>
                          {r.source === "pipeline" && r.quarter ? `Q${r.quarter}/${r.totalQuarters}` : src.label}
                        </span>
                      </td>
                      <td className={`${styles.td} ${styles.tdCenter}`}>{r.actualProgress !== null ? `${r.actualProgress.toFixed(0)}%` : "—"}</td>
                      <td className={`${styles.td} ${styles.tdCenter}`} style={r.slippage !== null ? { color: r.slippage > 5 ? "#e74c3c" : "#27ae60", fontWeight: 700 } : { color: "#94a3b8" }}>
                        {r.slippage !== null ? `${r.slippage > 0 ? "-" : "+"}${Math.abs(r.slippage).toFixed(1)} pts` : "Not reported"}
                      </td>
                      <td className={`${styles.td} ${styles.tdCenter}`}>{r.date}</td>
                      <td className={`${styles.td} ${styles.tdCenter}`}>
                        <span className={styles.statusBadge} style={{ background: st.bg, color: st.color }}>
                          {r.status}
                        </span>
                      </td>
                      <td className={`${styles.td} ${styles.tdCenter}`}>
                        {r.reviewStatus && r.reportId ? (
                          <>
                            <span className={styles.statusBadge} style={{ background: REVIEW_STYLE[r.reviewStatus].bg, color: REVIEW_STYLE[r.reviewStatus].color }}>
                              {REVIEW_STYLE[r.reviewStatus].label}
                            </span>
                            <div style={{ marginTop: 6 }}>
                              <button className={styles.clearBtn} onClick={() => setDetailTarget(r)}>
                                {canReview && r.reviewStatus === "pending" ? "Review Report" : "View Report"}
                              </button>
                            </div>
                          </>
                        ) : (
                          "—"
                        )}
                      </td>
                      <td className={styles.td}>{r.inspectorName}</td>
                    </tr>
                  );
                })}

                {!loadError && filtered.length === 0 && (
                  <tr>
                    <td colSpan={8} className={styles.emptyRow}>No reports found matching your filters.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

        </div>
      </div>

      {detailTarget && detailTarget.reportId && (
        <ReportDetailModal
          report={{
            reportId: detailTarget.reportId,
            projectId: detailTarget.projectId,
            inspectorName: detailTarget.inspectorName,
            date: detailTarget.date,
            actualProgress: detailTarget.actualProgress,
            financialAccomplishmentPct: detailTarget.financialAccomplishmentPct,
            slippage: detailTarget.slippage,
            issuesSummary: detailTarget.issuesSummary,
            notes: detailTarget.notes,
            photoUrls: detailTarget.photoUrls,
            reviewStatus: detailTarget.reviewStatus,
            reviewComment: detailTarget.reviewComment,
          } as ReportDetail}
          canReview={canReview}
          onClose={() => setDetailTarget(null)}
          onReviewed={handleReviewed}
        />
      )}
    </div>
  );
}
