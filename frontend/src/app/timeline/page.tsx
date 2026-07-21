"use client";

import { useState, useMemo, useEffect } from "react";
import Sidebar from "@/components/Sidebar";
import TopRight from "@/components/TopRight";
import styles from "./page.module.css";

/* ─── Types (mirror backend timeline.json payload) ────── */
interface TimelineProject {
  id: string;
  name: string;
  location: string;
  type: string;
  year: number;
  startDate: string;
  plannedEndDate: string;
  plannedMonths: number;
  actualDelayDays: number;
  predictedDelayDays: number;
  riskTier: "Low" | "Medium" | "High" | "Critical";
  status: "Completed" | "Ongoing" | "Delayed";
}

// Elapsed-months-since-start view: the manuscript delimits standard
// durations to 6 months (non-infrastructure) or 12 months (infrastructure),
// so every project is plotted on a shared 0-12 month window instead of a
// calendar axis (test projects span 2016-2025 and would not align).
const GRID_MONTHS = 12;
const MONTH_LABELS = Array.from({ length: GRID_MONTHS }, (_, i) => `Month ${i + 1}`);

const STATUS_COLORS: Record<string, { solid: string; light: string; border: string }> = {
  Completed: { solid: "#27ae60", light: "#d4efdf", border: "#27ae60" },
  Ongoing:   { solid: "#2756c5", light: "#d4e0f5", border: "#2756c5" },
  Delayed:   { solid: "#e74c3c", light: "#fadbd8", border: "#e74c3c" },
};

/* ─── Page Component ──────────────────────────────────── */
export default function TimelinePage() {
  const [projects, setProjects] = useState<TimelineProject[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState("All Status");
  const [sortBy, setSortBy] = useState("Delay (Worst First)");
  const [orderBy, setOrderBy] = useState("Descending");

  useEffect(() => {
    fetch("/api/timeline")
      .then(res => {
        if (!res.ok) throw new Error("no data");
        return res.json();
      })
      .then(setProjects)
      .catch(() => setLoadError("No timeline data found. Run the backend pipeline (python main.py) to generate it."));
  }, []);

  const filtered = useMemo(() => {
    let list = [...projects];
    if (search) {
      list = list.filter(p => p.name.toLowerCase().includes(search.toLowerCase()) || p.id.toLowerCase().includes(search.toLowerCase()));
    }
    if (status !== "All Status") {
      list = list.filter(p => p.status === status);
    }
    const dir = orderBy === "Ascending" ? 1 : -1;
    list.sort((a, b) => {
      if (sortBy === "Delay (Worst First)") {
        const da = a.status === "Completed" ? a.actualDelayDays : a.predictedDelayDays;
        const db = b.status === "Completed" ? b.actualDelayDays : b.predictedDelayDays;
        return (da - db) * -dir;
      }
      return a.name.localeCompare(b.name) * dir;
    });
    return list;
  }, [projects, search, status, sortBy, orderBy]);

  const clearFilters = () => { setSearch(""); setStatus("All Status"); };

  return (
    <div className={styles.shell}>
      <Sidebar />
      <div className={styles.main}>

        {/* ── Top bar ── */}
        <div className={styles.topbar}>
          <div className={styles.breadcrumb}>
            <span>Project Registry</span>
            <span className={styles.sep}>/</span>
            <span className={styles.breadActive}>Project Timeline</span>
          </div>
          <TopRight />
        </div>

        {/* ── Main white card ── */}
        <div className={styles.card}>

          {/* Card header */}
          <div className={styles.cardHeader}>
            <h1 className={styles.cardTitle}>
              Project <span className={styles.accent}>Timelines</span>
            </h1>
            <p className={styles.cardSub}>
              Actual vs. scheduled progress, elapsed months since project start ({filtered.length} of {projects.length} projects)
            </p>
          </div>
          <div className={styles.divider} />

          {/* Filters */}
          <div className={styles.filtersRow}>
            <div className={styles.filterGroup}>
              <label className={styles.filterLabel}>Search</label>
              <div className={styles.searchWrap}>
                <svg className={styles.searchIcon} width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
                <input
                  className={styles.searchInput}
                  placeholder="Search Projects ..."
                  value={search}
                  onChange={e => setSearch(e.target.value)}
                />
              </div>
            </div>

            <div className={styles.filterGroup}>
              <label className={styles.filterLabel}>Status</label>
              <div className={styles.selectWrap}>
                <select className={styles.select} value={status} onChange={e => setStatus(e.target.value)}>
                  <option>All Status</option><option>Completed</option><option>Ongoing</option>
                  <option>Delayed</option>
                </select>
                <svg className={styles.selectChevron} width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><polyline points="6 9 12 15 18 9"/></svg>
              </div>
            </div>

            <button className={styles.clearBtn} onClick={clearFilters}>Clear Filters</button>
          </div>

          {/* Sort + Legend */}
          <div className={styles.sortRow}>
            <div className={styles.sortControls}>
              <span className={styles.sortLabel}>Sort by:</span>
              <select className={styles.sortSelect} value={sortBy} onChange={e => setSortBy(e.target.value)}>
                <option>Delay (Worst First)</option><option>Name</option>
              </select>
              <span className={styles.sortLabel} style={{ marginLeft: "1rem" }}>Order by:</span>
              <select className={styles.sortSelect} value={orderBy} onChange={e => setOrderBy(e.target.value)}>
                <option>Ascending</option><option>Descending</option>
              </select>
            </div>

            <div className={styles.legend}>
              <span className={styles.legendLabel}>Legend</span>
              <div className={styles.legendItem}>
                <div className={styles.legendColor} style={{ background: "#27ae60" }} /> Completed
              </div>
              <div className={styles.legendItem}>
                <div className={styles.legendColor} style={{ background: "#2756c5" }} /> Ongoing
              </div>
              <div className={styles.legendItem}>
                <div className={styles.legendColor} style={{ background: "#e74c3c" }} /> Delayed
              </div>
              <div className={styles.legendItem}>
                <div className={styles.legendColorOutline} style={{ borderColor: "#e74c3c" }} /> Slippage (predicted/actual)
              </div>
            </div>
          </div>

          {/* Gantt Chart Area */}
          <div className={styles.ganttContainer}>

            {/* Header row */}
            <div className={styles.ganttHeaderRow}>
              <div className={styles.ganttLeftHeader}>Project Name</div>
              <div className={styles.ganttGridHeader}>
                {MONTH_LABELS.map(m => (
                  <div key={m} className={styles.ganttMonthBadge}>{m}</div>
                ))}
              </div>
            </div>

            {/* Rows */}
            <div className={styles.ganttBody}>
              {/* Background vertical grid lines */}
              <div className={styles.ganttBgGrid}>
                {MONTH_LABELS.map((_, i) => (
                  <div key={i} className={styles.ganttGridCol} />
                ))}
              </div>

              {loadError && <div className={styles.emptyState}>{loadError}</div>}

              {!loadError && filtered.map(p => {
                const colors = STATUS_COLORS[p.status];
                const duration = Math.min(p.plannedMonths, GRID_MONTHS);
                const delayDays = p.status === "Completed" ? p.actualDelayDays : p.predictedDelayDays;
                const overdueMonths = Math.max(0, Math.min(delayDays / 30, GRID_MONTHS - duration));
                return (
                  <div key={p.id} className={styles.ganttRow}>
                    {/* Left: Name and ID */}
                    <div className={styles.ganttLeftCol}>
                      <div className={styles.ganttProjName}>{p.name} — {p.location}</div>
                      <div className={styles.ganttProjId}>{p.type} · {p.riskTier} risk</div>
                    </div>

                    {/* Right: Timeline Grid */}
                    <div className={styles.ganttGridColWrapper}>
                      <div className={styles.ganttBarContainer}>
                        {/* Solid portion: scheduled duration */}
                        <div
                          className={styles.ganttBarSolid}
                          style={{
                            left: "0%",
                            width: `${(duration / GRID_MONTHS) * 100}%`,
                            background: colors.solid,
                            borderTopRightRadius: overdueMonths ? 0 : 50,
                            borderBottomRightRadius: overdueMonths ? 0 : 50,
                          }}
                          title={`Scheduled: ${p.plannedMonths} months`}
                        />
                        {/* Overdue/light portion: actual or predicted slippage */}
                        {overdueMonths > 0 && (
                          <div
                            className={styles.ganttBarLight}
                            style={{
                              left: `${(duration / GRID_MONTHS) * 100}%`,
                              width: `${(overdueMonths / GRID_MONTHS) * 100}%`,
                              background: colors.light,
                              borderColor: colors.border,
                            }}
                            title={`${p.status === "Completed" ? "Actual" : "Predicted"} slippage: ${delayDays.toFixed(0)} days`}
                          />
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}

              {!loadError && filtered.length === 0 && (
                <div className={styles.emptyState}>No projects match your filters.</div>
              )}
            </div>

          </div>
        </div>
      </div>
    </div>
  );
}
