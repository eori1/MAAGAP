"use client";

import { useState, useMemo } from "react";
import Sidebar from "@/components/Sidebar";
import TopRight from "@/components/TopRight";
import styles from "./page.module.css";

/* ─── Mock Timeline Data ──────────────────────────────── */
// Timeline spans 8 months: May 2025 (index 0) to Dec 2025 (index 7)
const MONTHS = [
  "May 2025", "June 2025", "July 2025", "August 2025",
  "September 2025", "October 2025", "November 2025", "December 2025"
];

const PROJECTS = [
  { id: "PPS - 124", name: "Aganan Flyover, Brgy. Aganan",           status: "Delayed",   start: 0,   duration: 2.5, overdue: 3.5 },
  { id: "PPS - 125", name: "Jibao-an Bridge, Brgy. Jibao-an",        status: "Completed", start: 3.5, duration: 1.5, overdue: 0.8 }, // Completed uses green, overdue part is light green
  { id: "PPS - 126", name: "Iloilo River Esplanade Phase 3, M...",   status: "Completed", start: 0,   duration: 2.8, overdue: 0.5 },
  { id: "PPS - 127", name: "Diversion Road Upgrade, Iloilo City",    status: "Ongoing",   start: 5.5, duration: 2.5, overdue: 0   },
  { id: "PPS - 128", name: "San Jose de Buenavista Road Impr...",    status: "Ongoing",   start: 0,   duration: 5.5, overdue: 2.5 },
  { id: "PPS - 129", name: "Terminal Market, Iloilo City",           status: "Delayed",   start: 0,   duration: 2.5, overdue: 1.0 },
  { id: "PPS - 130", name: "Road Rehabilitation, Leganes",           status: "Delayed",   start: 0,   duration: 4.5, overdue: 1.5 },
  { id: "PPS - 131", name: "Cabatangan, Cabatuan",                   status: "Delayed",   start: 3.2, duration: 3.8, overdue: 0.8 },
  { id: "PPS - 132", name: "Complex Development, Pavia",             status: "Completed", start: 0,   duration: 5.5, overdue: 1.5 },
  { id: "PPS - 133", name: "Sampaguita Farm, San Miguel",            status: "Completed", start: 0,   duration: 0.8, overdue: 0   },
  { id: "PPS - 134", name: "Port Expansion, Dumangas",               status: "Planned",   start: 0,   duration: 1.8, overdue: 2.2 },
];

const STATUS_COLORS: Record<string, { solid: string; light: string; border: string }> = {
  Completed: { solid: "#27ae60", light: "#d4efdf", border: "#27ae60" },
  Ongoing:   { solid: "#2756c5", light: "#d4e0f5", border: "#2756c5" },
  Delayed:   { solid: "#e74c3c", light: "#fadbd8", border: "#e74c3c" },
  Planned:   { solid: "#94a3b8", light: "#e2e8f0", border: "#94a3b8" },
};

/* ─── Page Component ──────────────────────────────────── */
export default function TimelinePage() {
  const [search, setSearch] = useState("");
  const [quarter, setQuarter] = useState("All Quarters");
  const [status, setStatus] = useState("All Status");
  const [sortBy, setSortBy] = useState("Most Recent");
  const [orderBy, setOrderBy] = useState("Ascending");

  const filtered = useMemo(() => {
    let list = [...PROJECTS];
    if (search) {
      list = list.filter(p => p.name.toLowerCase().includes(search.toLowerCase()) || p.id.toLowerCase().includes(search.toLowerCase()));
    }
    if (status !== "All Status") {
      list = list.filter(p => p.status === status);
    }
    return list;
  }, [search, status]);

  const clearFilters = () => { setSearch(""); setQuarter("All Quarters"); setStatus("All Status"); };

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
            <p className={styles.cardSub}>View projects timelines in a GANTT Chart</p>
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
                  <option>All Quarters</option><option>Q1</option><option>Q2</option><option>Q3</option><option>Q4</option>
                </select>
                <svg className={styles.selectChevron} width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><polyline points="6 9 12 15 18 9"/></svg>
              </div>
            </div>

            <div className={styles.filterGroup}>
              <label className={styles.filterLabel}>Status</label>
              <div className={styles.selectWrap}>
                <select className={styles.select} value={status} onChange={e => setStatus(e.target.value)}>
                  <option>All Status</option><option>Completed</option><option>Ongoing</option>
                  <option>Delayed</option><option>Planned</option>
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
                <option>Most Recent</option><option>Name</option>
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
                <div className={styles.legendColor} style={{ background: "#94a3b8" }} /> Planned
              </div>
              <div className={styles.legendItem}>
                <div className={styles.legendColorOutline} style={{ borderColor: "#e74c3c" }} /> Overdue (Red Border)
              </div>
            </div>
          </div>

          {/* Gantt Chart Area */}
          <div className={styles.ganttContainer}>
            
            {/* Header row */}
            <div className={styles.ganttHeaderRow}>
              <div className={styles.ganttLeftHeader}>Project Name</div>
              <div className={styles.ganttGridHeader}>
                {MONTHS.map(m => (
                  <div key={m} className={styles.ganttMonthBadge}>{m}</div>
                ))}
              </div>
            </div>

            {/* Rows */}
            <div className={styles.ganttBody}>
              {/* Background vertical grid lines */}
              <div className={styles.ganttBgGrid}>
                {MONTHS.map((_, i) => (
                  <div key={i} className={styles.ganttGridCol} />
                ))}
              </div>

              {filtered.map(p => {
                const colors = STATUS_COLORS[p.status];
                return (
                  <div key={p.id} className={styles.ganttRow}>
                    {/* Left: Name and ID */}
                    <div className={styles.ganttLeftCol}>
                      <div className={styles.ganttProjName}>{p.name}</div>
                      <div className={styles.ganttProjId}>{p.id}</div>
                    </div>

                    {/* Right: Timeline Grid */}
                    <div className={styles.ganttGridColWrapper}>
                      <div className={styles.ganttBarContainer}>
                        {/* Solid portion */}
                        <div 
                          className={styles.ganttBarSolid}
                          style={{
                            left: `${(p.start / 8) * 100}%`,
                            width: `${(p.duration / 8) * 100}%`,
                            background: colors.solid,
                            borderTopRightRadius: p.overdue ? 0 : 50,
                            borderBottomRightRadius: p.overdue ? 0 : 50,
                          }}
                        />
                        {/* Overdue/Light portion */}
                        {p.overdue > 0 && (
                          <div 
                            className={styles.ganttBarLight}
                            style={{
                              left: `${((p.start + p.duration) / 8) * 100}%`,
                              width: `${(p.overdue / 8) * 100}%`,
                              background: colors.light,
                              borderColor: p.status === "Delayed" ? "#e74c3c" : colors.border,
                            }}
                          />
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
              
              {filtered.length === 0 && (
                <div className={styles.emptyState}>No projects match your filters.</div>
              )}
            </div>

          </div>
        </div>
      </div>
    </div>
  );
}
