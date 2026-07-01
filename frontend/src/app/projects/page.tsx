"use client";

import { useState, useRef } from "react";
import dynamic from "next/dynamic";
import Sidebar from "@/components/Sidebar";
import TopRight from "@/components/TopRight";
import styles from "./page.module.css";
import type { ProjectPin } from "@/components/IloiloMap";

/* ── Dynamic import — Leaflet needs client-side only ─ */
const IloiloMap = dynamic(() => import("@/components/IloiloMap"), {
  ssr: false,
  loading: () => (
    <div style={{
      width: "100%", height: "100%",
      display: "flex", alignItems: "center", justifyContent: "center",
      background: "linear-gradient(145deg,#cde8f7,#a8d4ef)",
      color: "#1264ae", fontWeight: 700, fontSize: "1rem",
    }}>
      Loading map…
    </div>
  ),
});

import { useEffect } from "react";
import { fetchProjects } from "@/lib/api";
/* ── Badge helpers ────────────────────────────────── */
function riskBarClass(risk: string, s: typeof styles) {
  if (risk === "High" || risk === "Critical") return `${s.progressFill} ${s.progressRed}`;
  if (risk === "Medium") return `${s.progressFill} ${s.progressAmber}`;
  return `${s.progressFill} ${s.progressGreen}`;
}
function badgeClass(risk: string, s: typeof styles) {
  if (risk === "Critical") return `${s.badge} ${s.badgeCritical}`;
  if (risk === "High")     return `${s.badge} ${s.badgeHigh}`;
  if (risk === "Medium")   return `${s.badge} ${s.badgeMedium}`;
  return `${s.badge} ${s.badgeLow}`;
}
function statusBadgeClass(status: string, s: typeof styles) {
  if (status === "Delayed")     return `${s.statusBadge} ${s.statusDelayed}`;
  if (status === "On Schedule") return `${s.statusBadge} ${s.statusOnSchedule}`;
  if (status === "Completed")   return `${s.statusBadge} ${s.statusCompleted}`;
  return `${s.statusBadge} ${s.statusInProgress}`;
}

/* ── Top Bar ──────────────────────────────────────── */
function TopBar({ mapMode }: { mapMode: boolean }) {
  return (
    <div className={styles.topBar}>
      <div className={styles.breadcrumb}>
        Project Registry / <span>Programs and Activities{mapMode ? " / Map View" : ""}</span>
      </div>
      <TopRight />
    </div>
  );
}

interface ProjectData {
  id: string;
  name: string;
  municipality: string;
  progress: number;
  budget: string;
  risk: string;
  status: string;
  inspector: string;
  lat: number;
  lng: number;
}

/* ── Page ──────────────────────────────────────────── */
export default function ProjectsPage() {
  const [projects,   setProjects]   = useState<ProjectData[]>([]);
  const [mapMode,    setMapMode]    = useState(false);
  const [search,     setSearch]     = useState("");
  const [status,     setStatus]     = useState("All");
  const [municipality, setMun]      = useState("");
  const [activePin,  setActivePin]  = useState<string | null>(null);
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetchProjects().then(data => setProjects(data));
  }, []);

  const filtered = projects.filter(p => {
    const q = search.toLowerCase();
    const matchSearch = !search || p.name.toLowerCase().includes(q) || p.id.toLowerCase().includes(q);
    const matchStatus = status === "All" || !status || p.status.toLowerCase().includes(status.toLowerCase()) || p.risk.toLowerCase().includes(status.toLowerCase());
    const matchMun    = !municipality || p.municipality.toLowerCase().includes(municipality.toLowerCase());
    return matchSearch && matchStatus && matchMun;
  });

  function handlePinClick(id: string) {
    setActivePin(prev => prev === id ? null : id);
    // Scroll list item into view
    setTimeout(() => {
      const el = listRef.current?.querySelector(`[data-id="${id}"]`) as HTMLElement;
      el?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }, 100);
  }

  /* ── Shared header + filters ─────────────────── */
  const renderPageHeader = () => (
    <div className={styles.pageHeader}>
      <div className={styles.pageTitleGroup}>
        <h1 className={styles.pageTitle}>
          <span className={styles.pageTitleBlue}>Projects</span>,{" "}
          <span className={styles.pageTitleCyan}>Programs</span>, and Activities
        </h1>
        <p className={styles.pageSub}>Click any row for AI analysis</p>
      </div>
      <div className={styles.actionBtns}>
        <button className={styles.btnAction} onClick={() => setMapMode(m => !m)}>
          {mapMode ? (
            <>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M9 21V9"/>
              </svg>
              Toggle Table View
            </>
          ) : (
            <>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 10c0 7-9 13-9 13S3 17 3 10a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/>
              </svg>
              Toggle Map View
            </>
          )}
        </button>
        <button className={styles.btnAction}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>
          </svg>
          Add new PPA
        </button>
        <button className={`${styles.btnAction} ${styles.btnActionPrimary}`}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
            <polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>
          </svg>
          Import Data
        </button>
      </div>
    </div>
  );

  const renderFilters = () => (
    <div className={styles.filterRow}>
      <div className={styles.filterGroup}>
        <label className={styles.filterLabel}>Search</label>
        <input className={styles.filterInput} placeholder="Search Projects ..." value={search} onChange={e => setSearch(e.target.value)} />
      </div>
      <div className={styles.filterGroup} style={{ flex: 1.5 }}>
        <label className={styles.filterLabel}>Status</label>
        <div className={styles.statusPills}>
          {["All", "In Progress", "On Schedule", "Completed", "Delayed"].map(s => (
            <button
              key={s}
              className={`${styles.statusPill} ${status === s ? styles.statusPillActive : ""}`}
              onClick={() => setStatus(s)}
            >
              {s}
            </button>
          ))}
        </div>
      </div>
      <div className={styles.filterGroup}>
        <label className={styles.filterLabel}>Municipality</label>
        <input className={styles.filterInput} placeholder="Search Projects ..." value={municipality} onChange={e => setMun(e.target.value)} />
      </div>
      <button className={styles.clearBtn} onClick={() => { setSearch(""); setStatus(""); setMun(""); }}>
        Clear Filters
      </button>
    </div>
  );

  /* ═══ TABLE VIEW ═══ */
  if (!mapMode) return (
    <div className={styles.layout}>
      <Sidebar />
      <div className={styles.main}>
        <TopBar mapMode={false} />
        <div className={styles.body}>
          <div className={styles.contentCard}>
            {renderPageHeader()}
            {renderFilters()}
            <div className={styles.sortRow}>
              <div className={styles.sortLeft}>
                <button className={styles.sortBtn}>Sort by: <strong>Most Recent</strong> ▾</button>
                <button className={styles.sortBtn}>Order by: <strong>Ascending</strong> ▾</button>
              </div>
              <div className={styles.countBadge}>{filtered.length} PPAs Found</div>
            </div>
            <div className={styles.tableWrap}>
              <table className={styles.table}>
                <thead className={styles.thead}>
                  <tr>
                    {["Project","Municipality","Progress","Budget","Risk","Inspector"].map(h => (
                      <th key={h} className={styles.th}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {filtered.map(p => (
                    <tr key={p.id} className={styles.tr}>
                      <td className={styles.td}>
                        <div className={styles.projName}>{p.name}</div>
                        <div className={styles.projId}>{p.id}</div>
                        <div className={styles.progressBar}>
                          <div className={riskBarClass(p.risk, styles)} style={{ width: `${p.progress}%` }} />
                        </div>
                      </td>
                      <td className={styles.td}>{p.municipality}</td>
                      <td className={styles.td}>{p.progress}%</td>
                      <td className={styles.td}>{p.budget}</td>
                      <td className={styles.td}><span className={badgeClass(p.risk, styles)}>{p.risk}</span></td>
                      <td className={styles.td}>{p.inspector}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  /* ═══ MAP VIEW ═══ */
  const mapProjects: ProjectPin[] = filtered.map(p => ({
    id: p.id, name: p.name, municipality: p.municipality,
    status: p.status, risk: p.risk, lat: p.lat, lng: p.lng,
  }));

  return (
    <div className={styles.layout}>
      <Sidebar />
      <div className={styles.main}>
        <TopBar mapMode={true} />
        <div className={styles.bodyMap}>
          <div className={styles.contentCard} style={{ display: "flex", flexDirection: "column", flex: 1, minHeight: 0 }}>
            {renderPageHeader()}
            {renderFilters()}
            <div className={styles.divider} />

            {/* ── Map area ──────────────────────── */}
            <div className={styles.mapArea}>

              {/* Left list panel */}
              <div className={styles.mapList}>
                <div className={styles.mapListHeader}>
                  <div className={styles.mapListHeaderIcon}>
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2">
                      <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>
                    </svg>
                  </div>
                  <span className={styles.mapListHeaderTitle}>{filtered.length} PPAs Found</span>
                </div>

                <div className={styles.mapListScroll} ref={listRef}>
                  {filtered.map(p => (
                    <div
                      key={p.id}
                      data-id={p.id}
                      className={`${styles.mapListItem} ${activePin === p.id ? styles.mapListItemActive : ""}`}
                      onClick={() => handlePinClick(p.id)}
                    >
                      <div className={styles.mapListItemName}>{p.name}</div>
                      <div className={styles.mapListItemMeta}>
                        <div className={styles.mapListItemLoc}>
                          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="#9aaabb" strokeWidth="2.5">
                            <path d="M21 10c0 7-9 13-9 13S3 17 3 10a9 9 0 0 1 18 0z"/>
                            <circle cx="12" cy="10" r="3"/>
                          </svg>
                          {p.municipality}
                        </div>
                        <span className={statusBadgeClass(p.status, styles)}>{p.status}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Right: Leaflet Map */}
              <div className={styles.mapContainer}>
                <IloiloMap
                  projects={mapProjects}
                  activeId={activePin}
                  onPinClick={handlePinClick}
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
