"use client";

import { useState } from "react";
import Sidebar from "@/components/Sidebar";
import styles from "./page.module.css";

/* ─── Mock Data ─────────────────────────────────── */
const ALL_PROJECTS = [
  { id: "PPS-124", name: "Aganan Flyover, Brgy. Aganan",              municipality: "Pavia",        progress: 30, budget: "440,000,000",   risk: "High",     status: "Delayed",     inspector: "Engr. Rico Cruz",    pin: { top: "38%", left: "38%" } },
  { id: "PPS-125", name: "Bamboo Flyover, Brgy. Bamboo",              municipality: "Iloilo City",  progress: 65, budget: "550,000,000",   risk: "Medium",   status: "On Schedule", inspector: "Engr. Maria Santos", pin: { top: "55%", left: "48%" } },
  { id: "PPS-126", name: "Coco Flyover, Brgy. Coco",                  municipality: "San Miguel",   progress: 20, budget: "320,000,000",   risk: "High",     status: "Delayed",     inspector: "Engr. Juan Dela Cruz", pin: { top: "30%", left: "60%" } },
  { id: "PPS-127", name: "Dahlia Flyover, Brgy. Dahlia",              municipality: "Leganes",      progress: 100, budget: "600,000,000",  risk: "Low",      status: "Completed",   inspector: "Engr. Liza Tan",     pin: { top: "62%", left: "35%" } },
  { id: "PPS-128", name: "Eucalyptus Flyover, Brgy. Eucalyptus",      municipality: "Balasan",      progress: 55, budget: "700,000,000",   risk: "Medium",   status: "On Schedule", inspector: "Engr. Mark Velasco", pin: { top: "22%", left: "72%" } },
  { id: "PPS-129", name: "Fern Flyover, Brgy. Fern",                  municipality: "Cabatuan",     progress: 15, budget: "280,000,000",   risk: "High",     status: "Delayed",     inspector: "Engr. Ana Reyes",    pin: { top: "48%", left: "25%" } },
  { id: "PPS-130", name: "Ginkgo Flyover, Brgy. Ginkgo",              municipality: "Jordan",       progress: 100, budget: "380,000,000",  risk: "Low",      status: "Completed",   inspector: "Engr. Carlo Basa",   pin: { top: "72%", left: "55%" } },
  { id: "PPS-131", name: "Maple Bridge, Brgy. Maple",                 municipality: "Samantha",     progress: 40, budget: "820,000,000",   risk: "Critical", status: "In Progress", inspector: "Engr. Rey Osorio",   pin: { top: "42%", left: "68%" } },
  { id: "PPS-132", name: "Oak Avenue, Brgy. Oak",                     municipality: "Iloilo City",  progress: 55, budget: "1,200,000,000", risk: "Medium",   status: "On Schedule", inspector: "Engr. Petra Quinto", pin: { top: "58%", left: "78%" } },
  { id: "PPS-133", name: "Pine Road, Brgy. Pine",                     municipality: "Molo",         progress: 78, budget: "460,000,000",   risk: "Low",      status: "On Schedule", inspector: "Engr. Joel Mira",    pin: { top: "34%", left: "82%" } },
  { id: "PPS-134", name: "Quince Bridge, Brgy. Quince",               municipality: "La Paz",       progress: 35, budget: "520,000,000",   risk: "High",     status: "Delayed",     inspector: "Engr. Susan Go",     pin: { top: "65%", left: "88%" } },
  { id: "PPS-135", name: "Rose Bridge, Brgy. Rose",                   municipality: "Tigbauan",     progress: 88, budget: "340,000,000",   risk: "Low",      status: "Completed",   inspector: "Engr. Paul Dy",      pin: { top: "78%", left: "42%" } },
];

/* ─── Pin SVG by status ─────────────────────────── */
const PIN_COLORS: Record<string, string> = {
  "Delayed":     "#e74c3c",
  "On Schedule": "#27ae60",
  "Completed":   "#1264ae",
  "In Progress": "#f39c12",
};

function MapPin({ color, delay = 0 }: { color: string; delay?: number }) {
  return (
    <svg width="24" height="32" viewBox="0 0 24 32" fill="none" style={{ animationDelay: `${delay}ms` }}>
      <path d="M12 0C5.373 0 0 5.373 0 12c0 9 12 20 12 20S24 21 24 12C24 5.373 18.627 0 12 0z" fill={color} />
      <circle cx="12" cy="12" r="5" fill="white" fillOpacity="0.9" />
    </svg>
  );
}

/* ─── Badge helpers ─────────────────────────────── */
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
function statusClass(status: string, s: typeof styles) {
  if (status === "Delayed")     return `${s.statusBadge} ${s.statusDelayed}`;
  if (status === "On Schedule") return `${s.statusBadge} ${s.statusOnSchedule}`;
  if (status === "Completed")   return `${s.statusBadge} ${s.statusCompleted}`;
  return `${s.statusBadge} ${s.statusInProgress}`;
}

/* ─── Common TopBar ─────────────────────────────── */
function TopBar({ mapMode }: { mapMode: boolean }) {
  return (
    <div className={styles.topBar}>
      <div className={styles.breadcrumb}>
        Project Registry / <span>Programs and Activities{mapMode ? " / Map View" : ""}</span>
      </div>
      <div className={styles.topRight}>
        <div className={styles.datePill}>
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="4" width="18" height="18" rx="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>
          May 15, 2026
        </div>
        <button className={styles.iconBtn}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.73 21a2 2 0 0 1-3.46 0"/></svg>
        </button>
        <button className={styles.iconBtn} style={{ background: "#1264ae", color: "#fff", borderColor: "#1264ae" }}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
        </button>
      </div>
    </div>
  );
}

/* ─── Page ───────────────────────────────────────── */
export default function ProjectsPage() {
  const [mapMode, setMapMode]     = useState(false);
  const [search, setSearch]       = useState("");
  const [status, setStatus]       = useState("");
  const [municipality, setMun]    = useState("");
  const [activePin, setActivePin] = useState<string | null>(null);

  const filtered = ALL_PROJECTS.filter(p => {
    const q = search.toLowerCase();
    const matchSearch = !search || p.name.toLowerCase().includes(q) || p.id.toLowerCase().includes(q);
    const matchStatus = !status || p.status.toLowerCase().includes(status.toLowerCase()) || p.risk.toLowerCase().includes(status.toLowerCase());
    const matchMun    = !municipality || p.municipality.toLowerCase().includes(municipality.toLowerCase());
    return matchSearch && matchStatus && matchMun;
  });

  /* shared header block reused in both views */
  const PageHeader = () => (
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
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M9 21V9"/></svg>
              Toggle Table View
            </>
          ) : (
            <>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M9 21V9"/></svg>
              Toggle Map View
            </>
          )}
        </button>
        <button className={styles.btnAction}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
          Add new PPA
        </button>
        <button className={`${styles.btnAction} ${styles.btnActionPrimary}`}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
          Import Data
        </button>
      </div>
    </div>
  );

  const Filters = () => (
    <div className={styles.filterRow}>
      <div className={styles.filterGroup}>
        <label className={styles.filterLabel}>Search</label>
        <input className={styles.filterInput} placeholder="Search Projects ..." value={search} onChange={e => setSearch(e.target.value)} />
      </div>
      <div className={styles.filterGroup}>
        <label className={styles.filterLabel}>Status</label>
        <input className={styles.filterInput} placeholder="Search Projects ..." value={status} onChange={e => setStatus(e.target.value)} />
      </div>
      <div className={styles.filterGroup}>
        <label className={styles.filterLabel}>Municipality</label>
        <input className={styles.filterInput} placeholder="Search Projects ..." value={municipality} onChange={e => setMun(e.target.value)} />
      </div>
      <button className={styles.clearBtn} onClick={() => { setSearch(""); setStatus(""); setMun(""); }}>Clear Filters</button>
    </div>
  );

  /* ═══ TABLE VIEW ═══ */
  if (!mapMode) return (
    <div className={styles.layout}>
      <Sidebar />
      <div className={styles.main}>
        <TopBar mapMode={false} />
        <div className={styles.body}>
          <PageHeader />
          <Filters />
          <div className={styles.sortRow}>
            <div className={styles.sortLeft}>
              <button className={styles.sortBtn}>Sort by: Most Recent ▾</button>
              <button className={styles.sortBtn}>Order by: Ascending ▾</button>
            </div>
            <div className={styles.countBadge}>{filtered.length} PPAs Found</div>
          </div>
          <div className={styles.tableCard}>
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
                    <td className={styles.td}><span className={styles.progressPct}>{p.progress}%</span></td>
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
  );

  /* ═══ MAP VIEW ═══ */
  return (
    <div className={styles.layout}>
      <Sidebar />
      <div className={styles.main}>
        <TopBar mapMode={true} />
        <div className={styles.bodyMap}>
          <PageHeader />
          <Filters />

          {/* Map Area */}
          <div className={styles.mapArea}>
            {/* Left: Project List */}
            <div className={styles.mapList}>
              <div className={styles.mapListHeader}>
                <div className={styles.mapListHeaderIcon}>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>
                </div>
                <span className={styles.mapListHeaderTitle}>{filtered.length} PPAs Found</span>
              </div>
              <div className={styles.mapListScroll}>
                {filtered.map(p => (
                  <div
                    key={p.id}
                    className={`${styles.mapListItem} ${activePin === p.id ? styles.mapListItemActive : ""}`}
                    onClick={() => setActivePin(id => id === p.id ? null : p.id)}
                  >
                    <div className={styles.mapListItemName}>{p.name}</div>
                    <div className={styles.mapListItemMeta}>
                      <div className={styles.mapListItemLoc}>
                        <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="#9aaabb" strokeWidth="2.5"><path d="M21 10c0 7-9 13-9 13S3 17 3 10a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg>
                        {p.municipality}
                      </div>
                      <span className={statusClass(p.status, styles)}>{p.status}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Right: Map Placeholder */}
            <div className={styles.mapPlaceholder}>
              {/* Grid */}
              <div className={styles.mapGrid} />

              {/* Center label */}
              <div className={styles.mapCenterLabel}>
                <div className={styles.mapCenterLabelTitle}>🗺 Iloilo Province — Geospatial View</div>
                <div className={styles.mapCenterLabelSub}>
                  Interactive map will load here.<br />
                  Connect a mapping provider (e.g. Mapbox, Leaflet) to display live data.
                </div>
              </div>

              {/* Scattered pins */}
              <div className={styles.mapPins}>
                {filtered.map((p, i) => (
                  <div
                    key={p.id}
                    className={`${styles.mapPin} ${activePin === p.id ? styles.mapListItemActive : ""}`}
                    style={{ top: p.pin.top, left: p.pin.left, animationDelay: `${i * 60}ms` }}
                    onClick={() => setActivePin(id => id === p.id ? null : p.id)}
                  >
                    <MapPin color={PIN_COLORS[p.status] ?? "#1264ae"} delay={i * 60} />
                  </div>
                ))}
              </div>

              {/* Zoom controls */}
              <div className={styles.zoomControls}>
                <button className={styles.zoomBtn} title="Zoom in">+</button>
                <button className={styles.zoomBtn} title="Zoom out">−</button>
                <button className={styles.zoomBtn} title="Layers" style={{ fontSize: "0.8rem" }}>⊞</button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
