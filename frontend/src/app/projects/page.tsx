"use client";

import { useState, useRef, useEffect, useMemo } from "react";
import dynamic from "next/dynamic";
import Sidebar from "@/components/Sidebar";
import TopRight from "@/components/TopRight";
import Skeleton from "@/components/ui/Skeleton";
import EmptyState from "@/components/ui/EmptyState";
import Badge from "@/components/ui/Badge";
import ProgressBar from "@/components/ui/ProgressBar";
import AddPpaModal from "@/components/AddPpaModal";
import ImportPpaModal from "@/components/ImportPpaModal";
import { RISK_TONE } from "@/lib/riskTone";
import styles from "./page.module.css";
import type { ProjectPin } from "@/components/IloiloMap";
import { fetchProjects } from "@/lib/api";

const IloiloMap = dynamic(() => import("@/components/IloiloMap"), {
  ssr: false,
  loading: () => (
    <div style={{
      width: "100%", height: "100%",
      display: "flex", alignItems: "center", justifyContent: "center",
      background: "var(--surface-sunken)",
      color: "var(--ink-500)", fontWeight: 700, fontSize: "1rem",
    }}>
      Loading map…
    </div>
  ),
});

const STATUS_TONE: Record<string, "good" | "warning" | "serious" | "neutral"> = {
  Delayed: "serious",
  "On Schedule": "good",
  Completed: "good",
  "In Progress": "neutral",
};

type SortKey = "name" | "municipality" | "progress" | "risk" | "budget";
const RISK_ORDER: Record<string, number> = { Pending: -1, Low: 0, Medium: 1, High: 2, Critical: 3 };
const RISK_LABEL: Record<string, string> = { Pending: "Pending Assessment" };

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

function SortArrow({ active, asc }: { active: boolean; asc: boolean }) {
  if (!active) return null;
  return <span className={styles.sortArrow}>{asc ? "↑" : "↓"}</span>;
}

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

export default function ProjectsPage() {
  const [projects, setProjects] = useState<ProjectData[]>([]);
  const [loading, setLoading] = useState(true);
  const [mapMode, setMapMode] = useState(false);
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState("All");
  const [riskFilter, setRiskFilter] = useState("All");
  const [municipality, setMun] = useState("");
  const [activePin, setActivePin] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>("name");
  const [sortAsc, setSortAsc] = useState(true);
  const [viewerRole, setViewerRole] = useState<"manager" | "inspector" | "admin" | null>(null);
  const [showAddPpa, setShowAddPpa] = useState(false);
  const [showImportPpa, setShowImportPpa] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);

  const loadProjects = () => {
    fetchProjects().then((data) => {
      setProjects(data);
      setLoading(false);
    });
  };

  useEffect(() => {
    loadProjects();
    fetch("/api/me")
      .then((res) => (res.ok ? res.json() : null))
      .then((profile) => setViewerRole(profile?.role ?? null))
      .catch(() => setViewerRole(null));
  }, []);

  const canAddPpa = viewerRole === "manager" || viewerRole === "admin";

  const filtered = useMemo(() => {
    const list = projects.filter((p) => {
      const q = search.toLowerCase();
      const matchSearch = !search || p.name.toLowerCase().includes(q) || p.id.toLowerCase().includes(q);
      const matchStatus = status === "All" || !status || p.status.toLowerCase().includes(status.toLowerCase());
      const matchRisk = riskFilter === "All" || p.risk === riskFilter;
      const matchMun = !municipality || p.municipality.toLowerCase().includes(municipality.toLowerCase());
      return matchSearch && matchStatus && matchRisk && matchMun;
    });

    const dir = sortAsc ? 1 : -1;
    list.sort((a, b) => {
      if (sortKey === "progress") return (a.progress - b.progress) * dir;
      if (sortKey === "risk") return ((RISK_ORDER[a.risk] ?? 0) - (RISK_ORDER[b.risk] ?? 0)) * dir;
      if (sortKey === "municipality") return a.municipality.localeCompare(b.municipality) * dir;
      if (sortKey === "budget") return (parseFloat(a.budget.replace(/[^0-9.]/g, "")) - parseFloat(b.budget.replace(/[^0-9.]/g, ""))) * dir;
      return a.name.localeCompare(b.name) * dir;
    });
    return list;
  }, [projects, search, status, riskFilter, municipality, sortKey, sortAsc]);

  const criticalShown = filtered.filter((p) => p.risk === "Critical").length;

  function handleSort(key: SortKey) {
    if (sortKey === key) setSortAsc((a) => !a);
    else { setSortKey(key); setSortAsc(true); }
  }

  function sortHeaderProps(key: SortKey) {
    return {
      className: `${styles.th} ${sortKey === key ? styles.thSorted : ""}`,
      onClick: () => handleSort(key),
    };
  }

  function handlePinClick(id: string) {
    setActivePin((prev) => (prev === id ? null : id));
    setTimeout(() => {
      const el = listRef.current?.querySelector(`[data-id="${id}"]`) as HTMLElement;
      el?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }, 100);
  }

  const clearFilters = () => { setSearch(""); setStatus("All"); setRiskFilter("All"); setMun(""); };

  /* ── Shared header + filters ─────────────────── */
  const renderPageHeader = () => (
    <div className={styles.pageHeader}>
      <div>
        <h1 className={styles.pageTitle}>
          <span className={styles.pageTitleAccent}>Projects</span>, Programs, and Activities
        </h1>
        <p className={styles.pageSub}>Browse and search the full monitored cohort</p>
      </div>
      <div className={styles.headerActions}>
        {canAddPpa && (
          <>
            <button className={styles.importBtn} onClick={() => setShowImportPpa(true)}>
              Import Data
            </button>
            <button className={styles.addPpaBtn} onClick={() => setShowAddPpa(true)}>
              + Add new PPA
            </button>
          </>
        )}
        <div className={styles.viewToggle}>
          <button className={`${styles.viewToggleBtn} ${!mapMode ? styles.viewToggleBtnActive : ""}`} onClick={() => setMapMode(false)}>Table</button>
          <button className={`${styles.viewToggleBtn} ${mapMode ? styles.viewToggleBtnActive : ""}`} onClick={() => setMapMode(true)}>Map</button>
        </div>
      </div>
    </div>
  );

  const renderFilters = () => (
    <div className={styles.filterRow}>
      <div className={styles.filterGroup}>
        <label className={styles.filterLabel}>Search</label>
        <input className={styles.filterInput} placeholder="Search Projects ..." value={search} onChange={(e) => setSearch(e.target.value)} />
      </div>
      <div className={styles.filterGroup} style={{ flex: 1.5 }}>
        <label className={styles.filterLabel}>Status</label>
        <div className={styles.statusPills}>
          {["All", "In Progress", "On Schedule", "Completed", "Delayed"].map((s) => (
            <button key={s} className={`${styles.statusPill} ${status === s ? styles.statusPillActive : ""}`} onClick={() => setStatus(s)}>
              {s}
            </button>
          ))}
        </div>
      </div>
      <div className={styles.filterGroup}>
        <label className={styles.filterLabel}>Risk Tier</label>
        <select className={styles.filterSelect} value={riskFilter} onChange={(e) => setRiskFilter(e.target.value)}>
          {["All", "Pending", "Low", "Medium", "High", "Critical"].map((r) => <option key={r} value={r}>{RISK_LABEL[r] ?? r}</option>)}
        </select>
      </div>
      <div className={styles.filterGroup}>
        <label className={styles.filterLabel}>Municipality</label>
        <input className={styles.filterInput} placeholder="Search Municipality ..." value={municipality} onChange={(e) => setMun(e.target.value)} />
      </div>
      <button className={styles.clearBtn} onClick={clearFilters}>Clear Filters</button>
    </div>
  );

  const resultSummary = (
    <span className={styles.resultSummary}>
      <strong>{filtered.length}</strong> of {projects.length} shown{criticalShown > 0 ? ` · ${criticalShown} Critical among them` : ""}
    </span>
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
            <div className={styles.sortRow}>{resultSummary}</div>
            <div className={styles.tableWrap}>
              <table className={styles.table}>
                <thead className={styles.thead}>
                  <tr>
                    <th {...sortHeaderProps("name")}>Project <SortArrow active={sortKey === "name"} asc={sortAsc} /></th>
                    <th {...sortHeaderProps("municipality")}>Municipality <SortArrow active={sortKey === "municipality"} asc={sortAsc} /></th>
                    <th {...sortHeaderProps("progress")}>Progress <SortArrow active={sortKey === "progress"} asc={sortAsc} /></th>
                    <th {...sortHeaderProps("budget")}>Budget <SortArrow active={sortKey === "budget"} asc={sortAsc} /></th>
                    <th {...sortHeaderProps("risk")}>Risk <SortArrow active={sortKey === "risk"} asc={sortAsc} /></th>
                    <th className={styles.th}>Inspector</th>
                  </tr>
                </thead>
                <tbody>
                  {loading && Array.from({ length: 6 }).map((_, i) => (
                    <tr key={i} className={styles.tr}>
                      {Array.from({ length: 6 }).map((__, j) => (
                        <td key={j} className={styles.td}><Skeleton height="1rem" /></td>
                      ))}
                    </tr>
                  ))}

                  {!loading && filtered.map((p) => (
                    <tr key={p.id} className={styles.tr}>
                      <td className={styles.td}>
                        <div className={styles.projName}>{p.name}</div>
                        <div className={styles.projId}>{p.id}</div>
                        <ProgressBar value={p.progress} tone={RISK_TONE[p.risk] ?? "accent"} />
                      </td>
                      <td className={styles.td}>{p.municipality}</td>
                      <td className={styles.td}>{p.risk === "Pending" ? "—" : `${p.progress}%`}</td>
                      <td className={styles.td}>{p.budget}</td>
                      <td className={styles.td}><Badge tone={RISK_TONE[p.risk] ?? "neutral"}>{RISK_LABEL[p.risk] ?? p.risk}</Badge></td>
                      <td className={styles.td}>{p.inspector}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {!loading && filtered.length === 0 && (
                <EmptyState title="No projects match your filters" message="Try clearing filters or searching a different term." />
              )}
            </div>
          </div>
        </div>
      </div>
      {showAddPpa && (
        <AddPpaModal onClose={() => setShowAddPpa(false)} onCreated={() => { setShowAddPpa(false); loadProjects(); }} />
      )}
      {showImportPpa && (
        <ImportPpaModal onClose={() => setShowImportPpa(false)} onCreated={loadProjects} />
      )}
    </div>
  );

  /* ═══ MAP VIEW ═══ */
  const mapProjects: ProjectPin[] = filtered.map((p) => ({
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

            <div className={styles.mapArea}>
              <div className={styles.mapList}>
                <div className={styles.mapListHeader}>
                  <span className={styles.mapListHeaderTitle}>{filtered.length} PPAs Found</span>
                </div>

                <div className={styles.mapListScroll} ref={listRef}>
                  {filtered.map((p) => (
                    <div
                      key={p.id}
                      data-id={p.id}
                      className={`${styles.mapListItem} ${activePin === p.id ? styles.mapListItemActive : ""}`}
                      onClick={() => handlePinClick(p.id)}
                    >
                      <div className={styles.mapListItemName}>{p.name}</div>
                      <div className={styles.mapListItemMeta}>
                        <div className={styles.mapListItemLoc}>
                          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                            <path d="M21 10c0 7-9 13-9 13S3 17 3 10a9 9 0 0 1 18 0z" />
                            <circle cx="12" cy="10" r="3" />
                          </svg>
                          {p.municipality}
                        </div>
                        <Badge tone={STATUS_TONE[p.status] ?? "neutral"}>{p.status}</Badge>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

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
      {showAddPpa && (
        <AddPpaModal onClose={() => setShowAddPpa(false)} onCreated={() => { setShowAddPpa(false); loadProjects(); }} />
      )}
      {showImportPpa && (
        <ImportPpaModal onClose={() => setShowImportPpa(false)} onCreated={loadProjects} />
      )}
    </div>
  );
}
