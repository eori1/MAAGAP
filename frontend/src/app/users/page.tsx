"use client";

import { useState, useMemo, useEffect } from "react";
import Sidebar from "@/components/Sidebar";
import TopRight from "@/components/TopRight";
import styles from "./page.module.css";

/* ─── Types (mirror backend inspectors.json payload) ──── */
interface InspectorRecord {
  id: string;
  name: string;
  email: string;
  position: string;
  role: string;
  status: "Active" | "On Duty";
  vehicleAccess: boolean;
  capacity: number;
  assigned: number;
}

const STATUS_STYLE: Record<string, { bg: string; color: string }> = {
  "Active":  { bg: "#d4efdf", color: "#27ae60" },
  "On Duty": { bg: "#fcf3cf", color: "#f39c12" },
};

/* ─── Icons ───────────────────────────────────────────── */
function AvatarIcon() {
  return (
    <svg width="40" height="40" viewBox="0 0 40 40" fill="none">
      <circle cx="20" cy="20" r="19" fill="#03a9f4" />
      <circle cx="20" cy="14" r="5" fill="#fff" />
      <path d="M10 32a10 10 0 0 1 20 0" stroke="#fff" strokeWidth="2.5" strokeLinecap="round" />
    </svg>
  );
}

/* ─── Page Component ──────────────────────────────────── */
export default function UserManagementPage() {
  const [inspectors, setInspectors] = useState<InspectorRecord[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState("All Status");
  const [sortBy, setSortBy] = useState("Alphabetical (A-Z)");
  const [orderBy, setOrderBy] = useState("Ascending");

  useEffect(() => {
    fetch("/api/inspectors")
      .then(res => {
        if (!res.ok) throw new Error("no data");
        return res.json();
      })
      .then(setInspectors)
      .catch(() => setLoadError("No inspector roster found. Run the backend pipeline (python main.py) to generate it."));
  }, []);

  const filtered = useMemo(() => {
    let list = [...inspectors];
    if (search) list = list.filter(u => u.name.toLowerCase().includes(search.toLowerCase()) || u.id.toLowerCase().includes(search.toLowerCase()));
    if (status !== "All Status") list = list.filter(u => u.status === status);
    const dir = orderBy === "Ascending" ? 1 : -1;
    list.sort((a, b) => sortBy === "Most Assigned"
      ? (a.assigned - b.assigned) * -dir
      : a.name.localeCompare(b.name) * dir);
    return list;
  }, [inspectors, search, status, sortBy, orderBy]);

  const clearFilters = () => { setSearch(""); setStatus("All Status"); };

  return (
    <div className={styles.shell}>
      <Sidebar />
      <div className={styles.main}>

        {/* ── Top bar ── */}
        <div className={styles.topbar}>
          <div className={styles.breadcrumb}>
            <span>Admin</span>
            <span className={styles.sep}>/</span>
            <span className={styles.breadActive}>User Management</span>
          </div>
          <TopRight />
        </div>

        {/* ── Main white card ── */}
        <div className={styles.card}>

          {/* Card header */}
          <div className={styles.cardHeader}>
            <div>
              <h1 className={styles.cardTitle}>
                Inspector <span className={styles.accent}>Roster</span>
              </h1>
              <p className={styles.cardSub}>PPDO field inspectors and LP-computed visit capacity ({filtered.length} of {inspectors.length})</p>
            </div>
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
                  placeholder="Search by name ..."
                  value={search}
                  onChange={e => setSearch(e.target.value)}
                />
              </div>
            </div>

            <div className={styles.filterGroup}>
              <label className={styles.filterLabel}>Status</label>
              <div className={styles.selectWrap}>
                <select className={styles.select} value={status} onChange={e => setStatus(e.target.value)}>
                  <option>All Status</option><option>Active</option><option>On Duty</option>
                </select>
                <svg className={styles.selectChevron} width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><polyline points="6 9 12 15 18 9"/></svg>
              </div>
            </div>

            <button className={styles.clearBtn} onClick={clearFilters}>Clear Filters</button>
          </div>

          {/* Sort row */}
          <div className={styles.sortRow}>
            <div className={styles.sortControls}>
              <span className={styles.sortLabel}>Sort by:</span>
              <select className={styles.sortSelect} value={sortBy} onChange={e => setSortBy(e.target.value)}>
                <option>Alphabetical (A-Z)</option><option>Most Assigned</option>
              </select>
              <span className={styles.sortLabel} style={{ marginLeft: "1rem" }}>Order by:</span>
              <select className={styles.sortSelect} value={orderBy} onChange={e => setOrderBy(e.target.value)}>
                <option>Ascending</option><option>Descending</option>
              </select>
            </div>
          </div>

          {/* Table */}
          <div className={styles.tableWrap}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th className={styles.th}>Inspector</th>
                  <th className={styles.th}>Position</th>
                  <th className={styles.th} style={{ textAlign: "center" }}>Vehicle</th>
                  <th className={styles.th} style={{ textAlign: "center" }}>Assigned / Capacity</th>
                  <th className={styles.th} style={{ textAlign: "center" }}>Status</th>
                </tr>
              </thead>
              <tbody>
                {loadError && (
                  <tr><td colSpan={5} className={styles.emptyRow}>{loadError}</td></tr>
                )}

                {!loadError && filtered.map(u => {
                  const st = STATUS_STYLE[u.status];
                  const utilPct = u.capacity > 0 ? Math.min(100, (u.assigned / u.capacity) * 100) : 0;
                  return (
                    <tr key={u.id} className={styles.row}>
                      <td className={styles.td}>
                        <div className={styles.employeeCell}>
                          <AvatarIcon />
                          <div>
                            <div className={styles.employeeName}>{u.name}</div>
                            <div className={styles.employeeId}>{u.id} / {u.email}</div>
                          </div>
                        </div>
                      </td>
                      <td className={styles.td}>{u.position}</td>
                      <td className={`${styles.td} ${styles.tdCenter}`}>{u.vehicleAccess ? "Yes" : "No"}</td>
                      <td className={`${styles.td} ${styles.tdCenter}`}>
                        {u.assigned} / {u.capacity}
                        <div className={styles.progressBarBg} style={{ marginTop: 4 }}>
                          <div className={styles.progressBar} style={{ width: `${utilPct}%` }} />
                        </div>
                      </td>
                      <td className={`${styles.td} ${styles.tdCenter}`}>
                        <span className={styles.statusBadge} style={{ background: st.bg, color: st.color }}>
                          {u.status}
                        </span>
                      </td>
                    </tr>
                  );
                })}

                {!loadError && filtered.length === 0 && (
                  <tr><td colSpan={5} className={styles.emptyRow}>No inspectors match your filters.</td></tr>
                )}
              </tbody>
            </table>
          </div>

        </div>
      </div>
    </div>
  );
}
