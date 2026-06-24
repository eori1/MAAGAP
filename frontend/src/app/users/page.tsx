"use client";

import { useState, useMemo } from "react";
import Sidebar from "@/components/Sidebar";
import TopRight from "@/components/TopRight";
import styles from "./page.module.css";

/* ─── Mock User Data ──────────────────────────────────── */
const USERS = [
  { id: "EMP - 321", name: "Juan de la Cruz",   email: "jdlc@iloilo.gov.ph",      position: "Civil Engineer III",               role: "Inspector", assigned: 12, status: "Active" },
  { id: "EMP - 322", name: "Maria Santos",      email: "msantos@iloilo.gov.ph",   position: "Planning Officer II",              role: "Inspector", assigned: 13, status: "Active" },
  { id: "EMP - 323", name: "Jose Rizal",        email: "jrizar@iloilo.gov.ph",    position: "Urban Planner I",                  role: "Inspector", assigned: 14, status: "Active" },
  { id: "EMP - 324", name: "Liza Soberano",     email: "lsoberano@iloilo.gov.ph", position: "Environmental Analyst II",         role: "Inspector", assigned: 15, status: "Active" },
  { id: "EMP - 325", name: "Anthony Gonzales",  email: "agonzales@iloilo.gov.ph", position: "Urban Planner III",                role: "Inspector", assigned: 16, status: "Active" },
  { id: "EMP - 326", name: "Carmen Reyes",      email: "creyes@iloilo.gov.ph",    position: "Regional Development Officer",     role: "Auditor",   assigned: 17, status: "Active" },
  { id: "EMP - 327", name: "Miguel Alonzo",     email: "malonzo@iloilo.gov.ph",   position: "Building Code Compliance Officer", role: "Inspector", assigned: 18, status: "Active" },
  { id: "EMP - 328", name: "Veronica Cruz",     email: "vcruz@iloilo.gov.ph",     position: "Transportation Planning Associate",role: "Analyst",   assigned: 19, status: "Active" },
  { id: "EMP - 329", name: "Daniel Padilla",    email: "dpadilla@iloilo.gov.ph",  position: "Development Coordinator",          role: "Inspector", assigned: 20, status: "On Leave" },
  { id: "EMP - 330", name: "Julia Barretto",    email: "jbarretto@iloilo.gov.ph", position: "Land Use Planner",                 role: "Inspector", assigned: 21, status: "Inactive" },
];

const STATUS_STYLE: Record<string, { bg: string; color: string }> = {
  "Active":   { bg: "#d4efdf", color: "#27ae60" },
  "On Leave": { bg: "#fcf3cf", color: "#f39c12" },
  "Inactive": { bg: "#e2e8f0", color: "#94a3b8" },
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

function PlusIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
      <line x1="12" y1="5" x2="12" y2="19" />
      <line x1="5" y1="12" x2="19" y2="12" />
    </svg>
  );
}

/* ─── Page Component ──────────────────────────────────── */
export default function UserManagementPage() {
  const [search, setSearch] = useState("");
  const [role, setRole] = useState("All Roles");
  const [dept, setDept] = useState("All Departments");
  const [sortBy, setSortBy] = useState("Alphabetical (A-Z)");
  const [orderBy, setOrderBy] = useState("Ascending");

  const filtered = useMemo(() => {
    let list = [...USERS];
    if (search) list = list.filter(u => u.name.toLowerCase().includes(search.toLowerCase()));
    if (role !== "All Roles") list = list.filter(u => u.role === role);
    return list;
  }, [search, role]);

  const clearFilters = () => { setSearch(""); setRole("All Roles"); setDept("All Departments"); };

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
                Role <span className={styles.accent}>Management</span>
              </h1>
              <p className={styles.cardSub}>Manage user access and permissions</p>
            </div>
            <button className={styles.addBtn}>
              <PlusIcon /> Add Employee
            </button>
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
              <label className={styles.filterLabel}>Roles</label>
              <div className={styles.selectWrap}>
                <select className={styles.select} value={role} onChange={e => setRole(e.target.value)}>
                  <option>All Roles</option><option>Inspector</option><option>Auditor</option><option>Analyst</option>
                </select>
                <svg className={styles.selectChevron} width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><polyline points="6 9 12 15 18 9"/></svg>
              </div>
            </div>

            <div className={styles.filterGroup}>
              <label className={styles.filterLabel}>Departments</label>
              <div className={styles.selectWrap}>
                <select className={styles.select} value={dept} onChange={e => setDept(e.target.value)}>
                  <option>All Departments</option><option>Engineering</option><option>Planning</option>
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
                <option>Alphabetical (A-Z)</option><option>Most Recent</option>
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
                  <th className={styles.th}>Employee</th>
                  <th className={styles.th}>Position</th>
                  <th className={styles.th}>Role</th>
                  <th className={styles.th} style={{ textAlign: "center" }}>Projects Assigned</th>
                  <th className={styles.th} style={{ textAlign: "center" }}>Status</th>
                  <th className={styles.th} style={{ textAlign: "center" }}>Assign</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map(u => {
                  const st = STATUS_STYLE[u.status];
                  const isActive = u.status === "Active";
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
                      <td className={styles.td}>{u.role}</td>
                      <td className={`${styles.td} ${styles.tdCenter}`}>{u.assigned}</td>
                      <td className={`${styles.td} ${styles.tdCenter}`}>
                        <span className={styles.statusBadge} style={{ background: st.bg, color: st.color }}>
                          {u.status}
                        </span>
                      </td>
                      <td className={`${styles.td} ${styles.tdCenter}`}>
                        <button className={`${styles.assignBtn} ${!isActive ? styles.assignBtnDisabled : ""}`}>
                          Assign
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

        </div>
      </div>
    </div>
  );
}
