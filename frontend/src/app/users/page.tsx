"use client";

import { useState, useMemo, useEffect } from "react";
import { useRouter } from "next/navigation";
import Sidebar from "@/components/Sidebar";
import TopRight from "@/components/TopRight";
import styles from "./page.module.css";

/* ─── Types ────────────────────────────────────────────── */
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

type AccountRole = "manager" | "inspector" | "admin";

interface AccountRecord {
  id: string;
  email: string;
  full_name: string | null;
  role: AccountRole;
  inspector_id: string | null;
  created_at: string;
}

interface SessionProfile {
  role: AccountRole;
}

const STATUS_STYLE: Record<string, { bg: string; color: string }> = {
  "Active":  { bg: "#d4efdf", color: "#27ae60" },
  "On Duty": { bg: "#fcf3cf", color: "#f39c12" },
};

const ROLE_STYLE: Record<AccountRole, { bg: string; color: string }> = {
  admin:     { bg: "#e8e0fb", color: "#6c3fc5" },
  manager:   { bg: "#d4e0f5", color: "#2756c5" },
  inspector: { bg: "#e2e8f0", color: "#4a5a6a" },
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

/* ─── Create Account Form (Admin only) ───────────────────── */
function CreateAccountForm({ inspectors, onCreated }: { inspectors: InspectorRecord[]; onCreated: () => void }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [role, setRole] = useState<AccountRole>("inspector");
  const [inspectorId, setInspectorId] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    const res = await fetch("/api/admin/users", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        email, password, fullName,
        role,
        inspectorId: role === "inspector" ? (inspectorId || null) : null,
      }),
    });
    setBusy(false);
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      setError(body.error ?? "Failed to create account");
      return;
    }
    setEmail(""); setPassword(""); setFullName(""); setInspectorId("");
    onCreated();
  }

  return (
    <form className={styles.createForm} onSubmit={handleSubmit}>
      {error && <div className={styles.formError}>{error}</div>}
      <div className={styles.createFormRow}>
        <input className={styles.searchInput} placeholder="Full name" value={fullName} onChange={(e) => setFullName(e.target.value)} />
        <input className={styles.searchInput} type="email" placeholder="Email" required value={email} onChange={(e) => setEmail(e.target.value)} />
        <input className={styles.searchInput} type="password" placeholder="Temporary password" required minLength={6} value={password} onChange={(e) => setPassword(e.target.value)} />
        <select className={styles.select} value={role} onChange={(e) => setRole(e.target.value as AccountRole)}>
          <option value="inspector">Inspector</option>
          <option value="manager">Manager</option>
          <option value="admin">Admin</option>
        </select>
        {role === "inspector" && (
          <select className={styles.select} value={inspectorId} onChange={(e) => setInspectorId(e.target.value)}>
            <option value="">Link to roster entry...</option>
            {inspectors.map((i) => (
              <option key={i.id} value={i.id}>{i.name} ({i.id})</option>
            ))}
          </select>
        )}
        <button className={styles.addBtn} type="submit" disabled={busy}>
          <PlusIcon /> {busy ? "Creating..." : "Create Account"}
        </button>
      </div>
    </form>
  );
}

/* ─── Page Component ──────────────────────────────────── */
export default function UserManagementPage() {
  const router = useRouter();
  const [sessionProfile, setSessionProfile] = useState<SessionProfile | null>(null);
  const [inspectors, setInspectors] = useState<InspectorRecord[]>([]);
  const [accounts, setAccounts] = useState<AccountRecord[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState("All Status");
  const [sortBy, setSortBy] = useState("Alphabetical (A-Z)");
  const [orderBy, setOrderBy] = useState("Ascending");

  const loadAccounts = () => {
    fetch("/api/admin/users")
      .then((res) => (res.ok ? res.json() : []))
      .then(setAccounts)
      .catch(() => setAccounts([]));
  };

  useEffect(() => {
    fetch("/api/me")
      .then((res) => (res.ok ? res.json() : null))
      .then((profile: SessionProfile | null) => {
        if (!profile || profile.role === "inspector") {
          router.replace("/dashboard");
          return;
        }
        setSessionProfile(profile);
      });

    fetch("/api/inspectors")
      .then((res) => (res.ok ? res.json() : []))
      .then(setInspectors)
      .catch(() => setLoadError("No inspector roster found. Run the backend pipeline (python main.py) to generate it."));

    loadAccounts();
  }, [router]);

  async function handleRoleChange(id: string, role: AccountRole) {
    setAccounts((prev) => prev.map((a) => (a.id === id ? { ...a, role } : a)));
    await fetch(`/api/admin/users/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ role }),
    });
  }

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
  const isAdmin = sessionProfile?.role === "admin";

  if (!sessionProfile) return null;

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

        {/* ── Account Access (login accounts / AuthService) ── */}
        <div className={styles.card} style={{ marginBottom: "1rem" }}>
          <div className={styles.cardHeader}>
            <div>
              <h1 className={styles.cardTitle}>
                Account <span className={styles.accent}>Access</span>
              </h1>
              <p className={styles.cardSub}>
                {isAdmin ? "Manage login accounts and role assignments" : "View login accounts (role changes require Admin)"}
              </p>
            </div>
          </div>
          <div className={styles.divider} />

          {isAdmin && <CreateAccountForm inspectors={inspectors} onCreated={loadAccounts} />}

          <div className={styles.tableWrap}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th className={styles.th}>Account</th>
                  <th className={styles.th}>Linked Inspector</th>
                  <th className={styles.th} style={{ textAlign: "center" }}>Role</th>
                </tr>
              </thead>
              <tbody>
                {accounts.map((a) => (
                  <tr key={a.id} className={styles.row}>
                    <td className={styles.td}>
                      <div className={styles.employeeCell}>
                        <AvatarIcon />
                        <div>
                          <div className={styles.employeeName}>{a.full_name || a.email}</div>
                          <div className={styles.employeeId}>{a.email}</div>
                        </div>
                      </div>
                    </td>
                    <td className={styles.td}>{a.inspector_id ?? "—"}</td>
                    <td className={`${styles.td} ${styles.tdCenter}`}>
                      {isAdmin ? (
                        <select
                          className={styles.select}
                          value={a.role}
                          onChange={(e) => handleRoleChange(a.id, e.target.value as AccountRole)}
                        >
                          <option value="inspector">Inspector</option>
                          <option value="manager">Manager</option>
                          <option value="admin">Admin</option>
                        </select>
                      ) : (
                        <span className={styles.statusBadge} style={ROLE_STYLE[a.role]}>{a.role}</span>
                      )}
                    </td>
                  </tr>
                ))}
                {accounts.length === 0 && (
                  <tr><td colSpan={3} className={styles.emptyRow}>No accounts yet.</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* ── Main white card: Inspector Roster ── */}
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
