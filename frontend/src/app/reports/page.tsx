"use client";

import { useState, useMemo } from "react";
import Sidebar from "@/components/Sidebar";
import TopRight from "@/components/TopRight";
import styles from "./page.module.css";

/* ─── Mock report data ────────────────────────────────── */
const REPORTS = [
  { id:"PPS-124", name:"Aganan Flyover, Brgy. Aganan",           quarter:"Q1", progress:30, date:"May 2, 2026",       status:"Validated", engineer:"Engr. Rico Cruz"    },
  { id:"PPS-125", name:"Aglipay Street, Brgy. Balabago",         quarter:"Q2", progress:45, date:"April 10, 2026",    status:"Pending",   engineer:"Engr. Maria Santos"  },
  { id:"PPS-126", name:"Baluarte, Brgy. Jaro",                   quarter:"Q1", progress:35, date:"April 15, 2026",    status:"Validated", engineer:"Engr. John Doe"     },
  { id:"PPS-127", name:"Diversion Road, Mandurriao",             quarter:"Q3", progress:50, date:"March 22, 2026",    status:"Pending",   engineer:"Engr. Lina Reyes"   },
  { id:"PPS-128", name:"SM City Iloilo, Mandurriao",             quarter:"Q4", progress:60, date:"February 14, 2026", status:"Validated", engineer:"Engr. Albert Lim"   },
  { id:"PPS-129", name:"Iloilo River Esplanade, City Proper",    quarter:"Q1", progress:75, date:"February 3, 2026",  status:"Approved",  engineer:"Engr. Sara Tan"     },
  { id:"PPS-130", name:"Molo Plaza Rehabilitation, Molo",        quarter:"Q2", progress:20, date:"January 28, 2026",  status:"Pending",   engineer:"Engr. Jose Reyes"   },
  { id:"PPS-131", name:"Jaro Flood Control, Jaro",               quarter:"Q3", progress:88, date:"January 15, 2026",  status:"Validated", engineer:"Engr. Nina Cruz"    },
  { id:"PPS-132", name:"Pavia Public Market Upgrade, Pavia",     quarter:"Q4", progress:42, date:"December 20, 2025", status:"Rejected",  engineer:"Engr. Mike Delos"   },
  { id:"PPS-133", name:"Cabatuan Bridge Widening, Cabatuan",     quarter:"Q1", progress:65, date:"December 5, 2025",  status:"Validated", engineer:"Engr. Lyn Navarro"  },
  { id:"PPS-134", name:"Oton Seawall Phase 3, Oton",             quarter:"Q2", progress:15, date:"November 28, 2025", status:"Pending",   engineer:"Engr. Ben Garcia"   },
  { id:"PPS-135", name:"San Miguel Road Extension, San Miguel",  quarter:"Q3", progress:55, date:"November 10, 2025", status:"Approved",  engineer:"Engr. Clara Vega"   },
  { id:"PPS-136", name:"Tigbauan Port Improvement, Tigbauan",    quarter:"Q4", progress:33, date:"October 25, 2025",  status:"Pending",   engineer:"Engr. Rey Lim"      },
  { id:"PPS-137", name:"Leon Water Supply, Leon",                quarter:"Q1", progress:90, date:"October 12, 2025",  status:"Validated", engineer:"Engr. Paz Flores"   },
  { id:"PPS-138", name:"Lambunao Health Center, Lambunao",       quarter:"Q2", progress:48, date:"October 1, 2025",   status:"Approved",  engineer:"Engr. Rod Santos"   },
  { id:"PPS-139", name:"Calinog Evacuation Center, Calinog",     quarter:"Q3", progress:22, date:"September 18, 2025",status:"Pending",   engineer:"Engr. Ria Cruz"     },
  { id:"PPS-140", name:"Barotac Nuevo Fish Port, Barotac Nuevo", quarter:"Q4", progress:71, date:"September 5, 2025", status:"Validated", engineer:"Engr. Dan Tan"      },
  { id:"PPS-141", name:"Dumangas Public Library, Dumangas",      quarter:"Q1", progress:38, date:"August 22, 2025",   status:"Rejected",  engineer:"Engr. Ana Bello"    },
  { id:"PPS-142", name:"Pototan Sports Complex, Pototan",        quarter:"Q2", progress:82, date:"August 10, 2025",   status:"Approved",  engineer:"Engr. Vic Santos"   },
  { id:"PPS-143", name:"Banate Eco-Tourism Site, Banate",        quarter:"Q3", progress:10, date:"July 28, 2025",     status:"Pending",   engineer:"Engr. Luz Reyes"    },
];

const STATUS_STYLE: Record<string, { bg: string; color: string }> = {
  Validated: { bg: "#27ae60", color: "#fff" },
  Approved:  { bg: "#2756c5", color: "#fff" },
  Pending:   { bg: "#f59e0b", color: "#fff" },
  Rejected:  { bg: "#e74c3c", color: "#fff" },
};

/* ─── Page Component ──────────────────────────────────── */
export default function ReportsPage() {
  const [search,  setSearch]  = useState("");
  const [quarter, setQuarter] = useState("All Quarters");
  const [status,  setStatus]  = useState("All Status");
  const [sortBy,  setSortBy]  = useState("Most Recent");
  const [orderBy, setOrderBy] = useState("Ascending");

  /* ── Filter logic ── */
  const filtered = useMemo(() => {
    let list = [...REPORTS];
    if (search)                         list = list.filter(r => r.name.toLowerCase().includes(search.toLowerCase()) || r.id.toLowerCase().includes(search.toLowerCase()));
    if (quarter !== "All Quarters")     list = list.filter(r => r.quarter === quarter);
    if (status  !== "All Status")       list = list.filter(r => r.status  === status);
    return list;
  }, [search, quarter, status]);

  const clearFilters = () => { setSearch(""); setQuarter("All Quarters"); setStatus("All Status"); };

  return (
    <div className={styles.shell}>
      <Sidebar />
      <div className={styles.main}>

        {/* ── Top bar ── */}
        <div className={styles.topbar}>
          <div className={styles.breadcrumb}>
            <span>Reports</span>
            <span className={styles.sep}>/</span>
            <span className={styles.breadActive}>Project Reports</span>
          </div>
          <TopRight />
        </div>

        {/* ── Main white card ── */}
        <div className={styles.card}>

          {/* Card header */}
          <div className={styles.cardHeader}>
            <h1 className={styles.cardTitle}>
              Project <span className={styles.accent}>Reports</span>
            </h1>
            <p className={styles.cardSub}>View recent project reports and assessments</p>
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
                  <option>All Quarters</option>
                  <option>Q1</option><option>Q2</option><option>Q3</option><option>Q4</option>
                </select>
                <svg className={styles.selectChevron} width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><polyline points="6 9 12 15 18 9"/></svg>
              </div>
            </div>

            <div className={styles.filterGroup}>
              <label className={styles.filterLabel}>Status</label>
              <div className={styles.selectWrap}>
                <select className={styles.select} value={status} onChange={e => setStatus(e.target.value)}>
                  <option>All Status</option>
                  <option>Validated</option><option>Approved</option>
                  <option>Pending</option><option>Rejected</option>
                </select>
                <svg className={styles.selectChevron} width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><polyline points="6 9 12 15 18 9"/></svg>
              </div>
            </div>

            <button className={styles.clearBtn} onClick={clearFilters}>Clear Filters</button>
          </div>

          {/* Sort + count row */}
          <div className={styles.sortRow}>
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
                  <th className={styles.th} style={{ textAlign:"center" }}>Date</th>
                  <th className={styles.th} style={{ textAlign:"center" }}>Status</th>
                  <th className={styles.th}>Submitted By</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((r, i) => {
                  const st = STATUS_STYLE[r.status] ?? { bg:"#94a3b8", color:"#fff" };
                  return (
                    <tr key={r.id} className={styles.row}>
                      <td className={styles.td}>
                        <div className={styles.projectName}>{r.name}</div>
                        <div className={styles.projectId}>{r.id}</div>
                        <div className={styles.progressBarBg}>
                          <div className={styles.progressBar} style={{ width:`${r.progress}%` }} />
                        </div>
                      </td>
                      <td className={`${styles.td} ${styles.tdCenter}`}>{r.quarter}</td>
                      <td className={`${styles.td} ${styles.tdCenter}`}>{r.progress}%</td>
                      <td className={`${styles.td} ${styles.tdCenter}`}>{r.date}</td>
                      <td className={`${styles.td} ${styles.tdCenter}`}>
                        <span className={styles.statusBadge} style={{ background: st.bg, color: st.color }}>
                          {r.status}
                        </span>
                      </td>
                      <td className={styles.td}>{r.engineer}</td>
                    </tr>
                  );
                })}

                {filtered.length === 0 && (
                  <tr>
                    <td colSpan={6} className={styles.emptyRow}>No reports found matching your filters.</td>
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
