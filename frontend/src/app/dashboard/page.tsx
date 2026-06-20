"use client";

import Link from "next/link";
import Sidebar from "@/components/Sidebar";
import styles from "./page.module.css";

/* ─── Mock Data ─────────────────────────────────── */
const PROJECTS = [
  { id: "PPS-124", name: "Aganan Flyover, Brgy. Aganan", municipality: "Pavia",     progress: 30, budget: "440,000,000", risk: "High",   inspector: "Engr. Rico Cruz"  },
  { id: "PPS-125", name: "Buntat Road, Brgy. Buntat",    municipality: "San Miguel", progress: 45, budget: "320,000,000", risk: "Medium", inspector: "Engr. Maria Lopez" },
  { id: "PPS-126", name: "Carmen Bridge, Brgy. Carmen",  municipality: "Carmen",     progress: 60, budget: "550,000,000", risk: "High",   inspector: "Engr. Jose Ramos"  },
];

const ALERTS = [
  { name: "Aganan Flyover",              desc: "Likely delayed by 35 days", conf: "95 % conf.", severity: "high"  },
  { name: "Pototan Flood Control Project", desc: "Likely delayed by 6 days",  conf: "78 % conf.", severity: "high"  },
  { name: "Carles Seawall",              desc: "Minor delay risk rising",    conf: "67 % conf.", severity: "amber" },
];

/* ─── Simple SVG sparkline ──────────────────────── */
const SPARK_POINTS = [2,3,2.5,4,3.5,5,4.5,6.5,5,4,5.5,4,3,4.5,3.5,4,3,2.5,3,2.5,3,4,3.5,4,3.5,4.5,3,4,2.5,3,2.5,2,3,2.5,3,2,1.5];
function SparkLine() {
  const W = 380, H = 150, PAD = 20;
  const xs = SPARK_POINTS.map((_, i) => PAD + (i / (SPARK_POINTS.length - 1)) * (W - PAD * 2));
  const max = Math.max(...SPARK_POINTS), min = Math.min(...SPARK_POINTS);
  const ys = SPARK_POINTS.map(v => H - PAD - ((v - min) / (max - min)) * (H - PAD * 2));
  const linePath = xs.map((x, i) => `${i === 0 ? "M" : "L"} ${x} ${ys[i]}`).join(" ");
  const areaPath = `${linePath} L ${xs[xs.length-1]} ${H-PAD} L ${xs[0]} ${H-PAD} Z`;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className={styles.chartWrap} preserveAspectRatio="none">
      <defs>
        <linearGradient id="area-grad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#e74c3c" stopOpacity="0.15" />
          <stop offset="100%" stopColor="#e74c3c" stopOpacity="0" />
        </linearGradient>
      </defs>
      {/* Grid lines */}
      {[0,1,2,3,4,5].map(i => (
        <line key={i} x1={PAD} x2={W-PAD} y1={PAD + i*((H-PAD*2)/5)} y2={PAD + i*((H-PAD*2)/5)} stroke="#e8edf2" strokeWidth="1" />
      ))}
      {/* Area fill */}
      <path d={areaPath} fill="url(#area-grad)" />
      {/* Line */}
      <path d={linePath} fill="none" stroke="#e74c3c" strokeWidth="2" strokeLinejoin="round" />
    </svg>
  );
}

function riskClass(risk: string, s: typeof styles) {
  if (risk === "High" || risk === "Critical") return `${s.badge} ${s.badgeHigh}`;
  if (risk === "Medium") return `${s.badge} ${s.badgeMedium}`;
  return `${s.badge} ${s.badgeLow}`;
}

/* ─── Dashboard Page ────────────────────────────── */
export default function DashboardPage() {
  return (
    <div className={styles.layout}>
      <Sidebar />
      <div className={styles.main}>
        {/* Top Bar */}
        <div className={styles.topBar}>
          <div className={styles.searchWrapper}>
            <svg className={styles.searchIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
            <input className={styles.searchInput} placeholder="Search for projects, locations..." />
          </div>
          <div className={styles.topRight}>
            <div className={styles.datePill}>
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="4" width="18" height="18" rx="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>
              May 15, 2026
            </div>
            <button className={styles.iconBtn}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.73 21a2 2 0 0 1-3.46 0"/></svg>
            </button>
            <button className={styles.iconBtn}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
            </button>
          </div>
        </div>

        {/* Body */}
        <div className={styles.body}>
          {/* Breadcrumb */}
          <div className={styles.breadcrumb}>
            Province of Iloilo / <span>Governance Intelligence Dashboard</span>
          </div>

          {/* Banner */}
          <div className={styles.banner}>
            <div className={styles.bannerTitle}>Good Morning, Ricardo!</div>
            <div className={styles.bannerSub}>
              As of <span className={styles.bannerHighlight}>Friday, May 15, 2026</span>, we are tracking{" "}
              <span className={styles.bannerHighlight}>5 Critical projects</span> and{" "}
              <span className={styles.bannerHighlight}>28 High-Risk</span> projects across Iloilo Province.
            </div>
          </div>

          {/* Stat Cards */}
          <div className={styles.statsRow}>
            {[
              { label: "Total Projects",     badge: "+0.3%", badgeType: "green", value: "643", valueClass: "",                   icon: FolderIcon },
              { label: "Completed Projects", badge: "+0.3%", badgeType: "green", value: "45",  valueClass: styles.statValueGreen, icon: CheckIcon  },
              { label: "Ongoing Projects",   badge: "+0.3%", badgeType: "green", value: "267", valueClass: styles.statValueBlue,  icon: SyncIcon   },
              { label: "Critical Projects",  badge: "+5.3%", badgeType: "red",   value: "15",  valueClass: styles.statValueRed,   icon: AlertIcon  },
            ].map(({ label, badge, badgeType, value, valueClass, icon: Icon }) => (
              <div key={label} className={styles.statCard}>
                <div className={styles.statLeft}>
                  <div className={styles.statLabel}>
                    {label}
                    <span className={`${styles.statBadge} ${badgeType === "green" ? styles.statBadgeGreen : styles.statBadgeRed}`}>{badge}</span>
                  </div>
                  <div className={`${styles.statValue} ${valueClass}`}>{value}</div>
                </div>
                <Icon className={styles.statIcon} />
              </div>
            ))}
          </div>

          {/* Mid Row */}
          <div className={styles.midRow}>
            {/* Delay Trends */}
            <div className={styles.card}>
              <div className={styles.cardTitle}>Delay Trends</div>
              <div className={styles.cardSub}>Rolling 1-year view with AI forecast</div>
              <SparkLine />
            </div>

            {/* AI Forecast Alerts */}
            <div className={styles.card}>
              <div className={styles.cardTitle}>AI Forecast Alerts</div>
              <div className={styles.cardSub}>Predictive Intelligence</div>
              <div className={styles.alertList}>
                {ALERTS.map((a) => (
                  <div key={a.name} className={`${styles.alertItem} ${a.severity === "amber" ? styles.alertItemAmber : ""}`}>
                    <div className={styles.alertLeft}>
                      <div className={`${styles.alertIconWrap} ${a.severity === "amber" ? styles.alertIconWrapAmber : ""}`}>
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={a.severity === "amber" ? "#d68910" : "#e74c3c"} strokeWidth="2.5">
                          <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                          <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
                        </svg>
                      </div>
                      <div>
                        <div className={styles.alertName}>{a.name}</div>
                        <div className={`${styles.alertDesc} ${a.severity === "amber" ? styles.alertDescAmber : ""}`}>
                          Likely <strong>delayed</strong> {a.desc.replace("Likely delayed ", "").replace("Minor delay risk rising", "")}
                          {a.severity === "amber" && "Minor delay risk rising"}
                        </div>
                      </div>
                    </div>
                    <div className={`${styles.alertConf} ${a.severity === "amber" ? styles.alertConfAmber : ""}`}>{a.conf}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* All Projects mini table */}
          <div className={styles.projectsCard}>
            <div className={styles.projectsHeader}>
              <div>
                <div className={styles.cardTitle}>All Projects</div>
                <div className={styles.cardSub} style={{ marginBottom: 0 }}>Click any row for AI analysis</div>
              </div>
              <Link href="/projects" className={styles.viewAll}>View All Projects →</Link>
            </div>
            <table className={styles.table}>
              <thead>
                <tr>
                  {["Project","Municipality","Progress","Budget","Risk","Inspector"].map(h => (
                    <th key={h} className={styles.th}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {PROJECTS.map(p => (
                  <tr key={p.id} className={styles.tr}>
                    <td className={styles.td}><div className={styles.projName}>{p.name}</div><div className={styles.projId}>{p.id}</div></td>
                    <td className={styles.td}>{p.municipality}</td>
                    <td className={styles.td}>{p.progress}%</td>
                    <td className={styles.td}>{p.budget}</td>
                    <td className={styles.td}><span className={riskClass(p.risk, styles)}>{p.risk}</span></td>
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
}

/* ─── Stat Card Icons ───────────────────────────── */
function FolderIcon({ className }: { className?: string }) {
  return <svg className={className} viewBox="0 0 24 24" fill="none" stroke="#1264ae" strokeWidth="1.5"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>;
}
function CheckIcon({ className }: { className?: string }) {
  return <svg className={className} viewBox="0 0 24 24" fill="none" stroke="#27ae60" strokeWidth="1.5"><rect x="3" y="4" width="18" height="18" rx="2"/><polyline points="9 11 12 14 22 4"/></svg>;
}
function SyncIcon({ className }: { className?: string }) {
  return <svg className={className} viewBox="0 0 24 24" fill="none" stroke="#2980b9" strokeWidth="1.5"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>;
}
function AlertIcon({ className }: { className?: string }) {
  return <svg className={className} viewBox="0 0 24 24" fill="none" stroke="#e74c3c" strokeWidth="1.5"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>;
}
