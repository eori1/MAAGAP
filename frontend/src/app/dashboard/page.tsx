"use client";

import Link from "next/link";
import Sidebar from "@/components/Sidebar";
import TopRight from "@/components/TopRight";
import styles from "./page.module.css";
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, ResponsiveContainer } from "recharts";

import { useState, useEffect } from "react";
import { fetchProjects } from "@/lib/api";

interface AlertData {
  id: string;
  type: "TIER_ESCALATION" | "CRITICAL_RISK";
  projectId: string;
  riskScore: number;
  message: string;
}

interface SessionProfile {
  fullName: string | null;
  email: string;
}

function timeOfDayGreeting(): string {
  const hour = new Date().getHours();
  if (hour < 12) return "Good Morning";
  if (hour < 18) return "Good Afternoon";
  return "Good Evening";
}

/* ─── Chart using Recharts ───────────────────────── */
const chartData = [
  { m: "0",  val: 2.3 }, { m: "",   val: 2.8 }, { m: "1",  val: 2.5 }, { m: "",   val: 3.2 }, { m: "2",  val: 3.0 },
  { m: "",   val: 3.8 }, { m: "3",  val: 3.5 }, { m: "",   val: 4.2 }, { m: "4",  val: 3.8 }, { m: "",   val: 4.8 },
  { m: "5",  val: 5.2 }, { m: "",   val: 6.5 }, { m: "6",  val: 7.2 }, { m: "",   val: 6.8 }, { m: "7",  val: 5.5 },
  { m: "",   val: 4.8 }, { m: "8",  val: 4.2 }, { m: "",   val: 3.8 }, { m: "9",  val: 3.2 }, { m: "",   val: 2.8 },
  { m: "10", val: 3.5 }, { m: "",   val: 4.2 }, { m: "11", val: 4.8 }, { m: "",   val: 5.5 }, { m: "",   val: 4.5 },
  { m: "",   val: 3.8 }, { m: "",   val: 3.2 }, { m: "",   val: 2.8 }, { m: "",   val: 3.5 }, { m: "",   val: 3.0 },
  { m: "",   val: 2.5 }, { m: "",   val: 2.0 }, { m: "",   val: 1.8 }
];

function DelayChart() {
  return (
    <div style={{ width: '100%', height: '160px', marginTop: '1rem' }}>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={chartData} margin={{ top: 10, right: 0, left: -25, bottom: 0 }}>
          <defs>
            <linearGradient id="colorVal" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#e74c3c" stopOpacity={0.15}/>
              <stop offset="95%" stopColor="#e74c3c" stopOpacity={0}/>
            </linearGradient>
          </defs>
          <CartesianGrid vertical={false} stroke="#eeeff3" />
          <XAxis 
            dataKey="m" 
            axisLine={false} 
            tickLine={false} 
            tick={{ fill: '#b0bbc9', fontSize: 10, fontFamily: 'Inter, sans-serif' }}
            dy={5}
          />
          <YAxis 
            axisLine={false} 
            tickLine={false} 
            tick={{ fill: '#b0bbc9', fontSize: 10, fontFamily: 'Inter, sans-serif' }}
            ticks={[0, 2.5, 5, 7.5, 10]}
            domain={[0, 10]}
          />
          <Area 
            type="monotone" 
            dataKey="val" 
            stroke="#e74c3c" 
            strokeWidth={2.5}
            fillOpacity={1} 
            fill="url(#colorVal)" 
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

function riskClass(risk: string, s: typeof styles) {
  if (risk === "Critical") return `${s.badge} ${s.badgeCritical}`;
  if (risk === "High")     return `${s.badge} ${s.badgeHigh}`;
  if (risk === "Medium")   return `${s.badge} ${s.badgeMedium}`;
  return `${s.badge} ${s.badgeLow}`;
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

/* ─── Dashboard Page ──────────────────────────────── */
export default function DashboardPage() {
  const [projects, setProjects] = useState<ProjectData[]>([]);
  const [alerts, setAlerts] = useState<AlertData[]>([]);
  const [profile, setProfile] = useState<SessionProfile | null>(null);

  useEffect(() => {
    fetchProjects().then(data => setProjects(data));
    fetch("/api/alerts").then(res => (res.ok ? res.json() : [])).then(setAlerts).catch(() => setAlerts([]));
    fetch("/api/me").then(res => (res.ok ? res.json() : null)).then(setProfile).catch(() => setProfile(null));
  }, []);

  const today = new Date().toLocaleDateString("en-US", { weekday: "long", month: "long", day: "numeric", year: "numeric" });
  const greetingName = profile?.fullName?.split(" ")[0] || profile?.email?.split("@")[0] || "";

  const totalProjects = projects.length;
  const completedProjects = projects.filter(p => p.status === "Completed").length;
  const ongoingProjects = projects.filter(p => p.status === "In Progress" || p.status === "On Schedule").length;
  const criticalCount = projects.filter(p => p.risk === "Critical").length;
  const highRiskCount = projects.filter(p => p.risk === "High").length;
  return (
    <div className={styles.layout}>
      <Sidebar />
      <div className={styles.main}>

        {/* ── Top Bar ─────────────────────────────── */}
        <div className={styles.topBar}>
          <div className={styles.searchWrapper}>
            <svg className={styles.searchIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
            </svg>
            <input className={styles.searchInput} placeholder="Search for projects, locations..." />
          </div>

          <TopRight />
        </div>

        {/* ── Page Body ───────────────────────────── */}
        <div className={styles.body}>

          {/* Breadcrumb */}
          <div className={styles.breadcrumb}>
            Province of Iloilo / <span>Governance Intelligence Dashboard</span>
          </div>

          {/* Welcome Banner */}
          <div className={styles.banner}>
            <div className={styles.bannerTitle}>{timeOfDayGreeting()}{greetingName ? `, ${greetingName}` : ""}!</div>
            <div className={styles.bannerSub}>
              As of <strong>{today}</strong>, we are tracking{" "}
              <strong>{criticalCount} Critical projects</strong> and{" "}
              <strong>{highRiskCount} High-Risk</strong> projects across Iloilo Province.
            </div>
          </div>

          {/* ── Stat Cards ───────────────────────── */}
          <div className={styles.statsRow}>
            <div className={styles.statCard}>
              <div className={styles.statLeft}>
                <div className={styles.statLabel}>Total Projects</div>
                <div className={`${styles.statValue} ${styles.statValueBlue}`}>{totalProjects}</div>
              </div>
              <FolderIcon className={styles.statIcon} />
            </div>

            <div className={styles.statCard}>
              <div className={styles.statLeft}>
                <div className={styles.statLabel}>Completed Projects</div>
                <div className={`${styles.statValue} ${styles.statValueGreen}`}>{completedProjects}</div>
              </div>
              <CheckIcon className={styles.statIcon} />
            </div>

            <div className={styles.statCard}>
              <div className={styles.statLeft}>
                <div className={styles.statLabel}>Ongoing Projects</div>
                <div className={`${styles.statValue} ${styles.statValueBlue}`}>{ongoingProjects}</div>
              </div>
              <SyncIcon className={styles.statIcon} />
            </div>

            <div className={styles.statCard}>
              <div className={styles.statLeft}>
                <div className={styles.statLabel}>Critical Projects</div>
                <div className={`${styles.statValue} ${styles.statValueRed}`}>{criticalCount}</div>
              </div>
              <AlertIcon className={styles.statIcon} />
            </div>
          </div>

          {/* ── Middle Row ───────────────────────── */}
          <div className={styles.midRow}>

            {/* Delay Trends */}
            <div className={styles.card}>
              <div className={styles.cardTitle}>Delay Trends</div>
              <div className={styles.cardSub}>Rolling 1-year view with AI forecast</div>
              <DelayChart />
            </div>

            {/* AI Forecast Alerts */}
            <div className={styles.card}>
              <div className={styles.cardTitle}>AI Forecast Alerts</div>
              <div className={styles.cardSub}>Predictive Intelligence</div>
              <div className={styles.alertList}>
                {alerts.length === 0 && (
                  <div className={styles.alertDesc} style={{ padding: "0.75rem 0" }}>No active alerts.</div>
                )}
                {alerts.slice(0, 5).map((a, idx) => {
                  const isAmber = a.type !== "CRITICAL_RISK";
                  return (
                    <div key={a.id} className={`${styles.alertItem} ${isAmber ? styles.alertItemAmber : ""} ${idx < Math.min(alerts.length, 5) - 1 ? styles.alertItemBorder : ""}`}>
                      <div className={styles.alertLeft}>
                        <div className={`${styles.alertIconWrap} ${isAmber ? styles.alertIconWrapAmber : ""}`}>
                          <svg width="15" height="15" viewBox="0 0 24 24" fill="none"
                            stroke={isAmber ? "#d97706" : "#e74c3c"} strokeWidth="2.5"
                            strokeLinecap="round" strokeLinejoin="round">
                            <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                            <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
                          </svg>
                        </div>
                        <div>
                          <div className={styles.alertName}>{a.projectId}</div>
                          <div className={`${styles.alertDesc} ${isAmber ? styles.alertDescAmber : ""}`}>
                            {a.message}
                          </div>
                        </div>
                      </div>
                      <div className={`${styles.alertConf} ${isAmber ? styles.alertConfAmber : ""}`}>
                        {(a.riskScore * 100).toFixed(0)}% risk
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* ── All Projects Table ───────────────── */}
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
                <tr className={styles.thead}>
                  {["Project", "Municipality", "Progress", "Budget", "Risk", "Inspector"].map(h => (
                    <th key={h} className={styles.th}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {projects.slice(0, 5).map(p => (
                  <tr key={p.id} className={styles.tr}>
                    <td className={styles.td}>
                      <div className={styles.projName}>{p.name}</div>
                      <div className={styles.projId}>{p.id}</div>
                    </td>
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

/* ─── Stat Card Icons ─────────────────────────────── */
function FolderIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="#1264ae" strokeWidth="1.6">
      <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>
    </svg>
  );
}
function CheckIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="#27ae60" strokeWidth="1.6">
      <rect x="3" y="4" width="18" height="18" rx="2"/>
      <polyline points="9 12 11 14 15 10"/>
    </svg>
  );
}
function SyncIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="#2980b9" strokeWidth="1.6">
      <path d="M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8"/>
      <path d="M21 3v5h-5"/>
    </svg>
  );
}
function AlertIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="#e74c3c" strokeWidth="1.6">
      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
      <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
    </svg>
  );
}
