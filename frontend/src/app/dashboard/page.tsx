"use client";

import Link from "next/link";
import Sidebar from "@/components/Sidebar";
import TopRight from "@/components/TopRight";
import Skeleton from "@/components/ui/Skeleton";
import StatCard from "@/components/ui/StatCard";
import EmptyState from "@/components/ui/EmptyState";
import Badge from "@/components/ui/Badge";
import type { BadgeTone } from "@/components/ui/Badge";
import styles from "./page.module.css";
import { motion } from "framer-motion";
import { fadeInUp, staggerContainer } from "@/lib/motion";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell, LabelList, ResponsiveContainer } from "recharts";

import { useEffect, useMemo, useState } from "react";
import type { Alert } from "@/lib/types";
import type { SessionProfile } from "@/lib/supabaseSessionServer";

function timeOfDayGreeting(): string {
  const hour = new Date().getHours();
  if (hour < 12) return "Good Morning";
  if (hour < 18) return "Good Afternoon";
  return "Good Evening";
}

const RISK_TONE: Record<string, BadgeTone> = {
  Low: "good",
  Medium: "warning",
  High: "serious",
  Critical: "critical",
};

const TIER_ORDER = ["Low", "Medium", "High", "Critical"] as const;
const TIER_COLOR_VAR: Record<(typeof TIER_ORDER)[number], string> = {
  Low: "var(--status-good)",
  Medium: "var(--status-warning)",
  High: "var(--status-serious)",
  Critical: "var(--status-critical)",
};

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
  delayProb: number;
}

/* ─── Risk-tier distribution chart (real data, no fake trend) ───── */
function RiskTierChart({ projects }: { projects: ProjectData[] }) {
  const data = useMemo(() => {
    const counts: Record<string, number> = { Low: 0, Medium: 0, High: 0, Critical: 0 };
    for (const p of projects) if (p.risk in counts) counts[p.risk]++;
    return TIER_ORDER.map((tier) => ({ tier, count: counts[tier] }));
  }, [projects]);

  return (
    <div className={styles.chartWrap}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 28, left: 4, bottom: 4 }}>
          <CartesianGrid horizontal={false} stroke="var(--border-subtle)" />
          <XAxis type="number" allowDecimals={false} axisLine={false} tickLine={false} tick={{ fill: "var(--ink-500)", fontSize: 10, fontFamily: "Inter, sans-serif" }} />
          <YAxis
            type="category"
            dataKey="tier"
            axisLine={false}
            tickLine={false}
            width={64}
            tick={{ fill: "var(--ink-700)", fontSize: 11, fontFamily: "Inter, sans-serif", fontWeight: 600 }}
          />
          <Tooltip
            cursor={{ fill: "var(--surface-sunken)" }}
            contentStyle={{ background: "var(--surface-card)", border: "1px solid var(--border-subtle)", borderRadius: 8, fontSize: 12 }}
            formatter={(value) => [`${value} project${value === 1 ? "" : "s"}`, "Count"]}
          />
          <Bar dataKey="count" radius={[0, 4, 4, 0]} maxBarSize={22}>
            {data.map((d) => (
              <Cell key={d.tier} fill={TIER_COLOR_VAR[d.tier as keyof typeof TIER_COLOR_VAR]} />
            ))}
            <LabelList dataKey="count" position="right" style={{ fill: "var(--ink-700)", fontSize: 11, fontWeight: 700 }} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/* ─── Dashboard Page ──────────────────────────────── */
export default function DashboardPage() {
  const [projects, setProjects] = useState<ProjectData[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [profile, setProfile] = useState<SessionProfile | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadProjects() {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch("/api/projects");
        if (!res.ok) throw new Error("Failed to load projects");
        const data = await res.json();
        if (!cancelled) setProjects(data);
      } catch {
        if (!cancelled) setError("Unable to load project data. Run the backend pipeline (python main.py) with Supabase configured, or check your connection.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    loadProjects();
    fetch("/api/alerts").then((res) => (res.ok ? res.json() : [])).then((data) => !cancelled && setAlerts(data)).catch(() => !cancelled && setAlerts([]));
    fetch("/api/me").then((res) => (res.ok ? res.json() : null)).then((data) => !cancelled && setProfile(data)).catch(() => !cancelled && setProfile(null));

    return () => { cancelled = true; };
  }, []);

  const today = new Date().toLocaleDateString("en-US", { weekday: "long", month: "long", day: "numeric", year: "numeric" });
  const greetingName = profile?.fullName?.split(" ")[0] || profile?.email?.split("@")[0] || "";

  const attentionProjects = useMemo(() => {
    return projects
      .filter((p) => p.risk === "Critical" || p.risk === "High")
      .sort((a, b) => b.delayProb - a.delayProb)
      .slice(0, 5);
  }, [projects]);

  const totalProjects = projects.length;
  const completedProjects = projects.filter((p) => p.status === "Completed").length;
  const ongoingProjects = projects.filter((p) => p.status === "In Progress" || p.status === "On Schedule").length;
  const criticalCount = projects.filter((p) => p.risk === "Critical").length;
  const highRiskCount = projects.filter((p) => p.risk === "High").length;

  return (
    <div className={styles.layout}>
      <Sidebar />
      <div className={styles.main}>

        {/* ── Top Bar ─────────────────────────────── */}
        <div className={styles.topBar}>
          <TopRight />
        </div>

        {/* ── Page Body ───────────────────────────── */}
        <motion.div className={styles.body} variants={staggerContainer} initial="hidden" animate="visible">

          {/* ── Slim header (replaces the old big gradient banner) ── */}
          <motion.div className={styles.header} variants={fadeInUp}>
            <div className={styles.headerLeft}>
              <div className={styles.breadcrumb}>
                Province of Iloilo / <span>Governance Intelligence Dashboard</span>
              </div>
              <div className={styles.greeting}>
                {timeOfDayGreeting()}{greetingName ? `, ${greetingName}` : ""} <span>— {today}</span>
              </div>
            </div>
            <div className={styles.headerRight}>
              <span className={`${styles.pill} ${styles.pillCritical}`}><span className={styles.pillDot} />{criticalCount} Critical</span>
              <span className={`${styles.pill} ${styles.pillSerious}`}><span className={styles.pillDot} />{highRiskCount} High-Risk</span>
            </div>
          </motion.div>

          {error && (
            <motion.div className={styles.card} variants={fadeInUp} style={{ marginBottom: "var(--space-5)", borderLeft: "3px solid var(--status-critical)" }}>
              <div className={styles.cardTitle}>Couldn&apos;t load project data</div>
              <div className={styles.cardSub} style={{ marginBottom: 0 }}>{error}</div>
            </motion.div>
          )}

          {/* ── Needs Attention (the headline of the page) ───────── */}
          <div className={styles.sectionLabel}>Needs Attention</div>
          <motion.div className={styles.attentionZone} variants={fadeInUp}>

            <div className={`${styles.card} ${styles.attentionCard}`}>
              <div className={styles.attentionListHead}>
                <div className={styles.cardTitle} style={{ marginBottom: 0 }}>Critical &amp; High-Risk Projects</div>
                <Link href="/projects" className={styles.viewAll}>View all →</Link>
              </div>

              <div className={styles.attentionCardBody}>
                {loading ? (
                  <div className={styles.attnSkeletonList}>
                    {Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} height={52} radius="var(--radius-md)" />)}
                  </div>
                ) : attentionProjects.length === 0 ? (
                  <EmptyState title="Nothing urgent right now" message="No Critical or High-risk projects in the monitored cohort." />
                ) : (
                  attentionProjects.map((p) => {
                    const isCritical = p.risk === "Critical";
                    return (
                      <div key={p.id} className={styles.attnRow}>
                        <div className={`${styles.stripe} ${isCritical ? styles.stripeCritical : styles.stripeSerious}`} />
                        <div className={styles.attnMain}>
                          <div className={styles.attnName}>{p.name} — {p.municipality}</div>
                          <div className={styles.attnMeta}>{p.inspector === "N/A" ? "Unassigned" : p.inspector} · {p.progress}% complete</div>
                        </div>
                        <div className={styles.attnMeter}>
                          <span style={{ width: `${Math.round(p.delayProb * 100)}%`, background: isCritical ? "var(--status-critical)" : "var(--status-serious)" }} />
                        </div>
                        <div className={`${styles.attnScore} ${isCritical ? styles.attnScoreCritical : styles.attnScoreSerious}`}>
                          {Math.round(p.delayProb * 100)}%
                        </div>
                      </div>
                    );
                  })
                )}
              </div>
            </div>

            {/* AI Forecast Alerts -- fixed to the same height as the list on the left; scrolls internally instead of growing the card when there are more items */}
            <div className={`${styles.card} ${styles.attentionCard}`}>
              <div className={styles.cardTitle}>AI Forecast Alerts</div>
              <div className={styles.cardSub}>Predictive Intelligence</div>
              <div className={styles.attentionCardBody}>
              {loading ? (
                <div style={{ display: "flex", flexDirection: "column", gap: "0.6rem" }}>
                  <Skeleton height={48} radius="var(--radius-md)" />
                  <Skeleton height={48} radius="var(--radius-md)" />
                  <Skeleton height={48} radius="var(--radius-md)" />
                </div>
              ) : alerts.length === 0 ? (
                <EmptyState title="No active alerts" message="You'll see tier-escalation and critical-risk alerts here as soon as the pipeline flags one." />
              ) : (
                <div className={styles.alertList}>
                  {alerts.slice(0, 5).map((a) => {
                    const isAmber = a.type !== "CRITICAL_RISK";
                    return (
                      <div key={a.id} className={`${styles.alertItem} ${isAmber ? styles.alertItemAmber : ""}`}>
                        <div className={styles.alertLeft}>
                          <div className={`${styles.alertIconWrap} ${isAmber ? styles.alertIconWrapAmber : ""}`}>
                            <svg width="15" height="15" viewBox="0 0 24 24" fill="none"
                              stroke={isAmber ? "var(--status-warning-text)" : "var(--status-critical-text)"} strokeWidth="2.5"
                              strokeLinecap="round" strokeLinejoin="round">
                              <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
                              <line x1="12" y1="9" x2="12" y2="13" /><line x1="12" y1="17" x2="12.01" y2="17" />
                            </svg>
                          </div>
                          <div>
                            <div className={styles.alertName}>{a.projectId}</div>
                            <div className={`${styles.alertDesc} ${isAmber ? styles.alertDescAmber : ""}`}>
                              {a.message}
                            </div>
                          </div>
                        </div>
                        {a.riskScore !== null && (
                          <div className={`${styles.alertConf} ${isAmber ? styles.alertConfAmber : ""}`}>
                            {(a.riskScore * 100).toFixed(0)}% risk
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
              </div>
            </div>
          </motion.div>

          {/* ── Portfolio at a Glance (demoted below the attention zone): the
               risk-tier chart pairs here with the KPI tiles, since both are
               portfolio context rather than an action item. ── */}
          <div className={styles.sectionLabel}>Portfolio at a Glance</div>
          <motion.div className={styles.glanceRow} variants={fadeInUp}>
            <div className={styles.card}>
              <div className={styles.cardTitle}>Risk Tier Distribution</div>
              <div className={styles.cardSub}>Monitored projects by current predicted risk</div>
              {loading ? <Skeleton height={150} radius="var(--radius-md)" /> : <RiskTierChart projects={projects} />}
            </div>
            <div className={styles.kpiGrid}>
              <StatCard label="Total Projects" value={totalProjects} tone="accent" loading={loading} icon={<FolderIcon />} />
              <StatCard label="Completed Projects" value={completedProjects} tone="good" loading={loading} icon={<CheckIcon />} />
              <StatCard label="Ongoing Projects" value={ongoingProjects} tone="accent" loading={loading} icon={<SyncIcon />} />
              <StatCard label="Critical Projects" value={criticalCount} tone="critical" loading={loading} icon={<AlertIcon />} />
            </div>
          </motion.div>

          {/* ── All Projects Table ───────────────── */}
          <motion.div className={styles.projectsCard} variants={fadeInUp}>
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
                  {["Project", "Municipality", "Progress", "Budget", "Risk", "Inspector"].map((h) => (
                    <th key={h} className={styles.th}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {loading && Array.from({ length: 5 }).map((_, i) => (
                  <tr key={i} className={styles.tr}>
                    {Array.from({ length: 6 }).map((__, j) => (
                      <td key={j} className={styles.td}><Skeleton height="1rem" /></td>
                    ))}
                  </tr>
                ))}

                {!loading && projects.slice(0, 5).map((p) => (
                  <tr key={p.id} className={styles.tr}>
                    <td className={styles.td}>
                      <div className={styles.projName}>{p.name}</div>
                      <div className={styles.projId}>{p.id}</div>
                    </td>
                    <td className={styles.td}>{p.municipality}</td>
                    <td className={styles.td}>{p.progress}%</td>
                    <td className={styles.td}>{p.budget}</td>
                    <td className={styles.td}><Badge tone={RISK_TONE[p.risk] ?? "neutral"}>{p.risk}</Badge></td>
                    <td className={styles.td}>{p.inspector}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </motion.div>

        </motion.div>
      </div>
    </div>
  );
}

/* ─── Stat Card Icons ─────────────────────────────── */
function FolderIcon() {
  return (
    <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
      <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
    </svg>
  );
}
function CheckIcon() {
  return (
    <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
      <rect x="3" y="4" width="18" height="18" rx="2" />
      <polyline points="9 12 11 14 15 10" />
    </svg>
  );
}
function SyncIcon() {
  return (
    <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
      <path d="M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8" />
      <path d="M21 3v5h-5" />
    </svg>
  );
}
function AlertIcon() {
  return (
    <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
      <line x1="12" y1="9" x2="12" y2="13" /><line x1="12" y1="17" x2="12.01" y2="17" />
    </svg>
  );
}
