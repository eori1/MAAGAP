"use client";

import { useState } from "react";
import Sidebar from "@/components/Sidebar";
import TopRight from "@/components/TopRight";
import styles from "./page.module.css";

/* ─── Types ───────────────────────────────────────────── */
interface AssignedProject {
  name: string;
  location: string;
  type: string;
  progress: number;
  delay: number;
  priority: "HIGH" | "MEDIUM" | "LOW";
  urgency: "VISIT ASAP" | "VISIT SOON" | "CHECK IN A WEEK";
}

interface Inspector {
  id: string;
  name: string;
  location: string;
  arCode: string;
  efficiency: number;
  priority: "HIGH" | "MEDIUM" | "LOW";
  totalProjects: number;
  projects: AssignedProject[];
}

/* ─── Mock Data ───────────────────────────────────────── */
const INSPECTORS: Inspector[] = [
  {
    id: "i1", name: "Juan de la Cruz", location: "Pavia, Iloilo",
    arCode: "AR-4213", efficiency: 89, priority: "HIGH", totalProjects: 12,
    projects: [
      { name: "Aganan Flyover Construction", location: "Pavia Iloilo", type: "Infrastructure",
        progress: 30, delay: 34, priority: "HIGH",   urgency: "VISIT ASAP" },
      { name: "Buhang Flyover Construction", location: "Tagbak Iloilo City", type: "Infrastructure",
        progress: 30, delay: 7,  priority: "MEDIUM", urgency: "VISIT ASAP" },
    ],
  },
  {
    id: "i2", name: "Maria Clara", location: "San Miguel, Iloilo",
    arCode: "AR-4214", efficiency: 75, priority: "MEDIUM", totalProjects: 9,
    projects: [
      { name: "San Miguel Bridge Rehabilitation", location: "San Miguel Iloilo", type: "Infrastructure",
        progress: 40, delay: 20, priority: "MEDIUM", urgency: "VISIT SOON" },
      { name: "Jose Rizal Memorial",  location: "Jaro, Iloilo City", type: "Infrastructure",
        progress: 50, delay: 10, priority: "LOW",    urgency: "CHECK IN A WEEK" },
    ],
  },
  {
    id: "i3", name: "Andrea Bonifacio", location: "Oton, Iloilo",
    arCode: "AR-4215", efficiency: 92, priority: "HIGH", totalProjects: 7,
    projects: [
      { name: "Oton Flood Control Phase 2", location: "Oton Iloilo", type: "Flood Control",
        progress: 20, delay: 45, priority: "HIGH",   urgency: "VISIT ASAP" },
      { name: "Tigbauan Road Widening",      location: "Tigbauan Iloilo", type: "Road",
        progress: 55, delay: 5,  priority: "LOW",    urgency: "CHECK IN A WEEK" },
    ],
  },
  {
    id: "i4", name: "Pedro Martinez", location: "Cabatuan, Iloilo",
    arCode: "AR-4216", efficiency: 81, priority: "MEDIUM", totalProjects: 7,
    projects: [
      { name: "Cabatuan Market Renovation", location: "Cabatuan Iloilo", type: "Public Bldg",
        progress: 65, delay: 0,  priority: "LOW",    urgency: "CHECK IN A WEEK" },
      { name: "Janiuay River Dike",          location: "Janiuay Iloilo", type: "Flood Control",
        progress: 35, delay: 18, priority: "MEDIUM", urgency: "VISIT SOON" },
    ],
  },
  {
    id: "i5", name: "Sofia Reyes", location: "Iloilo City",
    arCode: "AR-4217", efficiency: 95, priority: "LOW", totalProjects: 15,
    projects: [
      { name: "Iloilo City Boardwalk Extension", location: "Iloilo City", type: "Tourism",
        progress: 70, delay: 3,  priority: "LOW",    urgency: "CHECK IN A WEEK" },
      { name: "La Paz Water Supply Upgrade",      location: "La Paz Iloilo", type: "Utilities",
        progress: 45, delay: 12, priority: "MEDIUM", urgency: "VISIT SOON" },
    ],
  },
];

const OVERVIEW = [
  { name: "Juan de la Cruz", projects: 12, pct: 80 },
  { name: "Maria Gomez",     projects: 9,  pct: 60 },
  { name: "Pedro Martinez",  projects: 7,  pct: 47 },
  { name: "Sofia Reyes",     projects: 15, pct: 100 },
  { name: "Andrea Bonifacio",projects: 7,  pct: 47 },
];

const PRIORITY_COLORS = {
  HIGH:   { bg: "#fff2f2", border: "#e74c3c", text: "#e74c3c" },
  MEDIUM: { bg: "#fffbf0", border: "#f59e0b", text: "#f59e0b" },
  LOW:    { bg: "#f0fff4", border: "#27ae60", text: "#27ae60" },
};

const URGENCY_COLORS = {
  "VISIT ASAP":        "#e74c3c",
  "VISIT SOON":        "#f59e0b",
  "CHECK IN A WEEK":   "#27ae60",
};

const PROJECT_BG = {
  HIGH:   "#fff8ec",
  MEDIUM: "#fffbf0",
  LOW:    "#f8fff8",
};

/* ─── Icon Components ─────────────────────────────────── */
function InspectorIcon() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="7" r="4"/>
      <path d="M5.5 20a6.5 6.5 0 0 1 13 0"/>
    </svg>
  );
}

function ProjectIcon({ color }: { color: string }) {
  return (
    <div style={{
      width: 38, height: 38, borderRadius: "50%",
      background: color, display: "flex",
      alignItems: "center", justifyContent: "center", flexShrink: 0,
    }}>
      <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>
      </svg>
    </div>
  );
}

/* ─── Inspector Card ──────────────────────────────────── */
function InspectorCard({ inspector }: { inspector: Inspector }) {
  const [accepted, setAccepted] = useState(false);
  const pc = PRIORITY_COLORS[inspector.priority];

  return (
    <div className={styles.inspCard}>
      {/* Card header */}
      <div className={styles.inspHeader}>
        <div className={styles.inspLeft}>
          <div className={styles.inspAvatar}>
            <InspectorIcon />
          </div>
          <div>
            <div className={styles.inspName}>{inspector.name}</div>
            <div className={styles.inspMeta}>
              {inspector.location} · {inspector.arCode}&nbsp;
              <span style={{ color: "#27ae60", fontWeight: 700 }}>{inspector.efficiency}% Efficiency</span>
            </div>
          </div>
        </div>
        <div
          className={styles.priorityBadge}
          style={{ borderColor: pc.border, color: pc.text }}
        >
          {inspector.priority} PRIORITY
        </div>
      </div>

      {/* Assigned projects */}
      <div className={styles.projectsList}>
        {inspector.projects.map((p, i) => {
          const projBg = PROJECT_BG[p.priority];
          const urgColor = URGENCY_COLORS[p.urgency];
          const iconColor = p.priority === "HIGH" ? "#e74c3c" : p.priority === "MEDIUM" ? "#f59e0b" : "#27ae60";

          return (
            <div key={i} className={styles.projectRow} style={{ background: projBg }}>
              <ProjectIcon color={iconColor} />
              <div className={styles.projectInfo}>
                <div className={styles.projectName}>{p.name}</div>
                <div className={styles.projectMeta}>
                  {p.location} · {p.type} · {p.progress}% Complete
                  {p.delay > 0 && (
                    <span style={{ color: "#e74c3c", fontWeight: 700 }}> +{p.delay} days delayed</span>
                  )}
                </div>
              </div>
              <div className={styles.projectUrgency}>
                <div className={styles.urgPriority} style={{ color: urgColor }}>{p.priority}</div>
                <div className={styles.urgLabel}  style={{ color: urgColor }}>{p.urgency}</div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Action buttons */}
      <div className={styles.cardActions}>
        <button
          className={`${styles.acceptBtn} ${accepted ? styles.acceptBtnDone : ""}`}
          onClick={() => setAccepted(true)}
        >
          {accepted ? (
            <>✓ Allocation Accepted</>
          ) : (
            <>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><polyline points="20 6 9 17 4 12"/></svg>
              Accept AI Allocation
            </>
          )}
        </button>
        <button className={styles.editBtn}>
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>
          Manual Edit
        </button>
      </div>
    </div>
  );
}

/* ─── Page ────────────────────────────────────────────── */
export default function AllocationPage() {
  return (
    <div className={styles.shell}>
      <Sidebar />
      <div className={styles.main}>

        {/* Top bar */}
        <div className={styles.topbar}>
          <div className={styles.breadcrumb}>
            <span>Allocation</span>
            <span className={styles.sep}>/</span>
            <span className={styles.breadActive}>Inspector Allocation</span>
          </div>
          <TopRight />
        </div>

        {/* Page heading */}
        <div className={styles.headCard}>
          <h1 className={styles.headTitle}>Inspector Allocation</h1>
          <p className={styles.headSub}>AI-recommended field assignments</p>
        </div>

        {/* Stats row */}
        <div className={styles.statsRow}>
          {[
            { label: "Active Inspectors",   value: "8",   color: "#27ae60", icon: (
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#27ae60" strokeWidth="1.6"><circle cx="9" cy="7" r="3"/><path d="M3 20v-1a6 6 0 0 1 9.33-5"/><circle cx="17" cy="14" r="3"/><path d="M14 20v-1a3 3 0 0 1 6 0v1"/></svg>
            )},
            { label: "Assigned Projects",   value: "125", color: "#2756c5", icon: (
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#2756c5" strokeWidth="1.6"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>
            )},
            { label: "Critical Assignments",value: "13",  color: "#e74c3c", icon: (
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#e74c3c" strokeWidth="1.6"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
            )},
            { label: "Unassigned Projects",  value: "23",  color: "#94a3b8", icon: (
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" strokeWidth="1.6" strokeDasharray="3 2"><circle cx="12" cy="12" r="9"/></svg>
            )},
          ].map(({ label, value, color, icon }) => (
            <div key={label} className={styles.statCard}>
              <div>
                <div className={styles.statLabel}>{label}</div>
                <div className={styles.statValue} style={{ color }}>{value}</div>
              </div>
              <div className={styles.statIcon}>{icon}</div>
            </div>
          ))}
        </div>

        {/* Body */}
        <div className={styles.body}>

          {/* Inspector cards list */}
          <div className={styles.inspList}>
            {INSPECTORS.map(insp => (
              <InspectorCard key={insp.id} inspector={insp} />
            ))}
          </div>

          {/* Allocation overview sidebar */}
          <div className={styles.overviewPanel}>
            <div className={styles.overviewTitle}>Allocation Overview</div>
            <div className={styles.overviewList}>
              {OVERVIEW.map(item => (
                <div key={item.name} className={styles.overviewRow}>
                  <div className={styles.overviewTop}>
                    <span className={styles.overviewName}>{item.name}</span>
                    <span className={styles.overviewCount}>{item.projects} projects</span>
                  </div>
                  <div className={styles.overviewBarBg}>
                    <div
                      className={styles.overviewBar}
                      style={{ width: `${item.pct}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}
