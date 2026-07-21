"use client";

import { useEffect, useState } from "react";
import Sidebar from "@/components/Sidebar";
import TopRight from "@/components/TopRight";
import SubmitReportModal from "@/components/SubmitReportModal";
import styles from "./page.module.css";

/* ─── Types (mirror backend assignments.json payload) ─── */
type Priority = "HIGH" | "MEDIUM" | "LOW";
type Urgency = "VISIT ASAP" | "VISIT SOON" | "CHECK IN A WEEK" | "ROUTINE";
type Role = "manager" | "inspector" | "admin";

interface AssignedProject {
  assignmentId: string;
  projectId: string;
  name: string;
  location: string;
  type: string;
  riskScore: number;
  riskTier: "Low" | "Medium" | "High" | "Critical";
  priority: Priority;
  urgency: Urgency;
  status: "pending" | "accepted";
  hasReport: boolean;
}

interface Inspector {
  id: string;
  name: string;
  availability: string;
  vehicleAccess: boolean;
  capacity: number;
  currentWorkload: number;
  totalProjects: number;
  projects: AssignedProject[];
}

interface AssignmentData {
  generatedAt: string;
  solver: string;
  totalProjects: number;
  assignedProjects: number;
  unassignedProjects: number;
  criticalAssignments: number;
  inspectors: Inspector[];
}

const PRIORITY_COLORS: Record<Priority, { bg: string; border: string; text: string }> = {
  HIGH:   { bg: "#fff2f2", border: "#e74c3c", text: "#e74c3c" },
  MEDIUM: { bg: "#fffbf0", border: "#f59e0b", text: "#f59e0b" },
  LOW:    { bg: "#f0fff4", border: "#27ae60", text: "#27ae60" },
};

const URGENCY_COLORS: Record<Urgency, string> = {
  "VISIT ASAP":      "#e74c3c",
  "VISIT SOON":      "#f59e0b",
  "CHECK IN A WEEK": "#27ae60",
  "ROUTINE":         "#64748b",
};

const PROJECT_BG: Record<Priority, string> = {
  HIGH:   "#fff8ec",
  MEDIUM: "#fffbf0",
  LOW:    "#f8fff8",
};

const MAX_VISIBLE_PROJECTS = 5;

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
function inspectorPriority(inspector: Inspector): Priority {
  if (inspector.projects.some(p => p.priority === "HIGH")) return "HIGH";
  if (inspector.projects.some(p => p.priority === "MEDIUM")) return "MEDIUM";
  return "LOW";
}

function ProjectAction({
  project, isViewerInspector, onAccept, onSubmitReport,
}: {
  project: AssignedProject;
  isViewerInspector: boolean;
  onAccept: (assignmentId: string) => void;
  onSubmitReport: (project: AssignedProject) => void;
}) {
  if (project.hasReport) {
    return <span className={styles.statusBadgeDone}>✓ Reported</span>;
  }
  if (!isViewerInspector) {
    return (
      <span className={project.status === "accepted" ? styles.statusBadgeAccepted : styles.statusBadgePending}>
        {project.status === "accepted" ? "Accepted" : "Pending"}
      </span>
    );
  }
  if (project.status === "pending") {
    return (
      <button className={styles.rowAcceptBtn} onClick={() => onAccept(project.assignmentId)}>
        Accept
      </button>
    );
  }
  return (
    <button className={styles.rowSubmitBtn} onClick={() => onSubmitReport(project)}>
      Submit Report
    </button>
  );
}

function InspectorCard({
  inspector, isViewerInspector, onAccept, onSubmitReport,
}: {
  inspector: Inspector;
  isViewerInspector: boolean;
  onAccept: (assignmentId: string) => void;
  onSubmitReport: (project: AssignedProject) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const priority = inspectorPriority(inspector);
  const pc = PRIORITY_COLORS[priority];
  const visible = expanded ? inspector.projects : inspector.projects.slice(0, MAX_VISIBLE_PROJECTS);
  const hidden = inspector.projects.length - MAX_VISIBLE_PROJECTS;

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
              {inspector.id} · {inspector.vehicleAccess ? "Vehicle assigned" : "No vehicle"}&nbsp;
              <span style={{ color: "#2756c5", fontWeight: 700 }}>
                {inspector.totalProjects}/{inspector.capacity} visit slots used
              </span>
            </div>
          </div>
        </div>
        <div
          className={styles.priorityBadge}
          style={{ borderColor: pc.border, color: pc.text }}
        >
          {priority} PRIORITY
        </div>
      </div>

      {/* Assigned projects (sorted by risk score by the LP export) */}
      <div className={styles.projectsList}>
        {visible.map((p) => {
          const projBg = PROJECT_BG[p.priority];
          const urgColor = URGENCY_COLORS[p.urgency];
          const iconColor = p.priority === "HIGH" ? "#e74c3c" : p.priority === "MEDIUM" ? "#f59e0b" : "#27ae60";

          return (
            <div key={p.assignmentId} className={styles.projectRow} style={{ background: projBg }}>
              <ProjectIcon color={iconColor} />
              <div className={styles.projectInfo}>
                <div className={styles.projectName}>{p.name}</div>
                <div className={styles.projectMeta}>
                  {p.location} · {p.type} ·{" "}
                  <span style={{ color: urgColor, fontWeight: 700 }}>
                    {p.riskTier} risk ({(p.riskScore * 100).toFixed(0)}%)
                  </span>
                </div>
              </div>
              <div className={styles.projectUrgency}>
                <div className={styles.urgPriority} style={{ color: urgColor }}>{p.priority}</div>
                <div className={styles.urgLabel}  style={{ color: urgColor }}>{p.urgency}</div>
              </div>
              <div className={styles.projectActionCell}>
                <ProjectAction
                  project={p}
                  isViewerInspector={isViewerInspector}
                  onAccept={onAccept}
                  onSubmitReport={onSubmitReport}
                />
              </div>
            </div>
          );
        })}
        {hidden > 0 && (
          <button className={styles.editBtn} style={{ alignSelf: "flex-start" }} onClick={() => setExpanded(!expanded)}>
            {expanded ? "Show fewer" : `Show ${hidden} more assignment${hidden > 1 ? "s" : ""}`}
          </button>
        )}
      </div>
    </div>
  );
}

/* ─── Page ────────────────────────────────────────────── */
export default function AllocationPage() {
  const [data, setData] = useState<AssignmentData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [viewerRole, setViewerRole] = useState<Role | null>(null);
  const [reportModalProject, setReportModalProject] = useState<AssignedProject | null>(null);

  const loadAssignments = () => {
    fetch("/api/assignments")
      .then(res => {
        if (!res.ok) throw new Error("Assignment schedule not available");
        return res.json();
      })
      .then(setData)
      .catch(() => setError("No optimized schedule found. Run the backend pipeline (python main.py) to generate inspector assignments."));
  };

  useEffect(() => {
    loadAssignments();
    fetch("/api/me")
      .then(res => (res.ok ? res.json() : null))
      .then(profile => setViewerRole(profile?.role ?? null))
      .catch(() => setViewerRole(null));
  }, []);

  async function handleAccept(assignmentId: string) {
    const res = await fetch(`/api/assignments/${assignmentId}/accept`, { method: "PATCH" });
    if (res.ok) loadAssignments();
  }

  function handleReportSubmitted() {
    setReportModalProject(null);
    loadAssignments();
  }

  const activeInspectors = data?.inspectors.filter(i => i.totalProjects > 0).length ?? 0;
  const isViewerInspector = viewerRole === "inspector";

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
          <p className={styles.headSub}>
            {data
              ? `LP-optimized field assignments · ${data.solver} · generated ${new Date(data.generatedAt).toLocaleDateString()}`
              : "AI-recommended field assignments"}
          </p>
        </div>

        {/* Stats row */}
        <div className={styles.statsRow}>
          {[
            { label: "Active Inspectors",    value: data ? String(activeInspectors) : "—", color: "#27ae60", icon: (
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#27ae60" strokeWidth="1.6"><circle cx="9" cy="7" r="3"/><path d="M3 20v-1a6 6 0 0 1 9.33-5"/><circle cx="17" cy="14" r="3"/><path d="M14 20v-1a3 3 0 0 1 6 0v1"/></svg>
            )},
            { label: "Assigned Projects",    value: data ? String(data.assignedProjects) : "—", color: "#2756c5", icon: (
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#2756c5" strokeWidth="1.6"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>
            )},
            { label: "Critical Assignments", value: data ? String(data.criticalAssignments) : "—", color: "#e74c3c", icon: (
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#e74c3c" strokeWidth="1.6"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
            )},
            { label: "Unassigned Projects",  value: data ? String(data.unassignedProjects) : "—", color: "#94a3b8", icon: (
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

          {error && (
            <div className={styles.headCard} style={{ borderLeft: "4px solid #f59e0b" }}>
              <p className={styles.headSub}>{error}</p>
            </div>
          )}

          {data && (
            <>
              {/* Inspector cards list */}
              <div className={styles.inspList}>
                {data.inspectors
                  .filter(insp => insp.totalProjects > 0)
                  .map(insp => (
                    <InspectorCard
                      key={insp.id}
                      inspector={insp}
                      isViewerInspector={isViewerInspector}
                      onAccept={handleAccept}
                      onSubmitReport={setReportModalProject}
                    />
                  ))}
              </div>

              {/* Allocation overview sidebar */}
              <div className={styles.overviewPanel}>
                <div className={styles.overviewTitle}>Allocation Overview</div>
                <div className={styles.overviewList}>
                  {data.inspectors.map(insp => (
                    <div key={insp.id} className={styles.overviewRow}>
                      <div className={styles.overviewTop}>
                        <span className={styles.overviewName}>{insp.name}</span>
                        <span className={styles.overviewCount}>
                          {insp.totalProjects}/{insp.capacity} visits
                        </span>
                      </div>
                      <div className={styles.overviewBarBg}>
                        <div
                          className={styles.overviewBar}
                          style={{ width: `${insp.capacity > 0 ? Math.min(100, (insp.totalProjects / insp.capacity) * 100) : 0}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}

        </div>
      </div>

      {reportModalProject && (
        <SubmitReportModal
          assignmentId={reportModalProject.assignmentId}
          projectName={reportModalProject.name}
          onClose={() => setReportModalProject(null)}
          onSubmitted={handleReportSubmitted}
        />
      )}
    </div>
  );
}
