"use client";

import { useState, useEffect, useMemo } from "react";
import { fetchProjects } from "@/lib/api";
import Sidebar from "@/components/Sidebar";
import TopRight from "@/components/TopRight";
import Skeleton from "@/components/ui/Skeleton";
import EmptyState from "@/components/ui/EmptyState";
import Badge from "@/components/ui/Badge";
import { RISK_TONE } from "@/lib/riskTone";
import styles from "./page.module.css";

interface ProjectData {
  id: string;
  name: string;
  description: string | null;
  municipality: string;
  risk: string;
  budget: string;
}

interface ShapFactor {
  feature: string;
  friendlyLabel: string;
  shapValue: number;
}

interface ProjectDetail {
  projectId: string;
  riskTier: string;
  delayProbability: number;
  costOverrunProbability: number;
  predictedDelayDays: number;
  shap: { baseValue: number; predictedContribution: number; factors: ShapFactor[] } | null;
}

type SortKey = "name" | "risk";
const RISK_ORDER: Record<string, number> = { Low: 0, Medium: 1, High: 2, Critical: 3 };

/* ─── SHAP feature-attribution bars (real per-project model output) ── */
function ShapFactors({ factors }: { factors: ShapFactor[] }) {
  const maxAbs = Math.max(...factors.map((f) => Math.abs(f.shapValue)), 0.0001);
  return (
    <>
      {factors.map((f) => {
        const increases = f.shapValue >= 0;
        const widthPct = (Math.abs(f.shapValue) / maxAbs) * 45; // half-track max
        return (
          <div key={f.feature} className={styles.shapRow}>
            <div className={styles.shapLabel}>{f.friendlyLabel}</div>
            <div className={styles.shapTrack}>
              <span className={styles.shapMid} />
              <span
                className={`${styles.shapBar} ${increases ? styles.shapBarIncrease : styles.shapBarDecrease}`}
                style={{ width: `${widthPct}%` }}
              />
            </div>
            <div className={`${styles.shapVal} ${increases ? styles.shapValIncrease : styles.shapValDecrease}`}>
              {increases ? "+" : "−"}{Math.abs(f.shapValue).toFixed(2)}
            </div>
          </div>
        );
      })}
    </>
  );
}

/* ─── Main Page ───────────────────────────────────────── */
export default function ForecastEnginePage() {
  const [projects, setProjects] = useState<ProjectData[]>([]);
  const [listLoading, setListLoading] = useState(true);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<ProjectDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [search, setSearch] = useState("");
  const [sortBy, setSortBy] = useState<SortKey>("risk");
  const [orderAsc, setOrderAsc] = useState(false);

  useEffect(() => {
    fetchProjects().then((data) => {
      setProjects(data);
      setListLoading(false);
      if (data.length > 0) setSelectedId(data[0].id);
    });
  }, []);

  useEffect(() => {
    if (!selectedId) return;
    let cancelled = false;
    setDetailLoading(true);
    setDetail(null);
    fetch(`/api/projects/${selectedId}`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => { if (!cancelled) setDetail(data); })
      .finally(() => { if (!cancelled) setDetailLoading(false); });
    return () => { cancelled = true; };
  }, [selectedId]);

  const sortedProjects = useMemo(() => {
    const q = search.trim().toLowerCase();
    const list = projects.filter((p) => !q || p.name.toLowerCase().includes(q) || p.municipality.toLowerCase().includes(q));
    const dir = orderAsc ? 1 : -1;
    list.sort((a, b) => {
      if (sortBy === "risk") return ((RISK_ORDER[a.risk] ?? 0) - (RISK_ORDER[b.risk] ?? 0)) * dir;
      return a.name.localeCompare(b.name) * dir;
    });
    return list;
  }, [projects, search, sortBy, orderAsc]);

  const selected = projects.find((p) => p.id === selectedId) ?? null;

  return (
    <div className={styles.shell}>
      <Sidebar />
      <div className={styles.main}>

        <div className={styles.topbar}>
          <div className={styles.breadcrumb}>
            <span>Forecast Engine</span>
            <span className={styles.sep}>/</span>
            <span className={styles.breadActive}>Prediction Explainability</span>
          </div>
          <TopRight />
        </div>

        <div className={styles.headCard}>
          <h1 className={styles.headTitle}>
            Forecast <span className={styles.accent}>Explainability</span> Engine
          </h1>
          <p className={styles.headSub}>Why the model thinks a project is at risk — real SHAP feature attributions, not a synthesized forecast</p>
        </div>

        <div className={styles.body}>

          {/* LEFT — project list */}
          <div className={styles.leftPanel}>
            <div className={styles.searchWrap}>
              <input
                className={styles.searchInput}
                placeholder="Search projects..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
              />
            </div>
            <div className={styles.listControls}>
              <div className={styles.ctrlGroup}>
                <span className={styles.ctrlLabel}>Sort by:</span>
                <select className={styles.ctrlSelect} value={sortBy} onChange={(e) => setSortBy(e.target.value as SortKey)}>
                  <option value="risk">Risk</option>
                  <option value="name">Name</option>
                </select>
              </div>
              <div className={styles.ctrlGroup}>
                <span className={styles.ctrlLabel}>Order:</span>
                <select className={styles.ctrlSelect} value={orderAsc ? "asc" : "desc"} onChange={(e) => setOrderAsc(e.target.value === "asc")}>
                  <option value="desc">Descending</option>
                  <option value="asc">Ascending</option>
                </select>
              </div>
            </div>

            <div className={styles.projectList}>
              {listLoading && Array.from({ length: 6 }).map((_, i) => (
                <div key={i} style={{ padding: "0 0.25rem" }}><Skeleton height={44} radius="var(--radius-md)" /></div>
              ))}

              {!listLoading && sortedProjects.map((p) => (
                <button
                  key={p.id}
                  className={`${styles.projectItem} ${selectedId === p.id ? styles.projectItemActive : ""}`}
                  onClick={() => setSelectedId(p.id)}
                >
                  <div className={styles.projectItemName}>{p.name}</div>
                  <div className={styles.projectItemBottom}>
                    <span className={styles.projectItemMuni}>
                      <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M12 21s-8-7.3-8-13a8 8 0 1 1 16 0c0 5.7-8 13-8 13z" /><circle cx="12" cy="8" r="3" /></svg>
                      {p.municipality}
                    </span>
                    <Badge tone={RISK_TONE[p.risk] ?? "neutral"}>{p.risk}</Badge>
                  </div>
                </button>
              ))}

              {!listLoading && sortedProjects.length === 0 && (
                <EmptyState title="No projects match" message="Try a different search term." />
              )}
            </div>
          </div>

          {/* RIGHT — explainability panel */}
          <div className={styles.rightPanel}>
            {!selected ? (
              <div className={styles.summaryCard}>
                <EmptyState title="Select a project" message="Choose a project from the list to see its risk explanation." />
              </div>
            ) : (
              <>
                <div className={styles.summaryCard}>
                  <div className={styles.summaryTop}>
                    <div>
                      <div className={styles.summaryTitle}>{selected.name}</div>
                      <div className={styles.summaryMeta}>{selected.id} · {selected.municipality} · {selected.budget}</div>
                      {selected.description && <div className={styles.summaryDescription}>{selected.description}</div>}
                    </div>
                    <Badge tone={RISK_TONE[selected.risk] ?? "neutral"}>
                      {selected.risk === "Pending" ? "Pending Assessment" : `${selected.risk} Risk`}
                    </Badge>
                  </div>

                  {detailLoading ? (
                    <div className={styles.statGrid}>
                      <Skeleton height={64} radius="var(--radius-md)" />
                      <Skeleton height={64} radius="var(--radius-md)" />
                      <Skeleton height={64} radius="var(--radius-md)" />
                    </div>
                  ) : detail ? (
                    <>
                      <div className={styles.statGrid}>
                        <div className={styles.statBox}>
                          <div className={styles.statBoxLabel}>Delay Probability</div>
                          <div className={styles.statBoxValue}>{Math.round(detail.delayProbability * 100)}%</div>
                        </div>
                        <div className={styles.statBox}>
                          <div className={styles.statBoxLabel}>Predicted Delay</div>
                          <div className={styles.statBoxValue}>{Math.round(detail.predictedDelayDays)}d</div>
                        </div>
                        <div className={styles.statBox}>
                          <div className={styles.statBoxLabel}>Cost Overrun Risk</div>
                          <div className={styles.statBoxValue}>{Math.round(detail.costOverrunProbability * 100)}%</div>
                        </div>
                      </div>

                      {(detail.riskTier === "High" || detail.riskTier === "Critical") && (
                        <div className={`${styles.riskCallout} ${detail.riskTier === "Critical" ? styles.riskCalloutCritical : styles.riskCalloutSerious}`}>
                          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" style={{ flexShrink: 0 }}><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" /><line x1="12" y1="9" x2="12" y2="13" /><line x1="12" y1="17" x2="12.01" y2="17" /></svg>
                          This project is predicted to be delayed by approximately {Math.round(detail.predictedDelayDays)} days ({Math.round(detail.delayProbability * 100)}% delay probability).
                        </div>
                      )}
                    </>
                  ) : null}
                </div>

                {detailLoading ? (
                  <div className={styles.shapCard}>
                    <div className={styles.shapTitle}>Top Contributing Factors</div>
                    <div className={styles.shapSub}>SHAP feature attribution for this prediction — how much each real factor pushed the risk score up or down</div>
                    <div style={{ display: "flex", flexDirection: "column", gap: "0.6rem" }}>
                      {Array.from({ length: 5 }).map((_, i) => <Skeleton key={i} height={16} />)}
                    </div>
                  </div>
                ) : !detail ? (
                  <div className={styles.shapCard}>
                    <EmptyState title="Not yet assessed" message="This project hasn't been scored by a pipeline run yet — it will get a full risk analysis (including SHAP feature attributions) the next time python main.py runs." />
                  </div>
                ) : (
                  <div className={styles.shapCard}>
                    <div className={styles.shapTitle}>Top Contributing Factors</div>
                    <div className={styles.shapSub}>SHAP feature attribution for this prediction — how much each real factor pushed the risk score up or down</div>
                    {detail.shap && detail.shap.factors.length > 0 ? (
                      <ShapFactors factors={detail.shap.factors} />
                    ) : (
                      <EmptyState title="No explanation available" message="This project doesn't have SHAP attribution data yet." />
                    )}
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
