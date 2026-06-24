"use client";

import { useState } from "react";
import Sidebar from "@/components/Sidebar";
import TopRight from "@/components/TopRight";
import styles from "./page.module.css";

import {
  ComposedChart, Line, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from "recharts";

/* ─── Mock project data ───────────────────────────────── */
const WEEKS = ["W1","W2","W3","W4","W5","W6","W7","W8"];

function makeChartData(history: number[], forecast: number[]) {
  return WEEKS.map((week, i) => ({
    week,
    actual:   history[i]  ?? null,
    forecast: forecast[i] ?? null,
  }));
}

const PROJECTS = [
  { id:"p1",  name:"Aganan Flyover, Brgy. Aganan",          municipality:"Pavia",       status:"Delayed",    delayProb:87, projectedDelay:67, costRisk:25, confidence:97,  history:[0,3,7,12,16,20],   forecast:[0,3,7,12,16,20,25,30]  },
  { id:"p2",  name:"Bamboo Flyover, Brgy. Bamboo",           municipality:"Iloilo City", status:"On Schedule",delayProb:22, projectedDelay:0,  costRisk:8,  confidence:88,  history:[0,5,11,18,26,35],  forecast:[0,5,11,18,26,35,42,50] },
  { id:"p3",  name:"Coco Flyover, Brgy. Coco",               municipality:"San Miguel",  status:"Delayed",    delayProb:74, projectedDelay:42, costRisk:18, confidence:91,  history:[0,2,5,9,13,16],    forecast:[0,2,5,9,13,16,21,28]   },
  { id:"p4",  name:"Dahlia Flyover, Brgy. Dahlia",           municipality:"Leganes",     status:"Completed",  delayProb:5,  projectedDelay:0,  costRisk:2,  confidence:99,  history:[0,8,17,28,40,50],  forecast:[0,8,17,28,40,50,50,50] },
  { id:"p5",  name:"Eucalyptus Flyover, Brgy. Eucalyptus",   municipality:"Balasan",     status:"On Schedule",delayProb:18, projectedDelay:0,  costRisk:5,  confidence:85,  history:[0,6,13,21,29,38],  forecast:[0,6,13,21,29,38,46,55] },
  { id:"p6",  name:"Fern Flyover, Brgy. Fern",               municipality:"Cabatuan",    status:"Delayed",    delayProb:61, projectedDelay:28, costRisk:14, confidence:82,  history:[0,4,8,13,17,20],   forecast:[0,4,8,13,17,20,26,33]  },
  { id:"p7",  name:"Maple Avenue, Brgy. Maple",              municipality:"Santiago",    status:"On Time",    delayProb:12, projectedDelay:0,  costRisk:4,  confidence:90,  history:[0,7,15,24,33,43],  forecast:[0,7,15,24,33,43,52,60] },
  { id:"p8",  name:"Cedar Lane, Brgy. Cedar",                municipality:"Banawe",      status:"Cancelled",  delayProb:0,  projectedDelay:0,  costRisk:0,  confidence:100, history:[0,3,7,10,10,10],   forecast:[0,3,7,10,10,10,10,10]  },
  { id:"p9",  name:"Oak Street, Brgy. Oak",                  municipality:"Lamut",       status:"Delayed",    delayProb:79, projectedDelay:55, costRisk:31, confidence:94,  history:[0,2,4,7,11,14],    forecast:[0,2,4,7,11,14,20,28]   },
  { id:"p10", name:"Pine Hill, Brgy. Pine",                  municipality:"Tarnugan",    status:"On Time",    delayProb:9,  projectedDelay:0,  costRisk:3,  confidence:87,  history:[0,9,18,28,38,48],  forecast:[0,9,18,28,38,48,57,65] },
  { id:"p11", name:"Willow Way, Brgy. Willow",               municipality:"Cabeza",      status:"Delayed",    delayProb:83, projectedDelay:72, costRisk:28, confidence:96,  history:[0,3,6,9,13,16],    forecast:[0,3,6,9,13,16,22,30]   },
];

const STATUS_COLOR: Record<string,string> = {
  "Delayed":"#e74c3c","On Schedule":"#f39c12",
  "On Time":"#27ae60","Completed":"#27ae60",
  "Cancelled":"#95a5a6","In Progress":"#2756c5",
};

/* ─── Custom Tooltip ──────────────────────────────────── */
function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background:"#fff", border:"1px solid #dde4f0",
      borderRadius:8, padding:"0.5rem 0.75rem",
      boxShadow:"0 4px 14px rgba(0,0,0,0.12)",
      fontSize:"0.75rem", fontFamily:"Inter,system-ui,sans-serif",
    }}>
      <div style={{ fontWeight:700, color:"#1b3a5e", marginBottom:4 }}>{label}</div>
      {payload.map((p: any) => (
        <div key={p.name} style={{ color: p.color, fontWeight:600 }}>
          {p.name === "actual" ? "Actual" : "AI Forecast"}: {p.value}%
        </div>
      ))}
    </div>
  );
}

/* ─── Progress Chart using Recharts ──────────────────── */
function ProgressChart({ history, forecast }: { history: number[]; forecast: number[] }) {
  const data = makeChartData(history, forecast);
  const lastActualIdx = history.length - 1;

  return (
    <ResponsiveContainer width="100%" height={200}>
      <ComposedChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id="actualGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%"  stopColor="#3b82f6" stopOpacity={0.3} />
            <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}   />
          </linearGradient>
        </defs>

        <CartesianGrid stroke="#eaeff6" strokeDasharray="3 0" vertical={false} />

        <XAxis
          dataKey="week"
          tick={{ fontSize: 11, fill: "#94a3b8", fontFamily:"Inter,system-ui" }}
          axisLine={false} tickLine={false}
        />
        <YAxis
          tick={{ fontSize: 11, fill: "#94a3b8", fontFamily:"Inter,system-ui" }}
          axisLine={false} tickLine={false}
          tickFormatter={(v) => `${v}`}
          width={32}
        />

        <Tooltip content={<CustomTooltip />} />

        {/* Divider between actual & forecast */}
        <ReferenceLine
          x={WEEKS[lastActualIdx]}
          stroke="#cbd5e1"
          strokeDasharray="4 3"
          label={{ value:"Now", position:"top", fontSize:10, fill:"#94a3b8" }}
        />

        {/* Area under actual */}
        <Area
          type="monotone"
          dataKey="actual"
          name="actual"
          stroke="#3b82f6"
          strokeWidth={2.5}
          fill="url(#actualGrad)"
          dot={{ r: 4, fill:"#fff", stroke:"#3b82f6", strokeWidth:2 }}
          activeDot={{ r: 6 }}
          connectNulls={false}
        />

        {/* Forecast line (dashed) */}
        <Line
          type="monotone"
          dataKey="forecast"
          name="forecast"
          stroke="#f59e0b"
          strokeWidth={2.5}
          strokeDasharray="6 4"
          dot={false}
          activeDot={{ r: 5, fill:"#f59e0b" }}
          connectNulls={false}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

/* ─── Main Page ───────────────────────────────────────── */
export default function ForecastEnginePage() {
  const [selected, setSelected] = useState(PROJECTS[0]);
  const [sortBy,   setSortBy]   = useState("Most Recent");
  const [orderBy,  setOrderBy]  = useState("Ascending");

  const confColor = selected.confidence >= 85 ? "#27ae60" : "#f59e0b";

  return (
    <div className={styles.shell}>
      <Sidebar />
      <div className={styles.main}>

        {/* ── Top bar ── */}
        <div className={styles.topbar}>
          <div className={styles.breadcrumb}>
            <span>Forecast Engine</span>
            <span className={styles.sep}>/</span>
            <span className={styles.breadActive}>Intelligence Engine</span>
          </div>
          <TopRight />
        </div>

        {/* ── Page heading card ── */}
        <div className={styles.headCard}>
          <h1 className={styles.headTitle}>
            Forecast <span className={styles.accent}>Intelligence</span> Engine
          </h1>
          <p className={styles.headSub}>AI-driven delay probability · XGBoost + LSTM forecasting models</p>
        </div>

        {/* ── Body ── */}
        <div className={styles.body}>

          {/* LEFT — project list */}
          <div className={styles.leftPanel}>
            <div className={styles.listHeader}>Select Project to Forecast</div>
            <div className={styles.listControls}>
              <div className={styles.ctrlGroup}>
                <span className={styles.ctrlLabel}>Sort by:</span>
                <select className={styles.ctrlSelect} value={sortBy} onChange={e => setSortBy(e.target.value)}>
                  <option>Most Recent</option><option>Name</option><option>Risk</option>
                </select>
              </div>
              <div className={styles.ctrlGroup}>
                <span className={styles.ctrlLabel}>Order by:</span>
                <select className={styles.ctrlSelect} value={orderBy} onChange={e => setOrderBy(e.target.value)}>
                  <option>Ascending</option><option>Descending</option>
                </select>
              </div>
            </div>

            <div className={styles.projectList}>
              {PROJECTS.map(p => (
                <button
                  key={p.id}
                  className={`${styles.projectItem} ${selected.id === p.id ? styles.projectItemActive : ""}`}
                  onClick={() => setSelected(p)}
                >
                  <div className={styles.projectItemName}>{p.name}</div>
                  <div className={styles.projectItemBottom}>
                    <span className={styles.projectItemMuni}>
                      <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M12 21s-8-7.3-8-13a8 8 0 1 1 16 0c0 5.7-8 13-8 13z"/><circle cx="12" cy="8" r="3"/></svg>
                      {p.municipality}
                    </span>
                    <span className={styles.badge} style={{ background: STATUS_COLOR[p.status] ?? "#999" }}>
                      {p.status}
                    </span>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* RIGHT — forecast panel */}
          <div className={styles.rightPanel}>

            {/* Blue forecast card */}
            <div className={styles.forecastCard}>
              <div className={styles.fcTitle}>AI Forecast – {selected.name}</div>
              <div className={styles.fcMetrics}>
                <div className={styles.fcMetric}>
                  <div className={styles.fcMetricLabel}>Delay Probability</div>
                  <div className={styles.fcMetricValue}>{selected.delayProb}%</div>
                  <div className={styles.fcMetricSub}>Confidence Level</div>
                </div>
                <div className={styles.fcMetric}>
                  <div className={styles.fcMetricLabel}>Projected Delay</div>
                  <div className={styles.fcMetricValue}>
                    {selected.projectedDelay > 0 ? `+${selected.projectedDelay}d` : "On Track"}
                  </div>
                  <div className={styles.fcMetricSub}>Confidence Level</div>
                </div>
              </div>
              {selected.projectedDelay > 0 && (
                <div className={styles.warningBanner}>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
                  Project is already delayed for {Math.round(selected.projectedDelay * 0.22)} days and likely will be delayed by {Math.round(selected.projectedDelay * 0.1)} more days.&nbsp;({selected.delayProb}% Confidence)
                </div>
              )}
            </div>

            {/* Stats row */}
            <div className={styles.statsRow}>
              <div className={styles.statCard}>
                <div className={styles.statLabel}>Cost Overrun Risk</div>
                <div className={styles.statValue} style={{ color:"#e74c3c" }}>{selected.costRisk}%</div>
              </div>
              <div className={styles.statCard}>
                <div className={styles.statLabel}>Confidence</div>
                <div className={styles.statValue} style={{ color: confColor }}>{selected.confidence}%</div>
              </div>
            </div>

            {/* Chart card */}
            <div className={styles.chartCard}>
              <div className={styles.chartTitle}>Historical Progress vs. Forecast</div>
              <div className={styles.chartSub}>Actual vs. AI Projected Trajectory</div>
              <div className={styles.chartLegend}>
                <span className={styles.legendItem}>
                  <span className={styles.legendDot} style={{ background:"#3b82f6" }} />
                  Actual Progress
                </span>
                <span className={styles.legendItem}>
                  <span style={{ display:"inline-block", width:16, borderTop:"2.5px dashed #f59e0b", marginRight:4 }} />
                  AI Forecast
                </span>
              </div>
              <ProgressChart history={selected.history} forecast={selected.forecast} />
            </div>

          </div>
        </div>
      </div>
    </div>
  );
}
