"use client";

import { useEffect, useRef, useState } from "react";
import styles from "./TopRight.module.css";

interface Alert {
  id: string;
  type: "TIER_ESCALATION" | "CRITICAL_RISK";
  projectId: string;
  fromTier: string | null;
  toTier: string;
  riskScore: number;
  message: string;
  date: string;
}

export default function TopRight() {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [open, setOpen] = useState(false);
  const panelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch("/api/alerts")
      .then(res => (res.ok ? res.json() : []))
      .then(setAlerts)
      .catch(() => setAlerts([]));
  }, []);

  useEffect(() => {
    function onClickOutside(e: MouseEvent) {
      if (panelRef.current && !panelRef.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, []);

  const today = new Date();
  const dateLabel = today.toLocaleDateString("en-US", { month: "long", day: "numeric", year: "numeric" });

  return (
    <div className={styles.topRight}>
      <div className={styles.datePill}>
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
          <line x1="16" y1="2" x2="16" y2="6" />
          <line x1="8" y1="2" x2="8" y2="6" />
          <line x1="3" y1="10" x2="21" y2="10" />
        </svg>
        {dateLabel}
      </div>

      <div className={styles.alertWrap} ref={panelRef}>
        <button className={styles.iconBtn} aria-label="Notifications" onClick={() => setOpen(o => !o)}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" />
            <path d="M13.73 21a2 2 0 0 1-3.46 0" />
          </svg>
          {alerts.length > 0 && <span className={styles.alertBadge}>{alerts.length > 9 ? "9+" : alerts.length}</span>}
        </button>

        {open && (
          <div className={styles.alertPanel}>
            <div className={styles.alertPanelHeader}>Risk Alerts</div>
            {alerts.length === 0 && <div className={styles.alertEmpty}>No active alerts.</div>}
            {alerts.slice(0, 10).map(a => (
              <div key={a.id} className={styles.alertItem}>
                <span
                  className={styles.alertDot}
                  style={{ background: a.type === "CRITICAL_RISK" ? "#e74c3c" : "#f59e0b" }}
                />
                <div>
                  <div className={styles.alertMsg}>{a.message}</div>
                  <div className={styles.alertMeta}>{a.date}</div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <button className={styles.iconBtn} aria-label="Profile">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
          <circle cx="12" cy="7" r="4" />
        </svg>
      </button>
    </div>
  );
}
