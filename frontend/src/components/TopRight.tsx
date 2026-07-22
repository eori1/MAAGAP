"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { motion, AnimatePresence } from "framer-motion";
import styles from "./TopRight.module.css";
import type { Alert } from "@/lib/types";
import EmptyState from "@/components/ui/EmptyState";

const ALERT_DOT_COLOR: Record<Alert["type"], string> = {
  CRITICAL_RISK: "var(--status-critical)",
  TIER_ESCALATION: "var(--status-warning)",
  REPORT_NEEDS_REVISION: "var(--info-600)",
};

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
          <AnimatePresence>
            {alerts.length > 0 && (
              <motion.span
                key={alerts.length}
                className={styles.alertBadge}
                initial={{ scale: 0.5, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.5, opacity: 0 }}
                transition={{ type: "spring", stiffness: 400, damping: 20 }}
              >
                {alerts.length > 9 ? "9+" : alerts.length}
              </motion.span>
            )}
          </AnimatePresence>
        </button>

        <AnimatePresence>
          {open && (
            <motion.div
              className={styles.alertPanel}
              initial={{ opacity: 0, y: -6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.15 }}
            >
              <div className={styles.alertPanelHeader}>Risk Alerts</div>
              {alerts.length === 0 && (
                <EmptyState title="No active alerts" message="You're all caught up." />
              )}
              {alerts.slice(0, 10).map(a => (
                <div key={a.id} className={styles.alertItem}>
                  <span
                    className={styles.alertDot}
                    style={{ background: ALERT_DOT_COLOR[a.type] }}
                  />
                  <div>
                    <div className={styles.alertMsg}>{a.message}</div>
                    <div className={styles.alertMeta}>{a.date}</div>
                  </div>
                </div>
              ))}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <Link className={styles.iconBtn} href="/account" aria-label="Account Settings">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
          <circle cx="12" cy="7" r="4" />
        </svg>
      </Link>
    </div>
  );
}
