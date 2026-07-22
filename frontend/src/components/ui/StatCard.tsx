"use client";

import { useEffect, useRef, useState } from "react";
import { animate } from "framer-motion";
import type { ReactNode } from "react";
import Skeleton from "./Skeleton";
import type { BadgeTone } from "./Badge";
import styles from "./StatCard.module.css";

const TONE_VAR: Record<BadgeTone, string> = {
  good: "var(--status-good-text)",
  warning: "var(--status-warning-text)",
  serious: "var(--status-serious-text)",
  critical: "var(--status-critical-text)",
  neutral: "var(--ink-700)",
  accent: "var(--accent-600)",
};

interface Props {
  label: string;
  value: number;
  icon: ReactNode;
  tone: BadgeTone;
  loading?: boolean;
}

// Replaces the ad hoc stat-card markup duplicated across Dashboard/
// Allocation/Model Validation with one reusable component: a loading
// skeleton state, and the value counting up on first render instead of
// snapping straight to its final number.
export default function StatCard({ label, value, icon, tone, loading }: Props) {
  const [display, setDisplay] = useState(0);
  const animatedOnce = useRef(false);

  useEffect(() => {
    if (loading) return;
    if (animatedOnce.current) {
      setDisplay(value);
      return;
    }
    animatedOnce.current = true;
    const controls = animate(0, value, {
      duration: 0.6,
      ease: [0.4, 0, 0.2, 1],
      onUpdate: (v) => setDisplay(Math.round(v)),
    });
    return () => controls.stop();
  }, [value, loading]);

  return (
    <div className={styles.card}>
      <div className={styles.left}>
        <div className={styles.label}>{label}</div>
        {loading ? (
          <Skeleton width={48} height="1.85rem" />
        ) : (
          <div className={styles.value} style={{ color: TONE_VAR[tone] }}>{display}</div>
        )}
      </div>
      <div className={styles.icon}>{icon}</div>
    </div>
  );
}
