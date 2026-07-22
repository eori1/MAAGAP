import type { ReactNode } from "react";
import styles from "./Badge.module.css";

export type BadgeTone = "good" | "warning" | "serious" | "critical" | "neutral" | "accent";

interface Props {
  tone: BadgeTone;
  children: ReactNode;
}

// Single reusable status/tone pill -- replaces the local STATUS_STYLE/
// RISK_TIER_STYLE lookup-object-plus-inline-style pattern each page
// (Reports, Allocation, Model Validation) currently hand-rolls separately.
export default function Badge({ tone, children }: Props) {
  return <span className={`${styles.badge} ${styles[tone]}`}>{children}</span>;
}
