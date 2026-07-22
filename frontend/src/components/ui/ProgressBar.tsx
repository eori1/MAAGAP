import type { BadgeTone } from "./Badge";
import styles from "./ProgressBar.module.css";

interface Props {
  value: number;
  tone?: BadgeTone;
}

// Replaces the progress-bar CSS hand-rolled separately on Dashboard,
// Projects, Reports, Users, and Allocation.
export default function ProgressBar({ value, tone = "accent" }: Props) {
  const clamped = Math.max(0, Math.min(100, value));
  return (
    <span className={styles.track}>
      <span className={`${styles.fill} ${styles[tone]}`} style={{ width: `${clamped}%` }} />
    </span>
  );
}
