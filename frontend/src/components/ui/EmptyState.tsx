import type { ReactNode } from "react";
import styles from "./EmptyState.module.css";

interface Props {
  title: string;
  message?: string;
  action?: ReactNode;
}

function DefaultIcon() {
  return (
    <svg className={styles.icon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
    </svg>
  );
}

// Replaces bare "No active alerts."-style text lines with a real empty
// state -- icon, a title, an explanatory message, and an optional action.
export default function EmptyState({ title, message, action }: Props) {
  return (
    <div className={styles.empty}>
      <DefaultIcon />
      <div className={styles.title}>{title}</div>
      {message && <div className={styles.message}>{message}</div>}
      {action && <div className={styles.action}>{action}</div>}
    </div>
  );
}
