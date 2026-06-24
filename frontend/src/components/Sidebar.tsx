"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import styles from "./Sidebar.module.css";

const NAV_ITEMS = [
  { label: "Dashboard",        href: "/dashboard",  icon: DashboardIcon  },
  { label: "Projects",         href: "/projects",   icon: ProjectsIcon   },
  { label: "Forecast Engine",  href: "/forecast-engine", icon: ForecastIcon   },
  { label: "Allocation",       href: "/allocation", icon: AllocationIcon },
  { label: "Reports",          href: "/reports",    icon: ReportsIcon    },
  { label: "Project Timeline", href: "/timeline",   icon: TimelineIcon   },
  { label: "User Management",  href: "/users",      icon: UsersIcon      },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className={styles.sidebar}>

      {/* ── Logo ─────────────────────────────────── */}
      <div className={styles.logoArea}>
        {/* Two-tone arrow/chevron mark matching MAAGAP brand */}
        <svg className={styles.logoIcon} viewBox="0 0 34 34" fill="none">
          {/* Dark navy body */}
          <path d="M17 3L5 28h8l4-9 4 9h8L17 3z" fill="#1b3a5e"/>
          {/* Blue accent stripe */}
          <path d="M17 3l5 11.5L17 18l-5-3.5L17 3z" fill="#1264ae"/>
        </svg>
        <span className={styles.logoText}>
          MAA<span className={styles.logoTextBlue}>GAP</span>
        </span>
      </div>

      {/* ── Navigation ──────────────────────────── */}
      <nav className={styles.nav}>
        {NAV_ITEMS.map(({ label, href, icon: Icon }) => {
          const isActive = pathname === href || pathname.startsWith(href + "/");
          return (
            <Link
              key={href}
              href={href}
              className={`${styles.navItem} ${isActive ? styles.navItemActive : ""}`}
            >
              <Icon className={styles.navIcon} />
              {label}
            </Link>
          );
        })}
      </nav>

      {/* ── User Section ────────────────────────── */}
      <div className={styles.userSection}>
        <div className={styles.userRow}>
          <div className={styles.userAvatar}>RS</div>
          <div className={styles.userInfo}>
            <div className={styles.userName}>Ricardo Santos</div>
            <div className={styles.userRole}>Administrator</div>
          </div>
        </div>
        <button className={styles.logoutBtn}>Log out</button>
      </div>
    </aside>
  );
}

/* ─── SVG Icons — matching reference design ──────── */

function DashboardIcon({ className }: { className?: string }) {
  // Speedometer / gauge circle icon
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="9" />
      <path d="M12 12L8.5 8.5" />
      <circle cx="12" cy="12" r="1.5" fill="currentColor" stroke="none" />
      <path d="M6.5 17.5A7 7 0 0 1 7 7" />
      <path d="M17.5 17.5A7 7 0 0 0 17 7" />
    </svg>
  );
}

function ProjectsIcon({ className }: { className?: string }) {
  // Folder icon
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
    </svg>
  );
}

function ForecastIcon({ className }: { className?: string }) {
  // Sine-wave / graph with nodes — matches reference
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="4"  cy="12" r="1.5" fill="currentColor" stroke="none" />
      <circle cx="10" cy="7"  r="1.5" fill="currentColor" stroke="none" />
      <circle cx="16" cy="15" r="1.5" fill="currentColor" stroke="none" />
      <circle cx="20" cy="9"  r="1.5" fill="currentColor" stroke="none" />
      <path d="M4 12 Q7 4 10 7 Q13 10 16 15 Q18 19 20 9" />
    </svg>
  );
}

function AllocationIcon({ className }: { className?: string }) {
  // Person with gear / settings circle
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="9" cy="7" r="3.5" />
      <path d="M2 21v-1a7 7 0 0 1 10.5-6.07" />
      <circle cx="18" cy="17" r="3" />
      <path d="M18 14v1M18 20v1M15.27 15.27l.73.73M21 18.73l-.73-.73M14 17h1M21 17h1M15.27 18.73l.73-.73M21 15.27l-.73.73" strokeWidth="1.5" />
    </svg>
  );
}

function ReportsIcon({ className }: { className?: string }) {
  // Document / file with lines
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
      <line x1="8" y1="13" x2="16" y2="13" />
      <line x1="8" y1="17" x2="16" y2="17" />
      <line x1="8" y1="9"  x2="10" y2="9" />
    </svg>
  );
}

function TimelineIcon({ className }: { className?: string }) {
  // Calendar with grid lines
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
      <line x1="16" y1="2"  x2="16" y2="6" />
      <line x1="8"  y1="2"  x2="8"  y2="6" />
      <line x1="3"  y1="10" x2="21" y2="10" />
      <line x1="8"  y1="14" x2="8"  y2="14" strokeWidth="2.5" strokeLinecap="round" />
      <line x1="12" y1="14" x2="12" y2="14" strokeWidth="2.5" strokeLinecap="round" />
      <line x1="16" y1="14" x2="16" y2="14" strokeWidth="2.5" strokeLinecap="round" />
      <line x1="8"  y1="18" x2="8"  y2="18" strokeWidth="2.5" strokeLinecap="round" />
      <line x1="12" y1="18" x2="12" y2="18" strokeWidth="2.5" strokeLinecap="round" />
    </svg>
  );
}

function UsersIcon({ className }: { className?: string }) {
  // Person inside circle
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <circle cx="12" cy="9"  r="3"  />
      <path d="M6.5 20a5.5 5.5 0 0 1 11 0" />
    </svg>
  );
}
