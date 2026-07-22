"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { getSupabaseBrowserClient } from "@/lib/supabaseBrowserClient";
import type { SessionProfile } from "@/lib/supabaseSessionServer";
import styles from "./Sidebar.module.css";

type Role = SessionProfile["role"];

const ROLE_LABELS: Record<Role, string> = {
  admin: "Administrator",
  manager: "PPDO Manager",
  inspector: "Field Inspector",
};

const NAV_ITEMS: { label: string; href: string; icon: typeof DashboardIcon; excludeInspector?: boolean }[] = [
  { label: "Dashboard",        href: "/dashboard",  icon: DashboardIcon  },
  { label: "Projects",         href: "/projects",   icon: ProjectsIcon   },
  { label: "Forecast Engine",  href: "/forecast-engine", icon: ForecastIcon   },
  { label: "Allocation",       href: "/allocation", icon: AllocationIcon },
  { label: "Reports",          href: "/reports",    icon: ReportsIcon    },
  { label: "Model Validation", href: "/model-validation", icon: ValidationIcon },
  { label: "Project Timeline", href: "/timeline",   icon: TimelineIcon   },
  // Manager + Admin see everything; Inspector is scoped to their own schedule/reports only.
  { label: "User Management",  href: "/users",      icon: UsersIcon, excludeInspector: true },
];

function initials(name: string | null, email: string | undefined): string {
  const source = name?.trim() || email || "";
  const parts = source.split(/[\s.@]+/).filter(Boolean);
  return (parts[0]?.[0] ?? "").toUpperCase() + (parts[1]?.[0] ?? "").toUpperCase();
}

export default function Sidebar() {
  const pathname = usePathname();
  const router = useRouter();
  const [profile, setProfile] = useState<SessionProfile | null>(null);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(() => (typeof window !== "undefined" ? window.matchMedia("(max-width: 768px)").matches : false));

  useEffect(() => {
    fetch("/api/me")
      .then((res) => (res.ok ? res.json() : null))
      .then(setProfile)
      .catch(() => setProfile(null));
  }, []);

  useEffect(() => {
    const mq = window.matchMedia("(max-width: 768px)");
    const onChange = (e: MediaQueryListEvent) => setIsMobile(e.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);

  // Close the mobile drawer on every navigation.
  useEffect(() => {
    setMobileOpen(false);
  }, [pathname]);

  async function handleLogout() {
    const supabase = getSupabaseBrowserClient();
    await supabase.auth.signOut();
    router.push("/");
    router.refresh();
  }

  const visibleNavItems = NAV_ITEMS.filter((item) => !item.excludeInspector || profile?.role !== "inspector");

  return (
    <>
      <button
        className={styles.mobileToggle}
        aria-label={mobileOpen ? "Close navigation" : "Open navigation"}
        onClick={() => setMobileOpen((o) => !o)}
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
          {mobileOpen ? (
            <><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></>
          ) : (
            <><line x1="3" y1="6" x2="21" y2="6" /><line x1="3" y1="12" x2="21" y2="12" /><line x1="3" y1="18" x2="21" y2="18" /></>
          )}
        </svg>
      </button>

      <AnimatePresence>
        {mobileOpen && (
          <motion.div
            className={styles.backdropOpen}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.15 }}
            onClick={() => setMobileOpen(false)}
          />
        )}
      </AnimatePresence>

      <motion.aside
        className={styles.sidebar}
        initial={false}
        animate={{ x: isMobile && !mobileOpen ? "-100%" : 0 }}
        transition={{ type: "spring", stiffness: 320, damping: 32 }}
      >

        {/* ── Logo ─────────────────────────────────── */}
        <div className={styles.logoArea}>
          {/* Two-tone arrow/chevron mark matching MAAGAP brand */}
          <svg className={styles.logoIcon} viewBox="0 0 34 34" fill="none">
            {/* Dark navy body */}
            <path d="M17 3L5 28h8l4-9 4 9h8L17 3z" fill="var(--ink-900)"/>
            {/* Accent stripe */}
            <path d="M17 3l5 11.5L17 18l-5-3.5L17 3z" fill="var(--accent-600)"/>
          </svg>
          <span className={styles.logoText}>
            MAA<span className={styles.logoTextBlue}>GAP</span>
          </span>
        </div>

        {/* ── Navigation ──────────────────────────── */}
        <nav className={styles.nav}>
          {visibleNavItems.map(({ label, href, icon: Icon }) => {
            const isActive = pathname === href || pathname.startsWith(href + "/");
            return (
              <Link
                key={href}
                href={href}
                className={`${styles.navItem} ${isActive ? styles.navItemActive : ""}`}
              >
                {isActive && (
                  <motion.span
                    layoutId="nav-active-pill"
                    className={styles.navActivePill}
                    transition={{ type: "spring", stiffness: 380, damping: 32 }}
                  />
                )}
                <Icon className={styles.navIcon} />
                <span className={styles.navLabel}>{label}</span>
              </Link>
            );
          })}
        </nav>

        {/* ── User Section ────────────────────────── */}
        <div className={styles.userSection}>
          <div className={styles.userRow}>
            <div className={styles.userAvatar}>{profile ? initials(profile.fullName, profile.email) : "…"}</div>
            <div className={styles.userInfo}>
              <div className={styles.userName}>{profile?.fullName || profile?.email || "Loading..."}</div>
              <div className={styles.userRole}>{profile ? ROLE_LABELS[profile.role] : ""}</div>
            </div>
          </div>
          <button className={styles.logoutBtn} onClick={handleLogout}>Log out</button>
        </div>
      </motion.aside>
    </>
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

function ValidationIcon({ className }: { className?: string }) {
  // Target with a checkmark -- prediction vs. reality
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="12" r="8" />
      <circle cx="11" cy="12" r="4" />
      <path d="M17 6l4-2M21 4v3.5" />
      <path d="M15.5 12.5l1.5 1.5 3-3" />
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
