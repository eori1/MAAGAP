"use client";

import { useEffect, useState } from "react";
import Sidebar from "@/components/Sidebar";
import TopRight from "@/components/TopRight";
import { getSupabaseBrowserClient } from "@/lib/supabaseBrowserClient";
import styles from "./page.module.css";

interface SessionProfile {
  email: string;
  role: "manager" | "inspector" | "admin";
  fullName: string | null;
  inspectorId: string | null;
}

const ROLE_LABELS: Record<string, string> = {
  admin: "Administrator",
  manager: "PPDO Manager",
  inspector: "Field Inspector",
};

export default function AccountPage() {
  const [profile, setProfile] = useState<SessionProfile | null>(null);
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    fetch("/api/me")
      .then((res) => (res.ok ? res.json() : null))
      .then(setProfile)
      .catch(() => setProfile(null));
  }, []);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setSuccess(false);

    if (password.length < 6) {
      setError("Password must be at least 6 characters.");
      return;
    }
    if (password !== confirmPassword) {
      setError("Passwords do not match.");
      return;
    }

    setSubmitting(true);
    const supabase = getSupabaseBrowserClient();
    const { error: updateError } = await supabase.auth.updateUser({ password });
    setSubmitting(false);

    if (updateError) {
      setError(updateError.message);
      return;
    }

    setSuccess(true);
    setPassword("");
    setConfirmPassword("");
  }

  return (
    <div className={styles.shell}>
      <Sidebar />
      <div className={styles.main}>

        <div className={styles.topbar}>
          <div className={styles.breadcrumb}>
            <span>Account</span>
            <span className={styles.sep}>/</span>
            <span className={styles.breadActive}>Settings</span>
          </div>
          <TopRight />
        </div>

        <div className={styles.body}>

          <div className={styles.card}>
            <div className={styles.cardTitle}>Account Information</div>
            <div className={styles.cardSub}>Your login details and assigned role</div>
            {profile ? (
              <>
                <div className={styles.infoRow}>
                  <span className={styles.infoLabel}>Name</span>
                  <span className={styles.infoValue}>{profile.fullName || "—"}</span>
                </div>
                <div className={styles.infoRow}>
                  <span className={styles.infoLabel}>Email</span>
                  <span className={styles.infoValue}>{profile.email}</span>
                </div>
                <div className={styles.infoRow}>
                  <span className={styles.infoLabel}>Role</span>
                  <span className={styles.infoValue}>{ROLE_LABELS[profile.role] ?? profile.role}</span>
                </div>
                {profile.inspectorId && (
                  <div className={styles.infoRow}>
                    <span className={styles.infoLabel}>Inspector ID</span>
                    <span className={styles.infoValue}>{profile.inspectorId}</span>
                  </div>
                )}
              </>
            ) : (
              <div className={styles.infoRow}><span className={styles.infoLabel}>Loading...</span></div>
            )}
          </div>

          <div className={styles.card}>
            <div className={styles.cardTitle}>Change Password</div>
            <div className={styles.cardSub}>Update the password for your account</div>

            {error && <div className={styles.error}>{error}</div>}
            {success && <div className={styles.success}>Password updated successfully.</div>}

            <form onSubmit={handleSubmit}>
              <div className={styles.field}>
                <label className={styles.label} htmlFor="password">New password</label>
                <input
                  id="password"
                  type="password"
                  required
                  minLength={6}
                  className={styles.input}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  autoComplete="new-password"
                />
              </div>
              <div className={styles.field}>
                <label className={styles.label} htmlFor="confirmPassword">Confirm new password</label>
                <input
                  id="confirmPassword"
                  type="password"
                  required
                  minLength={6}
                  className={styles.input}
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  autoComplete="new-password"
                />
              </div>
              <button type="submit" className={styles.submitBtn} disabled={submitting}>
                {submitting ? "Updating..." : "Update Password"}
              </button>
            </form>
          </div>

        </div>
      </div>
    </div>
  );
}
