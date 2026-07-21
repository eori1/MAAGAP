"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { getSupabaseBrowserClient } from "@/lib/supabaseBrowserClient";
import styles from "./page.module.css";

type Status = "checking" | "ready" | "invalid" | "success";

// Reached via the emailed password-reset link. The Supabase browser client
// exchanges the link's one-time code for a temporary "recovery" session as
// soon as it loads (before this component can read it), so we listen for
// the PASSWORD_RECOVERY auth event -- and also check for an already-active
// session in case the event fired before the listener attached.
export default function ResetPasswordPage() {
  const router = useRouter();
  const [status, setStatus] = useState<Status>("checking");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    const supabase = getSupabaseBrowserClient();

    const { data: listener } = supabase.auth.onAuthStateChange((event) => {
      if (event === "PASSWORD_RECOVERY") setStatus("ready");
    });

    supabase.auth.getSession().then(({ data }) => {
      if (data.session) setStatus((s) => (s === "checking" ? "ready" : s));
    });

    const timeout = setTimeout(() => {
      setStatus((s) => (s === "checking" ? "invalid" : s));
    }, 5000);

    return () => {
      listener.subscription.unsubscribe();
      clearTimeout(timeout);
    };
  }, []);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);

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

    setStatus("success");
    setTimeout(() => {
      router.push("/dashboard");
      router.refresh();
    }, 1500);
  }

  return (
    <div className={styles.shell}>
      <div className={styles.card}>
        <div className={styles.logoRow}>
          <svg width="30" height="30" viewBox="0 0 34 34" fill="none">
            <path d="M17 3L5 28h8l4-9 4 9h8L17 3z" fill="#1b3a5e" />
            <path d="M17 3l5 11.5L17 18l-5-3.5L17 3z" fill="#1264ae" />
          </svg>
          <span className={styles.logoText}>
            MAA<span className={styles.logoTextBlue}>GAP</span>
          </span>
        </div>

        <h1 className={styles.title}>Reset your password</h1>
        <p className={styles.subtitle}>Choose a new password for your account.</p>

        {status === "checking" && (
          <p className={styles.statusText}>Verifying your reset link...</p>
        )}

        {status === "invalid" && (
          <>
            <div className={styles.error}>
              This reset link is invalid or has expired. Request a new one from the login page.
            </div>
            <a href="/" className={styles.link}>&larr; Back to login</a>
          </>
        )}

        {status === "success" && (
          <div className={styles.success}>Password updated. Redirecting you to the dashboard...</div>
        )}

        {status === "ready" && (
          <form onSubmit={handleSubmit}>
            {error && <div className={styles.error}>{error}</div>}

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
        )}
      </div>
    </div>
  );
}
