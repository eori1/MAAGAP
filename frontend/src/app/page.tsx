"use client";

import styles from "./page.module.css";
import { useState } from "react";
import { useRouter } from "next/navigation";

/* ─── SVG Icons ─────────────────────────────────── */
const EmailIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="2" y="4" width="20" height="16" rx="2" />
    <path d="M2 7l10 7 10-7" />
  </svg>
);

const LockIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
    <path d="M7 11V7a5 5 0 0 1 10 0v4" />
  </svg>
);

const EyeIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
    <circle cx="12" cy="12" r="3" />
  </svg>
);

const EyeOffIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94" />
    <path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19" />
    <line x1="1" y1="1" x2="23" y2="23" />
  </svg>
);

/* ─── Page ───────────────────────────────────────── */
export default function LoginPage() {
  const router = useRouter();
  const [showPassword, setShowPassword] = useState(false);
  const [email, setEmail] = useState("admin@iloilo.gov.ph");
  const [password, setPassword] = useState("maagap2026");

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();
    router.push("/dashboard");
  };

  return (
    <main className={styles.container}>

      {/* ════ LEFT PANEL ════ */}
      <section className={styles.leftPanel}>
        <div className={styles.leftMain}>
          <div className={styles.leftContent}>

            {/* Logo Row — logo image to be added later */}
            <div className={styles.logoRow}>
              <div className={styles.logoPlaceholder}>LOGO<br/>SOON</div>
              <span className={styles.logoText}>
                <span className={styles.logoTextDark}>MAAG</span>
                <span className={styles.logoTextLight}>AP</span>
              </span>
            </div>

            <h1 className={styles.subtitle}>
              Predictive Intelligence for Smarter Provincial Governance.
            </h1>

            <p className={styles.description}>
              MAAGAP is an AI-driven governance platform that helps government agencies monitor,
              predict, and optimize public projects using predictive analytics, geospatial
              intelligence, and explainable AI.
            </p>

            <div className={styles.buttonGroup}>
              <button id="btn-get-started" className={styles.btnPrimary}>Get Started</button>
              <button id="btn-contact-support" className={styles.btnSecondary}>Contact Support</button>
            </div>
          </div>
        </div>

        <div className={styles.leftFooter}>
          <p className={styles.leftFooterText}>
            This system is for authorized Iloilo Provincial Government personnel only. All activities are logged and monitored. Unauthorized access attempts are subject to legal action.<br />
            © 2026 MAAGAP. All Rights Reserved. This Decision Support System was developed by Thesis Neighbors. WVSU.
          </p>
        </div>
      </section>

      {/* ════ RIGHT PANEL ════ */}
      <section className={styles.rightPanel}>
        <div className={styles.cardWrapper}>
          <div className={styles.glassCard}>

            <h2 className={styles.cardTitle}>Welcome Back!</h2>
            <p className={styles.cardSubtitle}>
              Access real-time governance insights and predictive project analytics.
            </p>

            <form onSubmit={handleLogin}>
              {/* Email */}
              <div className={styles.formGroup}>
                <label htmlFor="email" className={styles.formLabel}>Email Address</label>
                <div className={styles.inputWrapper}>
                  <span className={styles.inputIconLeft}><EmailIcon /></span>
                  <input
                    id="email"
                    type="email"
                    className={styles.input}
                    placeholder="emailexample@iloilo.gov.ph"
                    autoComplete="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                  />
                </div>
              </div>

              {/* Password */}
              <div className={styles.formGroup}>
                <label htmlFor="password" className={styles.formLabel}>Password</label>
                <div className={styles.inputWrapper}>
                  <span className={styles.inputIconLeft}><LockIcon /></span>
                  <input
                    id="password"
                    type={showPassword ? "text" : "password"}
                    className={styles.input}
                    placeholder="••••••••••••••"
                    autoComplete="current-password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                  />
                  <button
                    type="button"
                    className={styles.inputIconRight}
                    onClick={() => setShowPassword((p) => !p)}
                    aria-label="Toggle password visibility"
                  >
                    {showPassword ? <EyeOffIcon /> : <EyeIcon />}
                  </button>
                </div>
                <div className={styles.forgotRow}>
                  <button type="button" id="btn-forgot-password" className={styles.forgotLink}>
                    Forgot Password?
                  </button>
                </div>
              </div>

              <button id="btn-continue" type="submit" className={styles.btnSubmit}>
                Continue →
              </button>
            </form>

            <p className={styles.disclaimer}>
              By clicking &quot;Log In,&quot; you acknowledge this system is for authorized use only and agree
              to abide by the{" "}
              <span className={styles.disclaimerLink}>MAAGAP Terms and Conditions</span>{" "}
              regarding data handling and confidentiality.
            </p>
          </div>
        </div>

        <div className={styles.rightFooter}>
          <p className={styles.rightFooterText}>
            <strong>developed by Thesis Neighbors. WVSU.</strong>
          </p>
        </div>
      </section>

    </main>
  );
}
