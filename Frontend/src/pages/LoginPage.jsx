import { useState } from "react";
import { supabase } from "../lib/supabase";
import styles from "./LoginPage.module.css";

export default function LoginPage() {
  const [mode,     setMode]     = useState("signin"); // "signin" | "signup"
  const [email,    setEmail]    = useState("");
  const [password, setPassword] = useState("");
  const [loading,  setLoading]  = useState(false);
  const [error,    setError]    = useState("");
  const [msg,      setMsg]      = useState("");

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");
    setMsg("");
    setLoading(true);

    try {
      if (mode === "signup") {
        const { error } = await supabase.auth.signUp({ email, password });
        if (error) throw error;
        setMsg("Check your email to confirm your account.");
      } else {
        const { error } = await supabase.auth.signInWithPassword({ email, password });
        if (error) throw error;
        // App.jsx will redirect via onAuthStateChange
      }
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className={styles.wrapper}>
      <div className={styles.noise} />

      <div className={styles.panel + " fade-in"}>
        <div className={styles.logo}>
          <span className={styles.logoAccent}>EDA</span> Agent
        </div>
        <p className={styles.tagline}>
          Automated exploratory analysis, powered by GPT-4o.
        </p>

        <div className={styles.tabs}>
          <button
            className={`${styles.tab} ${mode === "signin" ? styles.tabActive : ""}`}
            onClick={() => { setMode("signin"); setError(""); setMsg(""); }}
          >
            Sign in
          </button>
          <button
            className={`${styles.tab} ${mode === "signup" ? styles.tabActive : ""}`}
            onClick={() => { setMode("signup"); setError(""); setMsg(""); }}
          >
            Sign up
          </button>
        </div>

        <form onSubmit={handleSubmit} className={styles.form}>
          <div>
            <label htmlFor="email">Email</label>
            <input
              id="email"
              type="email"
              autoComplete="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              required
            />
          </div>

          <div>
            <label htmlFor="password">Password</label>
            <input
              id="password"
              type="password"
              autoComplete={mode === "signup" ? "new-password" : "current-password"}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              required
              minLength={6}
            />
          </div>

          {error && <p className="error-msg">{error}</p>}
          {msg   && <p className={styles.success}>{msg}</p>}

          <button type="submit" className={`btn btn-primary ${styles.submitBtn}`} disabled={loading}>
            {loading ? <span className="spinner" /> : (mode === "signup" ? "Create account" : "Sign in")}
          </button>
        </form>
      </div>
    </div>
  );
}