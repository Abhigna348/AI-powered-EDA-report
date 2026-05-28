import { useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "../lib/supabase";
import api from "../lib/api";
import styles from "./Dashboard.module.css";

const ACCEPTED = [".csv", ".tsv", ".xlsx", ".xls", ".json", ".parquet"];

export default function Dashboard() {
  const navigate = useNavigate();
  const [file,     setFile]     = useState(null);
  const [dragging, setDragging] = useState(false);
  const [loading,  setLoading]  = useState(false);
  const [progress, setProgress] = useState("");
  const [error,    setError]    = useState("");

  async function handleSignOut() {
    await supabase.auth.signOut();
  }

  function validateFile(f) {
    const ext = "." + f.name.split(".").pop().toLowerCase();
    if (!ACCEPTED.includes(ext)) {
      setError(`Unsupported type "${ext}". Accepted: ${ACCEPTED.join(", ")}`);
      return false;
    }
    if (f.size > 50 * 1024 * 1024) {
      setError("File must be under 50 MB.");
      return false;
    }
    setError("");
    return true;
  }

  function onFileChange(e) {
    const f = e.target.files?.[0];
    if (f && validateFile(f)) setFile(f);
  }

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f && validateFile(f)) setFile(f);
  }, []);

  async function handleAnalyze() {
    if (!file) return;
    setLoading(true);
    setProgress("Uploading and parsing file…");
    setError("");

    try {
      const formData = new FormData();
      formData.append("file", file);

      setProgress("Running EDA engine…");
      const { data } = await api.post("/api/analyze", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setProgress("Generating AI narrative…");
      // Store results in sessionStorage so ResultsPage can read them
      sessionStorage.setItem("eda_results", JSON.stringify(data));
      navigate("/results");
    } catch (err) {
      const detail = err.response?.data?.detail || err.message;
      setError(`Analysis failed: ${detail}`);
    } finally {
      setLoading(false);
      setProgress("");
    }
  }

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <div className={styles.headerLogo}>
          <span className={styles.accent}>EDA</span> Agent
        </div>
        <button className="btn btn-secondary" onClick={handleSignOut}>
          Sign out
        </button>
      </header>

      <main className={styles.main}>
        <div className={styles.hero}>
          <h1 className={styles.title}>
            Drop your dataset.<br />
            <em>We'll handle the rest.</em>
          </h1>
          <p className={styles.subtitle}>
            Upload a CSV, Excel, JSON, or Parquet file and get an instant
            AI-powered EDA report — stats, charts, outliers, correlations,
            and a GPT-4o narrative.
          </p>
        </div>

        {/* Drop zone */}
        <div
          className={`${styles.dropzone} ${dragging ? styles.dragging : ""} ${file ? styles.hasFile : ""}`}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => document.getElementById("fileInput").click()}
        >
          <input
            id="fileInput"
            type="file"
            accept={ACCEPTED.join(",")}
            style={{ display: "none" }}
            onChange={onFileChange}
          />
          {file ? (
            <>
              <div className={styles.fileIcon}>📄</div>
              <p className={styles.fileName}>{file.name}</p>
              <p className={styles.fileSize}>{(file.size / 1024).toFixed(1)} KB</p>
              <p className={styles.fileHint}>Click or drop to replace</p>
            </>
          ) : (
            <>
              <div className={styles.uploadIcon}>⬆</div>
              <p className={styles.dropText}>
                <strong>Click to browse</strong> or drag & drop
              </p>
              <p className={styles.dropHint}>{ACCEPTED.join("  ·  ")}</p>
            </>
          )}
        </div>

        {error && <p className="error-msg" style={{ textAlign: "center" }}>{error}</p>}

        {loading && (
          <div className={styles.progressBar}>
            <div className={styles.progressFill} />
            <p className={styles.progressText}>{progress}</p>
          </div>
        )}

        <button
          className={`btn btn-primary ${styles.analyzeBtn}`}
          disabled={!file || loading}
          onClick={handleAnalyze}
        >
          {loading ? <><span className="spinner" /> Analyzing…</> : "Analyze dataset →"}
        </button>

        <div className={styles.features}>
          {[
            ["📊", "Descriptive stats", "Mean, median, std, skew, kurtosis per column"],
            ["🔍", "Outlier detection", "IQR & Z-score methods with severity ratings"],
            ["🔗", "Correlation matrix", "Pearson r with flagged strong pairs"],
            ["🤖", "AI narrative", "GPT-4o synthesises findings into plain English"],
            ["💬", "Chat interface", "Ask follow-up questions about your data"],
            ["📥", "Markdown report", "Download a complete analysis report"],
          ].map(([icon, title, desc]) => (
            <div key={title} className={styles.featureCard}>
              <span className={styles.featureIcon}>{icon}</span>
              <strong>{title}</strong>
              <span className={styles.featureDesc}>{desc}</span>
            </div>
          ))}
        </div>
      </main>
    </div>
  );
}