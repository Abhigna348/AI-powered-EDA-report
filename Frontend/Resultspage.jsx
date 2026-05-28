import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import Plot from "react-plotly.js";
import api from "../lib/api";
import { supabase } from "../lib/supabase";
import styles from "./ResultsPage.module.css";

const TABS = ["Overview", "Charts", "Flags", "Schema", "Chat", "Report"];

export default function ResultsPage() {
  const navigate  = useNavigate();
  const [results, setResults]  = useState(null);
  const [tab,     setTab]      = useState("Overview");

  useEffect(() => {
    const raw = sessionStorage.getItem("eda_results");
    if (!raw) { navigate("/dashboard"); return; }
    setResults(JSON.parse(raw));
  }, [navigate]);

  if (!results) return null;

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <button className="btn btn-secondary" onClick={() => navigate("/dashboard")}>
          ← New analysis
        </button>
        <span className={styles.filename}>{results.filename}</span>
        <button className="btn btn-secondary" onClick={() => supabase.auth.signOut()}>
          Sign out
        </button>
      </header>

      {/* Stats bar */}
      <div className={styles.statsBar}>
        {[
          ["Rows",       results.shape.rows.toLocaleString()],
          ["Columns",    results.shape.cols],
          ["Flags",      results.eda.flags.length],
          ["Agent steps", results.tools_run.length],
        ].map(([label, value]) => (
          <div key={label} className={styles.stat}>
            <span className={styles.statValue}>{value}</span>
            <span className={styles.statLabel}>{label}</span>
          </div>
        ))}
      </div>

      {/* Tab bar */}
      <nav className={styles.tabs}>
        {TABS.map((t) => (
          <button
            key={t}
            className={`${styles.tabBtn} ${tab === t ? styles.tabActive : ""}`}
            onClick={() => setTab(t)}
          >
            {t}
          </button>
        ))}
      </nav>

      <main className={styles.content}>
        {tab === "Overview"  && <OverviewTab  results={results} />}
        {tab === "Charts"    && <ChartsTab    charts={results.charts} />}
        {tab === "Flags"     && <FlagsTab     flags={results.eda.flags} />}
        {tab === "Schema"    && <SchemaTab    schema={results.schema_summary} />}
        {tab === "Chat"      && <ChatTab      results={results} />}
        {tab === "Report"    && <ReportTab    markdown={results.report_md} />}
      </main>
    </div>
  );
}

// ── Overview ──────────────────────────────────────────────────────
function OverviewTab({ results }) {
  return (
    <div className={styles.overviewGrid}>
      <div className={`card ${styles.narrativeCard}`}>
        <h2 className={styles.sectionTitle}>AI Narrative</h2>
        <p className={styles.toolsRun}>
          Tools run: <code>{results.tools_run.join(" → ")}</code>
        </p>
        <div className={styles.narrative}>{results.narrative}</div>
      </div>

      <div className={`card ${styles.statsCard}`}>
        <h2 className={styles.sectionTitle}>Descriptive Statistics</h2>
        {results.eda.descriptive.length > 0 ? (
          <div className={styles.tableWrap}>
            <table className={styles.table}>
              <thead>
                <tr>
                  {Object.keys(results.eda.descriptive[0]).map((k) => (
                    <th key={k}>{k}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {results.eda.descriptive.map((row, i) => (
                  <tr key={i}>
                    {Object.values(row).map((v, j) => (
                      <td key={j}>{typeof v === "number" ? v.toFixed(4) : String(v)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className={styles.empty}>No numeric columns found.</p>
        )}
      </div>

      {results.eda.missing.length > 0 && (
        <div className={`card ${styles.missingCard}`}>
          <h2 className={styles.sectionTitle}>Missing Values</h2>
          <div className={styles.tableWrap}>
            <table className={styles.table}>
              <thead>
                <tr><th>Column</th><th>Missing</th><th>%</th><th>Recommendation</th></tr>
              </thead>
              <tbody>
                {results.eda.missing.map((row) => (
                  <tr key={row.column}>
                    <td><code>{row.column}</code></td>
                    <td>{row.missing}</td>
                    <td>{row.pct}%</td>
                    <td>{row.recommendation}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {results.eda.correlation.strong_pairs?.length > 0 && (
        <div className={`card ${styles.corrCard}`}>
          <h2 className={styles.sectionTitle}>Strong Correlations</h2>
          <div className={styles.pairGrid}>
            {results.eda.correlation.strong_pairs.map((p, i) => (
              <div key={i} className={styles.pairChip}>
                <code>{p.col_a}</code>
                <span className={styles.arrow}>↔</span>
                <code>{p.col_b}</code>
                <span className={`badge ${p.strength === "strong" ? "badge-high" : "badge-medium"}`}>
                  r = {p.pearson_r}
                </span>
                <span className={styles.pairDir}>{p.direction}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Charts ────────────────────────────────────────────────────────
function ChartsTab({ charts }) {
  const entries = Object.entries(charts).filter(([, v]) => v !== null);
  if (!entries.length) return <p className={styles.empty}>No charts generated.</p>;

  return (
    <div className={styles.chartsGrid}>
      {entries.map(([key, chartData]) => (
        <div key={key} className={`card ${styles.chartCard}`}>
          <Plot
            data={chartData.data}
            layout={{
              ...chartData.layout,
              paper_bgcolor: "transparent",
              plot_bgcolor:  "transparent",
              font: { color: "#e8eaf0", family: "DM Mono, monospace" },
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: "100%", minHeight: "360px" }}
          />
        </div>
      ))}
    </div>
  );
}

// ── Flags ─────────────────────────────────────────────────────────
function FlagsTab({ flags }) {
  if (!flags.length) {
    return <p className={styles.empty}>✅ No anomalies flagged.</p>;
  }
  return (
    <div className={`card fade-in ${styles.flagsCard}`}>
      <h2 className={styles.sectionTitle}>Anomaly Flags ({flags.length})</h2>
      <div className={styles.flagList}>
        {flags.map((f, i) => (
          <div key={i} className={styles.flagRow}>
            <span className={`badge badge-${f.severity}`}>{f.severity}</span>
            <code className={styles.flagCol}>{f.column}</code>
            <span className={styles.flagIssue}>{f.issue}</span>
            <span className={styles.flagRec}>→ {f.recommendation}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Schema ────────────────────────────────────────────────────────
function SchemaTab({ schema }) {
  return (
    <div className={`card fade-in`}>
      <h2 className={styles.sectionTitle}>Column Schema</h2>
      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead>
            <tr>
              {Object.keys(schema[0] || {}).map((k) => <th key={k}>{k}</th>)}
            </tr>
          </thead>
          <tbody>
            {schema.map((row, i) => (
              <tr key={i}>
                {Object.entries(row).map(([k, v]) => (
                  <td key={k}>
                    {k === "column" ? <code>{v}</code> :
                     k === "role"   ? <span className={`badge badge-low`}>{v}</span> :
                     String(v)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── Chat ──────────────────────────────────────────────────────────
function ChatTab({ results }) {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content: `Hi! I've analysed **${results.filename}** (${results.shape.rows} rows × ${results.shape.cols} cols). What would you like to know?`,
    },
  ]);
  const [input,   setInput]   = useState("");
  const [loading, setLoading] = useState(false);

  const chatContext = {
    shape:        results.shape,
    schema_summary: results.schema_summary,
    flags:        results.eda.flags,
    correlations: results.eda.correlation.strong_pairs,
    narrative:    results.narrative,
    descriptive:  results.eda.descriptive,
  };

  async function sendMessage() {
    const q = input.trim();
    if (!q || loading) return;
    setInput("");
    const userMsg = { role: "user", content: q };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    try {
      const history = messages.filter((m) => m.role !== "assistant" || messages.indexOf(m) > 0);
      const { data } = await api.post("/api/chat", {
        question: q,
        context:  chatContext,
        history,
      });
      setMessages((prev) => [...prev, { role: "assistant", content: data.answer }]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Error: ${err.response?.data?.detail || err.message}` },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className={styles.chatWrap}>
      <div className={styles.chatMessages}>
        {messages.map((m, i) => (
          <div key={i} className={`${styles.bubble} ${m.role === "user" ? styles.bubbleUser : styles.bubbleBot}`}>
            <ReactMarkdown>{m.content}</ReactMarkdown>
          </div>
        ))}
        {loading && (
          <div className={`${styles.bubble} ${styles.bubbleBot} ${styles.thinking}`}>
            <span className="spinner" />
          </div>
        )}
      </div>
      <div className={styles.chatInput}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          placeholder="Ask about your data…"
          disabled={loading}
        />
        <button className="btn btn-primary" onClick={sendMessage} disabled={loading || !input.trim()}>
          Send
        </button>
      </div>
    </div>
  );
}

// ── Report ────────────────────────────────────────────────────────
function ReportTab({ markdown }) {
  function download() {
    const blob = new Blob([markdown], { type: "text/markdown" });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href     = url;
    a.download = "eda_report.md";
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className={`card fade-in ${styles.reportCard}`}>
      <div className={styles.reportHeader}>
        <h2 className={styles.sectionTitle}>Markdown Report</h2>
        <button className="btn btn-secondary" onClick={download}>⬇ Download .md</button>
      </div>
      <div className={styles.reportBody}>
        <ReactMarkdown>{markdown}</ReactMarkdown>
      </div>
    </div>
  );
}