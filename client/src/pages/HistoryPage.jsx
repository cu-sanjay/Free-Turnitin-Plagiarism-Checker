import { useMemo, useState } from "react";
import { createHistoryStore, recordToResult } from "@shared/history-store.js";
import ReportView from "@/components/ReportView";
import { downloadReport } from "@/lib/download-report";

function getStore() {
  if (typeof window === "undefined") return null;
  try {
    return createHistoryStore(window.localStorage);
  } catch {
    return null;
  }
}

export default function HistoryPage() {
  const store = useMemo(() => getStore(), []);
  const [entries, setEntries] = useState(() => store?.list() || []);
  const [activeId, setActiveId] = useState(entries[0]?.id || null);
  const active = entries.find((entry) => entry.id === activeId) || null;

  const refresh = (next, preferId) => {
    setEntries(next);
    setActiveId(preferId ?? next[0]?.id ?? null);
  };

  return (
    <div className="page">
      <header className="page-intro">
        <p className="kicker">On this device</p>
        <h1>History</h1>
        <p className="lede" style={{ maxWidth: "40rem", marginTop: "0.75rem" }}>
          Finished checks stay in this browser only. The full document text is not stored, and nothing is synced to an account.
        </p>
      </header>
      {!entries.length ? (
        <div className="panel panel-pad" style={{ marginTop: "1.2rem" }}>
          <p>No checks yet. A completed check on the Check page will appear here.</p>
        </div>
      ) : (
        <div className="history-layout" style={{ marginTop: "1.2rem" }}>
          <div className="history-list">
            <div className="row-between">
              <h2 style={{ fontSize: "1rem" }}>{entries.length} saved</h2>
              <button
                type="button"
                className="text-button"
                onClick={() => {
                  store?.clear();
                  refresh([]);
                }}
              >
                Clear history
              </button>
            </div>
            {entries.map((entry) => (
              <article key={entry.id} className="history-card">
                <header>
                  <h2 style={{ fontSize: "1.4rem" }}>{entry.plagiarismPercentage}%</h2>
                  <span className="meta">{new Date(entry.createdAt).toLocaleString()}</span>
                </header>
                <p style={{ margin: "0.45rem 0" }}>{entry.preview || "Untitled check"}</p>
                <p className="meta">
                  {entry.wordCount} words · {entry.passagesChecked} passages compared
                </p>
                <div className="row-between" style={{ marginTop: "0.6rem" }}>
                  <button type="button" className="secondary-button" onClick={() => setActiveId(entry.id)}>
                    Open
                  </button>
                  <button
                    type="button"
                    className="text-button"
                    onClick={() => {
                      const next = store?.remove(entry.id) || [];
                      refresh(next, activeId === entry.id ? null : activeId);
                    }}
                  >
                    Delete
                  </button>
                </div>
              </article>
            ))}
          </div>
          <div>
            {active ? (
              <ReportView result={recordToResult(active)} onDownload={() => downloadReport(recordToResult(active))} />
            ) : (
              <p className="meta">Select a check to open its report.</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
