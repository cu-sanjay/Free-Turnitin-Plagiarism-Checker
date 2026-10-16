import { useEffect, useMemo, useState } from "react";
import { Download } from "lucide-react";
import { buildHighlightRuns, formatPercent, runsToParagraphs } from "@shared/similarity.js";
import { sourceColor } from "@shared/source-colors.js";

function formatWhen(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleString();
}

function sourceStyle(id) {
  const color = sourceColor(id);
  return {
    "--hl": `hsl(${color.hsl})`,
    "--hl-soft": `hsl(${color.hsl} / 0.22)`,
  };
}

function Paragraph({ runs, activeSource, onMarker }) {
  if (!runs.length) return <p className="doc-blank">&nbsp;</p>;
  return (
    <p>
      {runs.map((run, index) => {
        if (!run.sourceId) return <span key={index}>{run.text}</span>;
        const dimmed = activeSource && activeSource !== run.sourceId;
        return (
          <mark
            key={index}
            id={index === 0 ? undefined : undefined}
            className="hl"
            data-source={run.sourceId}
            data-dimmed={dimmed ? "true" : "false"}
            style={sourceStyle(run.sourceId)}
          >
            {run.text}
            {run.marker ? (
              <a
                className="hl-mark"
                href={`#source-${run.marker}`}
                onClick={(event) => {
                  event.preventDefault();
                  onMarker(run.marker);
                }}
              >
                [{run.marker}]
              </a>
            ) : null}
          </mark>
        );
      })}
    </p>
  );
}

function HighlightedDocument({ text, spans, activeSource, onMarker }) {
  const paragraphs = useMemo(
    () => runsToParagraphs(buildHighlightRuns(text, spans || [])),
    [text, spans]
  );
  const [shown, setShown] = useState(48);

  useEffect(() => {
    setShown(48);
  }, [text, spans]);

  useEffect(() => {
    if (shown >= paragraphs.length) return undefined;
    const frame = window.requestAnimationFrame(() => {
      setShown((current) => Math.min(paragraphs.length, current + 64));
    });
    return () => window.cancelAnimationFrame(frame);
  }, [shown, paragraphs.length]);

  return (
    <div className="doc-paper" data-testid="report-document">
      {paragraphs.slice(0, shown).map((runs, index) => {
        const first = runs.find((run) => run.sourceId);
        return (
          <div
            key={index}
            id={first && !activeSource ? `span-p-${index}` : undefined}
            data-source-block={first?.sourceId || undefined}
          >
            <Paragraph runs={runs} activeSource={activeSource} onMarker={onMarker} />
          </div>
        );
      })}
    </div>
  );
}

export default function ReportView({ result, text = "", onDownload }) {
  const [activeSource, setActiveSource] = useState(null);
  if (!result) return null;

  const similarity = result.similarity || {};
  const overall = similarity.overallPercent ?? 0;
  const sources = result.sources || [];
  const coverage = result.coverage || {};
  const canHighlight = Boolean(text) && Array.isArray(result.spans);
  const visibleSources = activeSource ? sources.filter((source) => source.id === activeSource) : sources;
  const visibleSpans = activeSource ? (result.spans || []).filter((span) => span.sourceId === activeSource) : result.spans;

  const focusSource = (id) => {
    setActiveSource(id);
    window.requestAnimationFrame(() => {
      document.getElementById(`source-${id}`)?.scrollIntoView({ behavior: "smooth", block: "nearest" });
      const hit = document.querySelector(`[data-source-block="${id}"]`);
      hit?.scrollIntoView({ behavior: "smooth", block: "center" });
    });
  };

  return (
    <section className="report-shell rise" aria-labelledby="report-heading" data-testid="card-report">
      <div className="report-hero panel panel-pad">
        <p className="kicker">Overall similarity</p>
        <h2 id="report-heading" className="report-title">
          Report
        </h2>
        <div className="score-card">
          <span className="meta">Share of this document covered by matched text</span>
          <strong className="score-value" data-testid="text-plagiarism-percentage">
            {formatPercent(overall)}
          </strong>
        </div>
        <p className="report-summary">{result.summary}</p>
        <dl className="stat-list">
          <div>
            <dt className="meta">Matched words</dt>
            <dd>
              <strong data-testid="text-matched-words">
                {(similarity.matchedWords ?? 0).toLocaleString()}
              </strong>
              <span className="meta">of {(similarity.countedWords ?? result.totalWords ?? 0).toLocaleString()}</span>
            </dd>
          </div>
          <div>
            <dt className="meta">Sources</dt>
            <dd>
              <strong data-testid="text-source-count">{sources.length}</strong>
            </dd>
          </div>
          <div>
            <dt className="meta">Coverage</dt>
            <dd>
              <strong data-testid="text-total-sentences">
                {coverage.segmentsSearched ?? "—"}/{coverage.segmentsTotal ?? "—"}
              </strong>
              <span className="meta">segments searched</span>
            </dd>
          </div>
          <div>
            <dt className="meta">Longest match</dt>
            <dd>
              <strong>{(similarity.longestMatchWords ?? 0).toLocaleString()}</strong>
              <span className="meta">words</span>
            </dd>
          </div>
        </dl>
        <div className="note-stack">
          {(result.notes || []).map((note) => (
            <p key={note.text} className="meta" data-tone={note.level}>
              {note.text}
            </p>
          ))}
        </div>
        {result.checkedAt && <p className="meta">{formatWhen(result.checkedAt)}</p>}
        <button type="button" className="primary-button report-download" onClick={onDownload}>
          <Download size={16} aria-hidden="true" />
          Download PDF
        </button>
      </div>

      <div className="report-body">
        <aside className="panel panel-pad source-panel" data-testid="card-details">
          <div className="row-between" style={{ marginBottom: "0.7rem" }}>
            <h3>Sources</h3>
            {activeSource ? (
              <button type="button" className="text-button" onClick={() => setActiveSource(null)}>
                Show all
              </button>
            ) : (
              <span className="meta">{sources.length}</span>
            )}
          </div>
          {!sources.length && <p className="meta">No public page produced a long enough word run to count as a match.</p>}
          <div className="source-list">
            {visibleSources.map((source) => (
              <article
                key={source.url}
                id={`source-${source.id}`}
                className="source-item"
                data-active={activeSource === source.id ? "true" : "false"}
                style={sourceStyle(source.id)}
              >
                <button type="button" className="source-swatch" onClick={() => focusSource(source.id)} aria-label={`Focus source ${source.id}`}>
                  [{source.id}]
                </button>
                <div>
                  <a href={source.url} target="_blank" rel="noopener noreferrer">
                    {source.title || source.host || source.url}
                  </a>
                  <p className="meta">
                    {source.host} · {formatPercent(source.sharePercent)} of the document · {(source.matchedWords || 0).toLocaleString()} words
                  </p>
                </div>
              </article>
            ))}
          </div>
        </aside>

        <div className="panel panel-pad document-panel">
          <div className="row-between" style={{ marginBottom: "0.7rem" }}>
            <h3>Your document</h3>
            <span className="meta">{(result.totalWords || 0).toLocaleString()} words</span>
          </div>
          {!canHighlight ? (
            <p className="callout">
              This saved report does not include the original document, so the highlights cannot be redrawn. The overall
              similarity and sources are still the recorded figures from that check.
            </p>
          ) : (
            <HighlightedDocument
              text={text}
              spans={visibleSpans}
              activeSource={activeSource}
              onMarker={focusSource}
            />
          )}
        </div>
      </div>
    </section>
  );
}
