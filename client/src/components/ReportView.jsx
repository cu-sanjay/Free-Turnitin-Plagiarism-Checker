import { Download } from "lucide-react";

function formatWhen(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleString();
}

export default function ReportView({ result, onDownload }) {
  if (!result) return null;
  const checked = result.passagesChecked ?? result.totalSentences ?? 0;
  const total = result.passagesInDocument ?? checked;

  return (
    <section className="report-layout rise" aria-labelledby="report-heading" data-testid="card-report">
      <div className="panel panel-pad">
        <p className="kicker">Similarity</p>
        <h2 id="report-heading" className="report-title" style={{ fontSize: "1.6rem", margin: "0.3rem 0 0.8rem" }}>
          Report
        </h2>
        <div className="score-card">
          <span className="meta">Passages over 50% similarity</span>
          <strong className="score-value" data-testid="text-plagiarism-percentage">
            {result.plagiarismPercentage}%
          </strong>
        </div>
        <div className="score-card" style={{ marginTop: "0.7rem" }}>
          <span className="meta">Average similarity</span>
          <strong className="score-value" style={{ fontSize: "2.4rem" }} data-testid="text-overall-score">
            {result.overallScore}%
          </strong>
        </div>
        <dl className="stat-list">
          <div>
            <dt className="meta">Compared</dt>
            <dd>
              <strong data-testid="text-total-sentences">{result.totalSentences}</strong>
            </dd>
          </div>
          <div>
            <dt className="meta">Flagged</dt>
            <dd>
              <strong data-testid="text-plagiarized-sentences">{result.plagiarizedSentences}</strong>
            </dd>
          </div>
          <div>
            <dt className="meta">In document</dt>
            <dd>
              <strong>{total}</strong>
            </dd>
          </div>
          <div>
            <dt className="meta">Words</dt>
            <dd>
              <strong>{result.totalWords ?? "—"}</strong>
            </dd>
          </div>
        </dl>
        <p className="meta">{result.coverageNote}</p>
        <p style={{ marginTop: "0.7rem" }}>{result.analysisSummary}</p>
        {result.checkedAt && <p className="meta" style={{ marginTop: "0.7rem" }}>{formatWhen(result.checkedAt)}</p>}
        <button type="button" className="primary-button" style={{ marginTop: "1rem", width: "100%" }} onClick={onDownload}>
          <Download size={16} aria-hidden="true" />
          Download PDF
        </button>
      </div>

      <div data-testid="card-details">
        <div className="row-between" style={{ marginBottom: "0.7rem" }}>
          <h3 style={{ fontSize: "1.2rem", letterSpacing: "-0.03em" }}>Matches</h3>
          <span className="meta">{checked} shown</span>
        </div>
        {result.sources?.length > 0 && (
          <div className="panel panel-pad" style={{ marginBottom: "0.8rem" }}>
            <h3 style={{ marginBottom: "0.6rem" }}>Sources</h3>
            <div className="source-list">
              {result.sources.map((source) => (
                <div className="source-row" key={source.url}>
                  <a href={source.url} target="_blank" rel="noopener noreferrer">
                    {source.host || source.url}
                  </a>
                  <span className="badge">{source.maxSimilarity}%</span>
                </div>
              ))}
            </div>
          </div>
        )}
        <div className="match-list">
          {(result.results || []).map((item, index) => (
            <article key={`${item.passageIndex ?? index}-${index}`} className="match-card" data-testid={`result-sentence-${index}`}>
              <header>
                <p data-testid={`text-sentence-${index}`}>{item.sentence}</p>
                <span className="badge" data-hot={item.isPlagiarized ? "true" : "false"} data-testid={`badge-similarity-${index}`}>
                  {item.similarity}%
                </span>
              </header>
              {item.sources?.length > 0 && (
                <div className="source-list">
                  {item.sources.map((source, sourceIndex) => (
                    <div key={`${source.url}-${sourceIndex}`}>
                      <div className="source-row">
                        <a
                          href={source.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          data-testid={`link-source-${index}-${sourceIndex}`}
                        >
                          {source.url}
                        </a>
                        <span data-testid={`text-source-similarity-${index}-${sourceIndex}`}>{source.similarity}%</span>
                      </div>
                      {source.excerpt && <p className="excerpt">{source.excerpt}</p>}
                    </div>
                  ))}
                </div>
              )}
            </article>
          ))}
          {!result.results?.length && <p className="meta">No passage was long enough to compare.</p>}
        </div>
      </div>
    </section>
  );
}
