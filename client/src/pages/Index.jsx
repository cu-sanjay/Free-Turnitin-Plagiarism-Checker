import { useEffect, useRef, useState } from "react";
import {
  ArrowDownToLine,
  BookOpenCheck,
  Check,
  ChevronRight,
  CircleHelp,
  ExternalLink,
  FileText,
  Fingerprint,
  Plus,
  Printer,
  RotateCcw,
  ShieldCheck,
  Sparkles,
  Upload,
  X,
} from "lucide-react";
import { checkTextSchema } from "../../../shared/schema";
import { useToast } from "@/hooks/use-toast";
import { readDocumentFile } from "@/lib/readDocument";

const DEMO_TEXT = `Urban trees are an essential part of healthy cities. They cool neighborhoods by providing shade and releasing moisture into the air, while their roots help absorb rainwater and reduce flooding. Access to green spaces has also been linked to lower stress and stronger community connections.

Despite these benefits, tree cover is not shared equally across many cities. Lower-income neighborhoods often have fewer trees, leaving residents more exposed to extreme heat. City planners can help close this gap by prioritizing tree planting where temperatures are highest and maintaining the trees after they are planted.

Successful urban forestry depends on more than planting. Choosing species suited to local conditions, protecting young trees, and involving residents in care can make green infrastructure last for generations.`;

const DEMO_SOURCES = [
  {
    name: "Urban Forestry Review",
    text: "Urban trees are an essential part of healthy cities. They cool neighborhoods by providing shade and releasing moisture into the air. Tree roots help absorb rainwater and reduce flooding. Access to green spaces has also been linked to lower stress and stronger community connections.",
  },
  {
    name: "City Climate & Equity Report",
    text: "Tree cover is not shared equally across many cities. Lower-income neighborhoods often have fewer trees, leaving residents more exposed to extreme heat. City planners can close this gap by prioritizing tree planting where temperatures are highest and maintaining trees after they are planted.",
  },
];

function newSource(name = "") {
  return {
    id: globalThis.crypto?.randomUUID?.() ?? `source-${Date.now()}-${Math.random()}`,
    name,
    exclude: false,
    url: "",
    text: "",
  };
}

function formatNumber(value) {
  return new Intl.NumberFormat().format(value);
}

function HighlightedDraft({ text, ranges }) {
  const parts = [];
  let cursor = 0;

  for (const [index, range] of ranges.entries()) {
    if (range.start > cursor) {
      parts.push(<span key={`plain-${index}`}>{text.slice(cursor, range.start)}</span>);
    }
    if (range.start >= cursor) {
      parts.push(
        <mark
          className="draft-highlight"
          key={`match-${index}`}
          title={`Matched reference: ${range.sources.join(", ")}`}
        >
          {text.slice(range.start, range.end)}
        </mark>
      );
      cursor = range.end;
    }
  }

  if (cursor < text.length) {
    parts.push(<span key="plain-end">{text.slice(cursor)}</span>);
  }

  return <p className="draft-preview">{parts}</p>;
}

export default function Index() {
  const [text, setText] = useState("");
  const [sources, setSources] = useState([newSource("Reference 1")]);
  const [options, setOptions] = useState({
    excludeQuotes: false,
    excludeReferences: false,
    minMatchWords: 5,
  });
  const [result, setResult] = useState(null);
  const [isChecking, setIsChecking] = useState(false);
  const draftInput = useRef(null);
  const sourceInput = useRef(null);
  const { toast } = useToast();

  useEffect(() => {
    if (result) {
      document.getElementById("report")?.scrollIntoView({ behavior: "smooth" });
    }
  }, [result]);

  const wordCount = (text.match(/[\p{L}\p{N}]+(?:['’][\p{L}\p{N}]+)*/gu) ?? []).length;

  const updateSource = (id, field, value) => {
    setSources((current) =>
      current.map((source) =>
        source.id === id ? { ...source, [field]: value } : source
      )
    );
  };

  const importDraft = async (file) => {
    if (!file) return;
    try {
      setText(await readDocumentFile(file, 50000));
      setResult(null);
    } catch (error) {
      toast({
        title: "Could not import this draft",
        description: error instanceof Error ? error.message : "Try pasting its text instead.",
        variant: "destructive",
      });
    }
  };

  const importSources = async (files) => {
    const selectedFiles = Array.from(files ?? []);
    const replacePlaceholder =
      sources.length === 1 &&
      sources[0].name === "Reference 1" &&
      !sources[0].text.trim();
    if (sources.length - Number(replacePlaceholder) + selectedFiles.length > 25) {
      toast({
        title: "Reference limit reached",
        description: "You can compare a draft against up to 25 references.",
        variant: "destructive",
      });
      return;
    }
    const imported = [];
    for (const file of selectedFiles) {
      try {
        imported.push({
          ...newSource(file.name.replace(/\.(txt|md|docx|pdf)$/i, "")),
          text: await readDocumentFile(file, 100000),
        });
      } catch (error) {
        toast({
          title: "Could not read the file",
          description: `${file.name}: ${error instanceof Error ? error.message : "Try pasting its text instead."}`,
          variant: "destructive",
        });
      }
    }
    if (imported.length) {
      setSources((current) => [
        ...(replacePlaceholder
          ? current.filter(({ id }) => id !== sources[0].id)
          : current),
        ...imported,
      ]);
      setResult(null);
    }
  };

  const runCheck = async (event) => {
    event.preventDefault();
    const validation = checkTextSchema.safeParse({ text, sources, options });
    if (!validation.success) {
      toast({
        title: "A little more needed",
        description: validation.error.issues[0]?.message ?? "Check your draft and references.",
        variant: "destructive",
      });
      return;
    }

    setIsChecking(true);
    setResult(null);
    try {
      const response = await fetch("/api/plagiarism-check", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(validation.data),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error ?? data.message ?? "Could not compare these documents.");
      }
      setResult(data);
    } catch (error) {
      toast({
        title: "Could not complete the check",
        description:
          error instanceof Error ? error.message : "Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsChecking(false);
    }
  };

  const loadDemo = () => {
    setText(DEMO_TEXT);
    setSources(DEMO_SOURCES.map((source) => ({ ...newSource(), ...source })));
    setResult(null);
  };

  const reset = () => {
    setText("");
    setSources([newSource("Reference 1")]);
    setResult(null);
  };

  const downloadReport = () => {
    if (!result) return;
    const report = {
      product: "ClearMark",
      generatedAt: new Date().toISOString(),
      method: "Exact phrase overlap against user-provided reference documents only",
      exclusions: options,
      wordCount: result.wordCount,
      excludedWords: result.excludedWords,
      matchedWords: result.matchedWords,
      similarityPercentage: result.similarityPercentage,
      minMatchWords: result.minMatchWords,
      sources: result.sources,
    };
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(report, null, 2)], { type: "application/json" })
    );
    const link = document.createElement("a");
    link.href = url;
    link.download = "clearmark-similarity-report.json";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  const printReport = () => window.print();

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="ClearMark home">
          <span className="brand-mark"><Fingerprint size={21} strokeWidth={2.2} /></span>
          <span>clear<span>mark</span></span>
        </a>
        <div className="topbar-note"><ShieldCheck size={16} /> Your documents stay yours</div>
      </header>

      <section className="hero">
        <div className="hero-copy">
          <div className="eyebrow"><Sparkles size={14} /> ORIGINALITY, WITH CONTEXT</div>
          <h1>Make every source<br /><span>easy to see.</span></h1>
          <p>Compare your writing with the documents that matter. Find matching phrases, review the evidence, and make your next edit with confidence.</p>
          <button className="text-button demo-link" type="button" onClick={loadDemo}>
            Try a sample check <ChevronRight size={16} />
          </button>
        </div>
        <div className="hero-art" aria-hidden="true">
          <div className="art-paper paper-back"><span /><span /><span /></div>
          <div className="art-paper paper-front"><Fingerprint size={35} /><span /><span /><span className="art-highlight" /></div>
          <span className="art-spark spark-one">✳</span><span className="art-spark spark-two">✦</span>
        </div>
      </section>

      <form className="workspace" onSubmit={runCheck}>
        <div className="workspace-heading">
          <div>
            <span className="section-kicker">YOUR WORKSPACE</span>
            <h2>Start a new check</h2>
          </div>
          <button className="text-button reset-button" type="button" onClick={reset}><RotateCcw size={15} /> Clear all</button>
        </div>

        <section className="panel draft-panel">
          <div className="panel-heading">
            <div className="step-icon">01</div>
            <div className="panel-title">
              <h3>Your draft</h3>
              <p>Paste the writing you want to review</p>
            </div>
            <button className="quiet-button" type="button" onClick={() => draftInput.current?.click()}>
              <Upload size={15} /> Upload text
            </button>
            <input
              ref={draftInput}
              className="visually-hidden"
              type="file"
              accept=".txt,.md,.docx,.pdf,text/plain,text/markdown,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
              onChange={(event) => {
                void importDraft(event.target.files?.[0]);
                event.target.value = "";
              }}
              aria-label="Upload draft text file"
            />
          </div>
          <textarea
            className="draft-input"
            aria-label="Draft text"
            placeholder="Paste or type your draft here..."
            value={text}
            onChange={(event) => {
              setText(event.target.value);
              setResult(null);
            }}
            maxLength={50000}
          />
          <div className="field-footer">
            <span>{formatNumber(wordCount)} words <span className="footer-dot">·</span> {formatNumber(text.length)} / 50,000 characters</span>
            <span className={text.trim().length >= 100 ? "ready-hint" : ""}>
              {text.trim().length >= 100 ? <><Check size={13} /> Ready to check</> : "100 characters minimum"}
            </span>
          </div>
        </section>

        <section className="panel sources-panel">
          <div className="panel-heading sources-heading">
            <div className="step-icon">02</div>
            <div className="panel-title">
              <h3>Reference documents</h3>
              <p>Compare up to 25 references. Add each source link for report attribution.</p>
            </div>
            <div className="source-actions">
              <button className="quiet-button" type="button" onClick={() => sourceInput.current?.click()}>
                <Upload size={15} /> Upload files
              </button>
              <input
                ref={sourceInput}
                className="visually-hidden"
                type="file"
                multiple
                accept=".txt,.md,.docx,.pdf,text/plain,text/markdown,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                onChange={(event) => {
                  void importSources(event.target.files);
                  event.target.value = "";
                }}
                aria-label="Upload reference text files"
              />
              <button
                className="quiet-button add-button"
                type="button"
                onClick={() => {
                  if (sources.length >= 25) {
                    toast({ title: "Reference limit reached", description: "You can compare against up to 25 references." });
                    return;
                  }
                  setSources((current) => [...current, newSource(`Reference ${current.length + 1}`)]);
                  setResult(null);
                }}
              >
                <Plus size={15} /> Add source
              </button>
            </div>
          </div>

          <div className="source-list">
            {sources.map((source, index) => (
              <article className="source-card" key={source.id}>
                <div className="source-card-top">
                  <div className="source-label"><FileText size={15} /> SOURCE {String(index + 1).padStart(2, "0")}</div>
                  <label className="source-exclude">
                    <input
                      type="checkbox"
                      checked={source.exclude}
                      onChange={(event) => {
                        updateSource(source.id, "exclude", event.target.checked);
                        setResult(null);
                      }}
                    />
                    Exclude
                  </label>
                  {sources.length > 1 && (
                    <button
                      className="icon-button"
                      type="button"
                      aria-label={`Remove ${source.name || `source ${index + 1}`}`}
                      onClick={() => {
                        setSources((current) => current.filter(({ id }) => id !== source.id));
                        setResult(null);
                      }}
                    ><X size={16} /></button>
                  )}
                </div>
                <input
                  className="source-name"
                  aria-label={`Reference ${index + 1} name`}
                  placeholder="Give this source a name"
                  value={source.name}
                  maxLength={80}
                  onChange={(event) => updateSource(source.id, "name", event.target.value)}
                />
                <input
                  className="source-url"
                  type="url"
                  aria-label={`Reference ${index + 1} source link`}
                  placeholder="Source URL (optional)"
                  value={source.url}
                  maxLength={2048}
                  onChange={(event) => {
                    updateSource(source.id, "url", event.target.value);
                    setResult(null);
                  }}
                />
                <textarea
                  className="source-input"
                  aria-label={`Reference ${index + 1} text`}
                  placeholder="Paste the reference text here..."
                  value={source.text}
                  maxLength={100000}
                  onChange={(event) => {
                    updateSource(source.id, "text", event.target.value);
                    setResult(null);
                  }}
                />
                <div className="source-count">{formatNumber(source.text.length)} / 100,000 characters</div>
              </article>
            ))}
          </div>
          <div className="analysis-options">
            <span className="options-label">REFINE THIS CHECK</span>
            <label className="option-toggle match-length-control">
              <span>Minimum matching phrase</span>
              <select
                className="match-length-select"
                value={options.minMatchWords}
                onChange={(event) => {
                  setOptions((current) => ({
                    ...current,
                    minMatchWords: Number(event.target.value),
                  }));
                  setResult(null);
                }}
              >
                <option value={5}>5 words</option>
                <option value={8}>8 words</option>
                <option value={12}>12 words</option>
              </select>
            </label>
            <label className="option-toggle">
              <input
                type="checkbox"
                checked={options.excludeQuotes}
                onChange={(event) => {
                  setOptions((current) => ({ ...current, excludeQuotes: event.target.checked }));
                  setResult(null);
                }}
              />
              <span>Exclude quoted text</span>
            </label>
            <label className="option-toggle">
              <input
                type="checkbox"
                checked={options.excludeReferences}
                onChange={(event) => {
                  setOptions((current) => ({ ...current, excludeReferences: event.target.checked }));
                  setResult(null);
                }}
              />
              <span>Exclude bibliography / references section</span>
            </label>
            <span className="option-explainer">Longer phrases reduce matches on short common wording. Quotes and reference sections are detected from punctuation and common headings.</span>
          </div>
          <div className="privacy-note"><ShieldCheck size={16} /><span><strong>Private by design.</strong> Your text is compared on this app only and is not saved to a source database.</span></div>
        </section>

        <div className="run-row">
          <div className="method-note"><CircleHelp size={16} /><span>Checks for shared runs of {options.minMatchWords} or more words.<br />A match is a signal to review—not a verdict.</span></div>
          <button className="run-button" type="submit" disabled={isChecking || text.trim().length < 100}>
            {isChecking ? <span className="spinner" /> : <BookOpenCheck size={18} />}
            {isChecking ? "Comparing documents..." : "Check for matches"}
            {!isChecking && <ChevronRight size={17} />}
          </button>
        </div>
      </form>

      {result && (
        <section className="report" id="report" aria-live="polite">
          <div className="report-heading">
            <div><span className="section-kicker">YOUR RESULTS</span><h2>Similarity report</h2></div>
            <div className="report-actions">
              <button className="quiet-button" type="button" onClick={printReport}><Printer size={16} /> Print / Save PDF</button>
              <button className="quiet-button" type="button" onClick={downloadReport}><ArrowDownToLine size={16} /> Export JSON</button>
            </div>
          </div>
          <div className="report-grid">
            <div className="score-card">
              <div className="score-ring" style={{ "--score": `${result.similarityPercentage}%` }}>
                <div><strong>{result.similarityPercentage}<small>%</small></strong><span>similarity</span></div>
              </div>
              <div className="score-copy">
                <h3>{result.similarityPercentage === 0 ? "No matching phrases found" : `${result.matchedWords} words matched`}</h3>
                <p>Across {result.sources.length} of {sources.length} reference {sources.length === 1 ? "document" : "documents"}</p>
                <div className="score-legend"><span className="legend-match" /> Matched <span className="legend-clear" /> No match</div>
              </div>
              <div className="score-stats">
                <div><strong>{formatNumber(result.wordCount)}</strong><span>words checked</span></div>
                <div><strong>{result.sources.length}</strong><span>sources with matches</span></div>
                {result.excludedWords > 0 && <div><strong>{formatNumber(result.excludedWords)}</strong><span>words excluded</span></div>}
                {result.excludedSourceCount > 0 && <div><strong>{result.excludedSourceCount}</strong><span>sources excluded</span></div>}
              </div>
            </div>
            <div className="evidence-card">
              <div className="evidence-heading"><div><span className="section-kicker">EVIDENCE</span><h3>Matched passages</h3></div><span className="passage-count">{result.sources.reduce((sum, source) => sum + source.matches.length, 0)}</span></div>
              {result.sources.length ? (
                <div className="evidence-list">
                  {result.sources.map((source) => (
                    <article className="evidence-source" key={source.id}>
                      <div className="evidence-source-heading"><span><FileText size={15} /> {source.name} {source.url && <a className="source-link" href={source.url} target="_blank" rel="noopener noreferrer" aria-label={`Open ${source.name} source link`}><ExternalLink size={13} /></a>}</span><strong>{source.similarityPercentage}%</strong></div>
                      {source.matches.map((match, index) => (
                        <div className="match-pair" key={`${source.id}-${match.start}-${index}`}>
                          <div><span className="match-label">YOUR DRAFT</span><p>{match.text}</p></div>
                          <div><span className="match-label">REFERENCE</span><p>{match.referenceText}</p></div>
                        </div>
                      ))}
                    </article>
                  ))}
                </div>
              ) : (
                <div className="empty-evidence"><div className="empty-check"><Check size={19} /></div><p>No matching passages of five or more words were found in the references you provided.</p><span>This check does not search the public web or any external database.</span></div>
              )}
            </div>
          </div>
          <article className="draft-result-card">
            <div className="draft-result-heading"><div><span className="section-kicker">ANNOTATED DRAFT</span><h3>Review the highlighted text</h3></div><span className="highlight-key"><i /> Matching phrase</span></div>
            <HighlightedDraft text={text} ranges={result.highlightRanges} />
          </article>
          <p className="report-disclaimer"><ShieldCheck size={15} /> Similarity is a text-overlap measure, not a determination of plagiarism. Properly quoted or cited text can also match.</p>
        </section>
      )}

      <footer className="page-footer"><span>clearmark <span className="footer-dot">·</span> Thoughtful originality checks</span><span>Only compares the documents you provide</span></footer>
    </main>
  );
}
