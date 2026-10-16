import { useEffect, useRef, useState } from "react";
import { Link } from "wouter";
import { FileText, Loader2, Quote } from "lucide-react";
import { checkTextSchema } from "@shared/schema.js";
import { createHistoryStore, toHistoryRecord } from "@shared/history-store.js";
import ReportView from "@/components/ReportView";
import { downloadReport } from "@/lib/download-report";

const MAX_REQUEST_BYTES = 11_000_000;

function countWords(value) {
  if (!value) return 0;
  let words = 0;
  const step = 200_000;
  for (let index = 0; index < value.length; index += step) {
    const slice = value.slice(index, Math.min(value.length, index + step)).trim();
    if (!slice) continue;
    words += slice.split(/\s+/).filter(Boolean).length;
  }
  return words;
}

async function readEvents(response, onEvent) {
  const reader = response.body?.getReader();
  if (!reader) throw new Error("This browser could not read the check progress stream.");
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split("\n\n");
    buffer = chunks.pop() || "";
    for (const chunk of chunks) {
      const line = chunk.split("\n").find((entry) => entry.startsWith("data:"));
      if (!line) continue;
      onEvent(JSON.parse(line.slice(5).trim()));
    }
  }
}

const Index = () => {
  const textareaRef = useRef(null);
  const countTimer = useRef(0);
  const abortRef = useRef(null);
  const [counts, setCounts] = useState({ characters: 0, words: 0 });
  const [excludeQuotes, setExcludeQuotes] = useState(false);
  const [isChecking, setIsChecking] = useState(false);
  const [progress, setProgress] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => () => {
    window.clearTimeout(countTimer.current);
    abortRef.current?.abort();
  }, []);

  const refreshCounts = () => {
    const value = textareaRef.current?.value || "";
    setCounts((current) => ({ ...current, characters: value.length }));
    window.clearTimeout(countTimer.current);
    countTimer.current = window.setTimeout(() => {
      setCounts({ characters: value.length, words: countWords(value) });
    }, value.length > 400_000 ? 350 : 60);
  };

  const handleFile = async (event) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    const textFile = /\.(txt|md|markdown|text)$/i.test(file.name) || file.type.startsWith("text/");
    if (!textFile) {
      setError("Only plain-text files can be opened in the browser. PDF and Word files are not extracted.");
      return;
    }
    if (file.size > MAX_REQUEST_BYTES) {
      setError("That file is larger than 11 MB, which is more than one check can accept. Split it into smaller text files.");
      return;
    }
    const text = await file.text();
    if (textareaRef.current) textareaRef.current.value = text;
    setError("");
    setResult(null);
    refreshCounts();
  };

  const handleCheck = async () => {
    const text = textareaRef.current?.value || "";
    if (!text.trim()) {
      setError("Paste or open some text to check.");
      return;
    }
    const validation = checkTextSchema.safeParse({ text });
    if (!validation.success) {
      setError(validation.error.errors[0]?.message || "Validation failed");
      return;
    }
    if (new Blob([text]).size > MAX_REQUEST_BYTES) {
      setError("This document is larger than one request can carry. Split it, then check each part.");
      return;
    }

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setIsChecking(true);
    setResult(null);
    setError("");
    setProgress({ percent: 1, phase: "prepare", message: "Preparing the document", completed: 0, total: 0 });
    await new Promise((resolve) => setTimeout(resolve, 40));

    try {
      const response = await fetch("/api/plagiarism-check/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, excludeQuotes }),
        signal: controller.signal,
      });
      const contentType = response.headers.get("content-type") || "";
      if (!response.ok && !contentType.includes("text/event-stream")) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.error || data.message || "The check failed.");
      }
      if (!response.ok && !response.body) {
        throw new Error("The check failed.");
      }

      let finalResult = null;
      let streamError = "";
      await readEvents(response, (event) => {
        if (event.type === "progress") {
          setProgress({
            percent: event.percent,
            phase: event.phase,
            message: event.message,
            completed: event.completed || 0,
            total: event.total || 0,
          });
        } else if (event.type === "result") {
          finalResult = event.result;
        } else if (event.type === "error") {
          streamError = event.message || "The check failed.";
        }
      });

      if (streamError) throw new Error(streamError);
      if (!finalResult) throw new Error("The check ended before a report was ready.");
      setResult(finalResult);
      setProgress({ percent: 100, phase: "done", message: "Check complete", completed: finalResult.passagesChecked, total: finalResult.passagesChecked });
      try {
        const store = createHistoryStore(window.localStorage);
        store.save(toHistoryRecord({ text, result: finalResult, excludeQuotes }));
      } catch (storageError) {
        console.error("Could not save history", storageError);
      }
    } catch (checkError) {
      if (checkError?.name === "AbortError") {
        setProgress(null);
        return;
      }
      console.error("Error checking plagiarism:", checkError);
      setError(checkError.message || "Failed to check plagiarism. Please try again.");
    } finally {
      setIsChecking(false);
    }
  };

  const cancel = () => {
    abortRef.current?.abort();
    setIsChecking(false);
    setProgress(null);
  };

  const clearDocument = () => {
    if (textareaRef.current) textareaRef.current.value = "";
    setCounts({ characters: 0, words: 0 });
    setResult(null);
    setError("");
    setProgress(null);
    textareaRef.current?.focus();
  };

  return (
    <div className="page">
      <header className="page-intro">
        <p className="kicker">Document check</p>
        <h1>See what the public web already says</h1>
        <p className="lede" style={{ maxWidth: "40rem", marginTop: "0.8rem" }}>
          Paste your writing, or open a plain-text file. I only score the passages I actually compare with pages I can fetch.
        </p>
        <p className="quiet-note">
          <img className="turnitin-mark" src="/logos/turnitin.svg" alt="Turnitin" />
          <span>
            I used to email you free Turnitin reports from my class. I can’t anymore.{" "}
            <Link href="/about">Here’s what changed.</Link>
          </span>
        </p>
      </header>

      <div className="check-layout">
        <section className="panel panel-pad" data-testid="card-input" aria-labelledby="document-heading">
          <div className="row-between" style={{ marginBottom: "0.6rem" }}>
            <h2 id="document-heading" className="section-label" style={{ fontSize: "1.15rem" }}>
              <FileText size={18} aria-hidden="true" /> Document
            </h2>
            <label className="file-button">
              Open text
              <input
                type="file"
                accept=".txt,.md,.markdown,.text,text/plain"
                onChange={handleFile}
                style={{ position: "absolute", width: 1, height: 1, overflow: "hidden", clip: "rect(0 0 0 0)" }}
              />
            </label>
          </div>
          <label className="meta" htmlFor="document-text">
            Text to compare
          </label>
          <textarea
            id="document-text"
            ref={textareaRef}
            className="editor"
            data-testid="input-text"
            placeholder="Paste your document here. At least 100 characters."
            onInput={refreshCounts}
            spellCheck="true"
          />
          <div className="toolbar">
            <p className="meta" data-testid="text-character-count">
              {counts.characters.toLocaleString()} characters · {counts.words.toLocaleString()} words
            </p>
            <button type="button" className="text-button" onClick={clearDocument}>
              Clear
            </button>
          </div>
        </section>

        <aside className="panel panel-pad">
          <h2 className="section-label" style={{ fontSize: "1.15rem" }}>
            <Quote size={18} aria-hidden="true" /> Submission
          </h2>
          <label className="check-option">
            <input
              type="checkbox"
              checked={excludeQuotes}
              onChange={(event) => setExcludeQuotes(event.target.checked)}
            />
            <span>Exclude quotations and citation markers before comparing</span>
          </label>
          <div className="row-between">
            <button
              type="button"
              className="primary-button"
              data-testid="button-check-plagiarism"
              onClick={handleCheck}
              disabled={isChecking || counts.characters < 100}
            >
              {isChecking && <Loader2 size={16} className="animate-spin" aria-hidden="true" />}
              {isChecking ? "Checking" : "Check plagiarism"}
            </button>
            {isChecking && (
              <button type="button" className="secondary-button" onClick={cancel}>
                Cancel
              </button>
            )}
          </div>
          {isChecking && progress && (
            <div data-testid="alert-checking" style={{ marginTop: "1rem" }}>
              <div
                className="progress-track"
                role="progressbar"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={progress.percent}
                aria-label={progress.message}
              >
                <div className="progress-fill" style={{ width: `${progress.percent}%` }} />
              </div>
              <p className="status-line" aria-live="polite">
                {progress.message} · {progress.percent}%
              </p>
            </div>
          )}
          {error && (
            <div className="callout" data-tone="error" role="alert">
              {error}
            </div>
          )}
          {!isChecking && !error && (
            <p className="callout meta">
              Long documents stay in the editor and are compared as a spread of passages, so the page does not try to render every sentence at once.
            </p>
          )}
        </aside>
      </div>

      {result && <ReportView result={result} onDownload={() => downloadReport(result)} />}
    </div>
  );
};

export default Index;
