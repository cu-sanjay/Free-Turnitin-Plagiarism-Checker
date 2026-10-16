import {
  buildDocIndex,
  findSourceRuns,
  MIN_RUN_WORDS,
  planSegments,
  queryForWindow,
  resolveSpans,
  roundTo,
  SHINGLE_SIZE,
  spreadOrder,
  tokenizeDocument,
  tokenizeSource,
  formatPercent,
} from "../shared/similarity.js";
import { abortError, createSearchProvider, fetchPageContent, hostnameOf, shouldFetchUrl } from "./search.js";

export const RESULT_VERSION = 2;
const QUERY_WORDS = 10;
const SEGMENT_WORDS = 30;
const COVERED_FRACTION = 0.7;
const SEARCH_WORKERS = 3;
const FETCH_SLOTS = 4;
const FETCHES_PER_HOST = 2;

const yieldToEventLoop = () => new Promise((resolve) => setImmediate(resolve));

function clampPercent(value) {
  return Math.max(0, Math.min(100, Math.round(value)));
}

class Semaphore {
  constructor(limit) {
    this.limit = limit;
    this.active = 0;
    this.waiting = [];
  }

  async use(task) {
    if (this.active >= this.limit) await new Promise((resolve) => this.waiting.push(resolve));
    this.active++;
    try {
      return await task();
    } finally {
      this.active--;
      this.waiting.shift()?.();
    }
  }
}

export function defaultQueryBudget(wordCount) {
  return Math.max(24, Math.min(60, Math.round(24 + wordCount / 150)));
}

function normalizeFetched(value) {
  if (typeof value === "string") return value ? { ok: true, text: value, title: "" } : { ok: false, reason: "Empty page" };
  return value || { ok: false, reason: "No response" };
}

function normalizeSearch(value) {
  if (Array.isArray(value)) return { urls: value, answered: true, errors: [] };
  return { urls: value?.urls || [], answered: value?.answered ?? true, errors: value?.errors || [] };
}

export function describeSimilarity(stats, sourceCount) {
  if (!stats.countedWords) return "There was no text to compare.";
  const share = formatPercent(stats.overallPercent);
  if (!stats.matchedWords) {
    return "No passage of 7 or more words matched a public page that this check could open.";
  }
  return `${share} of the document (${stats.matchedWords.toLocaleString("en-US")} of ${stats.countedWords.toLocaleString("en-US")} words) matches ${sourceCount} public source${sourceCount === 1 ? "" : "s"}.`;
}

export async function checkPlagiarism(text, options = {}) {
  const {
    excludeQuotes = false,
    onProgress = () => {},
    signal,
    provider = options.search ? null : createSearchProvider(),
    search = (query, abortSignal) => provider.search(query, abortSignal),
    fetchContent = (url, abortSignal) => fetchPageContent(url, { signal: abortSignal }),
    maxQueries,
    maxFetches = 80,
    timeBudgetMs = 240_000,
    urlsPerQuery = 5,
    searchWorkers = SEARCH_WORKERS,
  } = options;

  let lastPercent = 0;
  const report = (phase, percent, message, extra = {}) => {
    lastPercent = Math.max(lastPercent, clampPercent(percent));
    onProgress({ phase, percent: lastPercent, message, ...extra });
  };
  const throwIfAborted = () => {
    if (signal?.aborted) throw abortError();
  };

  report("prepare", 2, "Reading the document");
  await yieldToEventLoop();
  throwIfAborted();

  const doc = tokenizeDocument(text, { excludeQuotes });
  const docIndex = buildDocIndex(doc);
  const segments = planSegments(doc, { segmentWords: SEGMENT_WORDS, queryWords: QUERY_WORDS });
  const budget = maxQueries ?? defaultQueryBudget(doc.counted);

  const order = spreadOrder(segments.length);
  const queue = [];
  for (const pass of ["primary", "alternate"]) {
    for (const index of order) {
      const window = segments[index][pass];
      if (window) queue.push({ segment: segments[index], window, pass });
    }
  }
  const planned = Math.min(budget, queue.length);
  const segmentsWithoutQuery = segments.filter((segment) => !segment.primary).length;

  report("prepare", 6, `Found ${doc.counted.toLocaleString("en-US")} words in ${segments.length} segments`, {
    completed: 0,
    total: planned,
  });

  const covered = new Uint8Array(doc.count);
  const sourceRuns = new Map();
  const fetchedUrls = new Set();
  const failures = [];
  const failureReasons = new Map();
  const searchErrors = new Map();
  const searchedSegments = new Set();
  const skippedSegments = new Set();
  const hostSlots = new Map();
  const fetchGate = new Semaphore(FETCH_SLOTS);
  const deadline = Date.now() + timeBudgetMs;

  let cursor = 0;
  let executed = 0;
  let processed = 0;
  let answeredQueries = 0;
  let pagesFetched = 0;
  let fetchAttempts = 0;
  let timeLimitReached = false;
  let fetchLimitReached = false;
  let coveredWords = 0;

  const currentSegmentCoverage = (segment) => {
    let hit = 0;
    for (let i = segment.start; i < segment.end; i++) hit += covered[i];
    return segment.end > segment.start ? hit / (segment.end - segment.start) : 1;
  };

  const emitProgress = (message) => {
    const fractionExecuted = planned ? executed / planned : 1;
    const fractionProcessed = queue.length ? processed / queue.length : 1;
    report("search", 8 + 84 * Math.min(1, Math.max(fractionExecuted, fractionProcessed)), message, {
      completed: executed,
      total: planned,
      sourcesFound: sourceRuns.size,
      pagesFetched,
      matchedPercent: doc.counted ? roundTo((coveredWords / doc.counted) * 100, 1) : 0,
    });
  };

  const noteFailure = (url, reason) => {
    failureReasons.set(reason, (failureReasons.get(reason) || 0) + 1);
    if (failures.length < 12) failures.push({ url, host: hostnameOf(url), reason });
  };

  const hostSemaphore = (host) => {
    if (!hostSlots.has(host)) hostSlots.set(host, new Semaphore(FETCHES_PER_HOST));
    return hostSlots.get(host);
  };

  async function comparePage(url) {
    const host = hostnameOf(url);
    const page = await fetchGate.use(() =>
      hostSemaphore(host).use(async () => {
        throwIfAborted();
        return normalizeFetched(await fetchContent(url, signal));
      })
    );
    throwIfAborted();
    if (!page.ok) {
      noteFailure(url, page.reason || "Could not be read");
      return;
    }
    pagesFetched++;
    const ids = tokenizeSource(page.text, doc);
    const runs = findSourceRuns(doc, docIndex, ids);
    await yieldToEventLoop();
    if (!runs.length) return;
    sourceRuns.set(url, { key: url, url, host, title: page.title || "", runs });
    for (const run of runs) {
      for (let i = run.start; i < run.end; i++) {
        if (!covered[i] && !doc.excluded[i]) coveredWords++;
        covered[i] = 1;
      }
    }
  }

  async function worker() {
    while (true) {
      throwIfAborted();
      if (Date.now() > deadline) {
        timeLimitReached = true;
        return;
      }
      if (executed >= budget || cursor >= queue.length) return;
      const item = queue[cursor++];
      processed++;
      if (currentSegmentCoverage(item.segment) >= COVERED_FRACTION) {
        if (!searchedSegments.has(item.segment.index)) skippedSegments.add(item.segment.index);
        continue;
      }
      if (item.pass === "alternate" && currentSegmentCoverage(item.segment) > 0) continue;
      executed++;
      searchedSegments.add(item.segment.index);
      skippedSegments.delete(item.segment.index);
      const query = queryForWindow(doc, item.window, QUERY_WORDS);
      emitProgress(`Searching phrase ${executed} of ${planned}`);

      let outcome;
      try {
        outcome = normalizeSearch(await search(query, signal));
      } catch (error) {
        if (error?.name === "AbortError") throw error;
        outcome = { urls: [], answered: false, errors: [{ engine: "search", message: error?.message || "Search failed" }] };
      }
      for (const error of outcome.errors) searchErrors.set(`${error.engine}: ${error.message}`, (searchErrors.get(`${error.engine}: ${error.message}`) || 0) + 1);
      if (outcome.answered) answeredQueries++;

      const fresh = [];
      for (const url of outcome.urls) {
        if (fresh.length >= urlsPerQuery) break;
        if (fetchedUrls.has(url) || !shouldFetchUrl(url)) continue;
        fetchedUrls.add(url);
        fresh.push(url);
      }
      const allowed = [];
      for (const url of fresh) {
        if (fetchAttempts >= maxFetches) {
          fetchLimitReached = true;
          break;
        }
        fetchAttempts++;
        allowed.push(url);
      }
      if (allowed.length) {
        emitProgress(`Reading ${allowed.length} page${allowed.length === 1 ? "" : "s"} for phrase ${executed} of ${planned}`);
        await Promise.all(allowed.map((url) => comparePage(url)));
      }
      emitProgress(
        sourceRuns.size
          ? `Phrase ${executed} of ${planned} · ${sourceRuns.size} source${sourceRuns.size === 1 ? "" : "s"} matched so far`
          : `Phrase ${executed} of ${planned} searched`
      );
    }
  }

  if (segments.length) {
    await Promise.all(Array.from({ length: Math.min(searchWorkers, Math.max(1, planned)) }, () => worker()));
  }
  throwIfAborted();

  if (executed > 0 && answeredQueries === 0) {
    const reasons = [...searchErrors.keys()].slice(0, 3).join("; ");
    const error = new Error(
      `No search engine could be reached, so nothing was checked and no score was produced. ${reasons}`.trim()
    );
    error.status = 502;
    throw error;
  }

  report("summarize", 94, "Locating matches in your text", { completed: executed, total: planned });
  await yieldToEventLoop();

  const resolved = resolveSpans(doc, [...sourceRuns.values()]);
  const sources = resolved.sources.map((source) => ({
    id: source.id,
    url: source.url,
    host: source.host,
    title: source.title,
    matchedWords: source.matchedWords,
    sharePercent: source.sharePercent,
    spanCount: source.spanCount,
    longestSpan: source.longestSpan,
    overlapWords: source.overlapWords,
  }));
  const spans = resolved.spans.map(({ start, end, words, sourceId }) => ({ start, end, words, sourceId }));
  const similarity = { ...resolved.stats, sourceCount: sources.length };

  const alreadyMatched = segments.filter(
    (segment) => !searchedSegments.has(segment.index) && (skippedSegments.has(segment.index) || currentSegmentCoverage(segment) >= COVERED_FRACTION)
  ).length;
  const segmentsSearched = searchedSegments.size;
  const segmentsNotSearched = Math.max(0, segments.length - segmentsSearched - alreadyMatched);
  const queriesFailed = executed - answeredQueries;

  const coverage = {
    segmentsTotal: segments.length,
    segmentsSearched,
    segmentsAlreadyMatched: alreadyMatched,
    segmentsNotSearched,
    segmentsWithoutQuery,
    queriesRun: executed,
    queriesFailed,
    queryBudget: budget,
    pagesFetched,
    pagesFailed: [...failureReasons.values()].reduce((a, b) => a + b, 0),
    timeLimitReached,
    fetchLimitReached,
  };

  const notes = [];
  if (!segments.length) {
    notes.push({ level: "warning", text: "The text is too short, or has too few distinctive words, to build search phrases from." });
  } else {
    let scope = `Searched ${segmentsSearched} of ${segments.length} text segments with ${executed} search phrase${executed === 1 ? "" : "s"}`;
    if (alreadyMatched) scope += `; ${alreadyMatched} more were already explained by a source found earlier`;
    scope += ".";
    notes.push({ level: segmentsNotSearched ? "warning" : "info", text: scope });
    if (segmentsNotSearched) {
      notes.push({
        level: "warning",
        text: `${segmentsNotSearched} segment${segmentsNotSearched === 1 ? " was" : "s were"} not searched (limit of ${budget} phrases per check${timeLimitReached ? ", and the time limit was reached" : ""}). Text in those parts could still match pages that were not looked up, so the real similarity can be higher than shown.`,
      });
    }
  }
  if (queriesFailed > 0) {
    notes.push({ level: "warning", text: `${queriesFailed} search request${queriesFailed === 1 ? "" : "s"} got no answer from any search engine. Results for those phrases are missing, not zero.` });
  }
  const engineStats = provider ? provider.stats() : [];
  for (const engine of engineStats) {
    if (engine.blocked || engine.disabled) {
      notes.push({
        level: "info",
        text: `${engine.name} ${engine.disabled ? "was switched off for this check" : "turned some requests away"}${engine.lastError ? ` (${engine.lastError})` : ""}. Other engines were used.`,
      });
    }
  }
  if (coverage.pagesFailed) {
    const reasons = [...failureReasons.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(([reason, count]) => `${reason} ×${count}`)
      .join(", ");
    notes.push({ level: "info", text: `${coverage.pagesFailed} candidate page${coverage.pagesFailed === 1 ? "" : "s"} could not be read (${reasons}), so they could not be compared.` });
  }
  if (fetchLimitReached) {
    notes.push({ level: "info", text: `Stopped opening new pages after ${maxFetches} to stay polite to other sites.` });
  }
  if (doc.excludedCount) {
    notes.push({ level: "info", text: `${doc.excludedCount.toLocaleString("en-US")} words in quotations or citations were excluded from the comparison and from the percentage.` });
  }

  const result = {
    version: RESULT_VERSION,
    checkedAt: new Date().toISOString(),
    similarity,
    summary: describeSimilarity(similarity, sources.length),
    sources,
    spans,
    coverage,
    engines: engineStats,
    failures,
    notes,
    settings: { excludeQuotes: Boolean(excludeQuotes), minRunWords: MIN_RUN_WORDS, shingleWords: SHINGLE_SIZE },
    totalWords: doc.count,
    characterCount: doc.text.length,
  };

  report("done", 100, "Check complete", { completed: executed, total: planned });
  return result;
}
