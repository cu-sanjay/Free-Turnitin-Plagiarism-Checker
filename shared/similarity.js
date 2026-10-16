// Pure similarity maths shared by the server (matching) and the client (reports, tests).
// Everything here works on word tokens with character offsets into the submitted text,
// so a matched span can be located exactly in what the user pasted.

export const SHINGLE_SIZE = 5;
export const MIN_RUN_WORDS = 7;
export const MAX_BRIDGE_GAP = 2;
const MAX_POSITIONS_PER_SHINGLE = 12;

const WORD_RE = /[\p{L}\p{N}]+(?:['’][\p{L}\p{N}]+)*/gu;
const CITATION_MARKER_RE =
  /\[(?:\d{1,3}(?:\s*[,–-]\s*\d{1,3})*|[a-z]|citation needed|note \d{1,3}|clarification needed|edit|when\?|who\?|verification needed|according to whom\?)\]/gi;
const QUOTE_RES = [/"[^"\n]{1,800}"/g, /“[^”\n]{1,800}”/g];
const CITATION_PAREN_RE = /\((?:[^()\n]{1,160})\)/g;

const STOP_WORDS = new Set(
  (
    "a an and are as at be been but by can could did do does for from had has have he her his i if in into is it its " +
    "may more most no not of on or our she should so some such than that the their them then there these they this " +
    "to was we were what when where which who will with would you your about after all also any been being both each " +
    "other over than too very while one two"
  ).split(/\s+/)
);

export function normalizeWord(word) {
  let value = word.toLowerCase().replace(/['’]/g, "");
  // eslint-disable-next-line no-control-regex
  if (/[^\u0000-\u007f]/.test(value)) {
    value = value.normalize("NFD").replace(/\p{M}/gu, "");
  }
  return value;
}

export function isStopWord(word) {
  return STOP_WORDS.has(word);
}

function markRanges(text, regexes, ranges) {
  for (const re of regexes) {
    re.lastIndex = 0;
    let match;
    while ((match = re.exec(text))) ranges.push([match.index, match.index + match[0].length]);
  }
}

function inAnyRange(sortedRanges, position, cursor) {
  let c = cursor;
  while (c < sortedRanges.length && sortedRanges[c][1] <= position) c++;
  return { cursor: c, hit: c < sortedRanges.length && sortedRanges[c][0] <= position };
}

/**
 * Split text into word tokens.
 * Bracketed citation markers such as [12] are not words and are skipped entirely.
 * With excludeQuotes, words inside quotation marks or author-year parentheses stay in the
 * token list (so offsets stay valid) but are flagged as excluded: they are neither counted
 * nor matched.
 */
export function tokenizeDocument(text, { excludeQuotes = false } = {}) {
  const source = String(text || "");
  const skipRanges = [];
  markRanges(source, [CITATION_MARKER_RE], skipRanges);
  skipRanges.sort((a, b) => a[0] - b[0]);

  const excludeRanges = [];
  if (excludeQuotes) {
    markRanges(source, QUOTE_RES, excludeRanges);
    CITATION_PAREN_RE.lastIndex = 0;
    let match;
    while ((match = CITATION_PAREN_RE.exec(source))) {
      if (/\b(?:1[5-9]|20)\d{2}\b|et al\./i.test(match[0])) {
        excludeRanges.push([match.index, match.index + match[0].length]);
      }
    }
    excludeRanges.sort((a, b) => a[0] - b[0]);
  }

  const vocab = new Map();
  const vocabWords = [];
  let capacity = Math.max(64, Math.ceil(source.length / 5));
  let starts = new Int32Array(capacity);
  let ends = new Int32Array(capacity);
  let ids = new Int32Array(capacity);
  let excluded = new Uint8Array(capacity);
  let count = 0;
  let skipCursor = 0;
  let excludeCursor = 0;

  WORD_RE.lastIndex = 0;
  let match;
  while ((match = WORD_RE.exec(source))) {
    const start = match.index;
    if (skipRanges.length) {
      const skip = inAnyRange(skipRanges, start, skipCursor);
      skipCursor = skip.cursor;
      if (skip.hit) continue;
    }
    const norm = normalizeWord(match[0]);
    if (!norm) continue;
    let id = vocab.get(norm);
    if (id === undefined) {
      id = vocabWords.length;
      vocab.set(norm, id);
      vocabWords.push(norm);
    }
    if (count === capacity) {
      capacity *= 2;
      const grow = (array, Type) => {
        const next = new Type(capacity);
        next.set(array);
        return next;
      };
      starts = grow(starts, Int32Array);
      ends = grow(ends, Int32Array);
      ids = grow(ids, Int32Array);
      excluded = grow(excluded, Uint8Array);
    }
    starts[count] = start;
    ends[count] = start + match[0].length;
    ids[count] = id;
    if (excludeRanges.length) {
      const hit = inAnyRange(excludeRanges, start, excludeCursor);
      excludeCursor = hit.cursor;
      excluded[count] = hit.hit ? 1 : 0;
    }
    count++;
  }

  let excludedCount = 0;
  for (let i = 0; i < count; i++) excludedCount += excluded[i];

  return {
    text: source,
    count,
    counted: count - excludedCount,
    excludedCount,
    starts: starts.subarray(0, count),
    ends: ends.subarray(0, count),
    ids: ids.subarray(0, count),
    excluded: excluded.subarray(0, count),
    vocab,
    vocabWords,
  };
}

function shingleHash(ids, offset, size) {
  let hash = 17;
  for (let j = 0; j < size; j++) hash = (Math.imul(hash, 1000003) + ids[offset + j] + 1) | 0;
  return hash;
}

/** Index every k-word shingle of the document (skipping shingles that touch excluded words). */
export function buildDocIndex(doc, size = SHINGLE_SIZE) {
  const matchIds = new Int32Array(doc.count);
  for (let i = 0; i < doc.count; i++) matchIds[i] = doc.excluded[i] ? -1 : doc.ids[i];
  const index = new Map();
  for (let i = 0; i + size <= doc.count; i++) {
    let valid = true;
    for (let j = 0; j < size; j++) {
      if (matchIds[i + j] < 0) {
        valid = false;
        break;
      }
    }
    if (!valid) continue;
    const key = shingleHash(matchIds, i, size);
    const bucket = index.get(key);
    if (!bucket) index.set(key, [i]);
    else if (bucket.length < MAX_POSITIONS_PER_SHINGLE) bucket.push(i);
  }
  return { size, matchIds, map: index };
}

/** Tokenize a source page into ids in the document vocabulary (-1 = word not in the document). */
export function tokenizeSource(sourceText, doc) {
  const text = String(sourceText || "").replace(CITATION_MARKER_RE, " ");
  let capacity = Math.max(64, Math.ceil(text.length / 5));
  let ids = new Int32Array(capacity);
  let count = 0;
  WORD_RE.lastIndex = 0;
  let match;
  while ((match = WORD_RE.exec(text))) {
    const norm = normalizeWord(match[0]);
    if (!norm) continue;
    if (count === capacity) {
      capacity *= 2;
      const next = new Int32Array(capacity);
      next.set(ids);
      ids = next;
    }
    const id = doc.vocab.get(norm);
    ids[count++] = id === undefined ? -1 : id;
  }
  return ids.subarray(0, count);
}

export function contentWordCount(doc, start, end) {
  let n = 0;
  for (let i = start; i < end; i++) {
    const word = doc.vocabWords[doc.ids[i]];
    if (word.length >= 3 && !STOP_WORDS.has(word) && !/^\d+$/.test(word)) n++;
  }
  return n;
}

function runIsMeaningful(doc, start, end, minRun) {
  return end - start >= minRun && contentWordCount(doc, start, end) >= 3;
}

/**
 * Find maximal exact word runs shared by the document and one source.
 * Returns runs as { start, end } token ranges in the document (end exclusive) plus the
 * location in the source. Nearby runs on the same alignment are bridged so that a single
 * edited word does not split one copied passage in two.
 */
export function findSourceRuns(doc, docIndex, sourceIds, { minRun = MIN_RUN_WORDS, maxGap = MAX_BRIDGE_GAP } = {}) {
  const size = docIndex.size;
  const matchIds = docIndex.matchIds;
  const active = new Map();
  const raw = [];

  for (let s = 0; s + size <= sourceIds.length; s++) {
    let valid = true;
    for (let j = 0; j < size; j++) {
      if (sourceIds[s + j] < 0) {
        valid = false;
        s += j;
        break;
      }
    }
    if (!valid) continue;
    const bucket = docIndex.map.get(shingleHash(sourceIds, s, size));
    if (!bucket) continue;
    for (const d of bucket) {
      let equal = true;
      for (let j = 0; j < size; j++) {
        if (matchIds[d + j] !== sourceIds[s + j]) {
          equal = false;
          break;
        }
      }
      if (!equal) continue;
      const diagonal = s - d;
      const run = active.get(diagonal);
      if (run && run.lastDoc + 1 === d) {
        run.lastDoc = d;
        run.end = d + size;
        run.srcEnd = s + size;
      } else {
        const next = { start: d, end: d + size, srcStart: s, srcEnd: s + size, lastDoc: d, diagonal };
        active.set(diagonal, next);
        raw.push(next);
      }
    }
  }

  raw.sort((a, b) => a.start - b.start || a.end - b.end);
  const merged = [];
  for (const run of raw) {
    let target = null;
    for (let i = merged.length - 1; i >= 0 && i >= merged.length - 6; i--) {
      const prev = merged[i];
      const docGap = run.start - prev.end;
      const srcGap = run.srcStart - prev.srcEnd;
      if (docGap >= 0 && docGap <= maxGap && srcGap >= 0 && srcGap <= maxGap + 2) {
        let gapClean = true;
        for (let t = prev.end; t < run.start; t++) {
          if (doc.excluded[t]) gapClean = false;
        }
        if (gapClean) {
          target = prev;
          break;
        }
      }
    }
    if (target) {
      target.end = Math.max(target.end, run.end);
      target.srcEnd = Math.max(target.srcEnd, run.srcEnd);
    } else {
      merged.push({ start: run.start, end: run.end, srcStart: run.srcStart, srcEnd: run.srcEnd });
    }
  }

  return merged.filter((run) => runIsMeaningful(doc, run.start, run.end, minRun));
}

function unionLength(runs) {
  const sorted = [...runs].sort((a, b) => a.start - b.start);
  let total = 0;
  let cursor = -1;
  for (const run of sorted) {
    const from = Math.max(run.start, cursor);
    if (run.end > from) total += run.end - from;
    cursor = Math.max(cursor, run.end);
  }
  return total;
}

/**
 * Turn raw per-source runs into non-overlapping spans, each document word attributed to at
 * most one source. The source covering the most words is served first, so mirrors and copies
 * of a page do not split credit. Remnants shorter than the minimum run are dropped.
 *
 * sourceRuns: Array<{ key, url?, runs: [{start,end}], ...meta }>
 */
export function resolveSpans(doc, sourceRuns, { minRun = MIN_RUN_WORDS } = {}) {
  const entries = sourceRuns
    .filter((entry) => entry.runs?.length)
    .map((entry, order) => ({ ...entry, order, overlapWords: unionLength(entry.runs) }))
    .sort((a, b) => b.overlapWords - a.overlapWords || a.order - b.order);

  const owner = new Int32Array(doc.count).fill(-1);
  const pieces = [];

  entries.forEach((entry, entryIndex) => {
    const sorted = [...entry.runs].sort((a, b) => a.start - b.start);
    for (const run of sorted) {
      let i = run.start;
      while (i < run.end) {
        while (i < run.end && owner[i] !== -1) i++;
        const from = i;
        while (i < run.end && owner[i] === -1) i++;
        if (i > from && runIsMeaningful(doc, from, i, minRun)) {
          for (let t = from; t < i; t++) owner[t] = entryIndex;
          pieces.push({ start: from, end: i, entryIndex });
        }
      }
    }
  });

  pieces.sort((a, b) => a.start - b.start);
  const joined = [];
  for (const piece of pieces) {
    const last = joined[joined.length - 1];
    if (last && last.entryIndex === piece.entryIndex && last.end === piece.start) last.end = piece.end;
    else joined.push({ ...piece });
  }

  const wordsBySource = new Map();
  for (const piece of joined) {
    wordsBySource.set(piece.entryIndex, (wordsBySource.get(piece.entryIndex) || 0) + (piece.end - piece.start));
  }
  const ranked = [...wordsBySource.entries()]
    .sort((a, b) => b[1] - a[1] || a[0] - b[0])
    .map(([entryIndex], rank) => ({ entryIndex, id: rank + 1 }));
  const idByEntry = new Map(ranked.map((item) => [item.entryIndex, item.id]));

  const spans = joined.map((piece) => ({
    start: doc.starts[piece.start],
    end: doc.ends[piece.end - 1],
    firstWord: piece.start,
    endWord: piece.end,
    words: piece.end - piece.start,
    sourceId: idByEntry.get(piece.entryIndex),
  }));

  const counted = Math.max(0, doc.counted);
  const sources = ranked.map(({ entryIndex, id }) => {
    const entry = entries[entryIndex];
    const own = joined.filter((piece) => piece.entryIndex === entryIndex);
    const matchedWords = own.reduce((sum, piece) => sum + (piece.end - piece.start), 0);
    const { runs: _runs, order: _order, ...meta } = entry;
    return {
      ...meta,
      id,
      matchedWords,
      sharePercent: counted ? roundTo((matchedWords / counted) * 100, 1) : 0,
      spanCount: own.length,
      longestSpan: own.reduce((max, piece) => Math.max(max, piece.end - piece.start), 0),
    };
  });

  return { spans, sources, stats: computeOverall(doc, spans) };
}

export function roundTo(value, digits = 1) {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

/**
 * The headline number: the share of the document's counted words that sit inside matched
 * spans. Spans must not overlap; each word is counted once.
 */
export function computeOverall(doc, spans) {
  let matchedWords = 0;
  let matchedChars = 0;
  let longest = 0;
  for (const span of spans) {
    let words = 0;
    for (let i = span.firstWord; i < span.endWord; i++) if (!doc.excluded[i]) words++;
    matchedWords += words;
    matchedChars += span.end - span.start;
    longest = Math.max(longest, span.words);
  }
  const countedWords = doc.counted;
  const totalChars = doc.text.length;
  return {
    overallPercent: countedWords ? roundTo((matchedWords / countedWords) * 100, 1) : 0,
    matchedWords,
    countedWords,
    totalWords: doc.count,
    excludedWords: doc.excludedCount,
    matchedChars,
    totalChars,
    charPercent: totalChars ? roundTo((matchedChars / totalChars) * 100, 1) : 0,
    longestMatchWords: longest,
    spanCount: spans.length,
  };
}

/** Whole-number label for a percentage, never rounding a real match down to 0. */
export function formatPercent(value) {
  if (!Number.isFinite(value) || value <= 0) return "0%";
  if (value < 1) return "<1%";
  if (value >= 99.5 && value < 100) return "99%";
  return `${Math.round(value)}%`;
}

// Queries --------------------------------------------------------------------------------

function windowScore(doc, start, length) {
  const seen = new Set();
  let score = 0;
  for (let i = start; i < start + length; i++) {
    const word = doc.vocabWords[doc.ids[i]];
    if (doc.excluded[i]) return -1;
    if (STOP_WORDS.has(word) || seen.has(word)) continue;
    seen.add(word);
    if (/^\d+$/.test(word)) score += 0.4;
    else if (word.length >= 8) score += 3;
    else if (word.length >= 5) score += 2;
    else if (word.length >= 3) score += 1;
  }
  return seen.size >= 4 ? score : -1;
}

/**
 * Split the document into segments of roughly segmentWords words and pick, in each, the
 * most distinctive window of queryWords words (plus a second, mostly non-overlapping one).
 */
export function planSegments(doc, { segmentWords = 30, queryWords = 12 } = {}) {
  const segments = [];
  if (doc.count < queryWords) return segments;
  const segmentCount = Math.max(1, Math.round(doc.count / segmentWords));
  for (let s = 0; s < segmentCount; s++) {
    const start = Math.floor((s * doc.count) / segmentCount);
    const end = Math.floor(((s + 1) * doc.count) / segmentCount);
    const windows = [];
    for (let w = start; w + queryWords <= Math.min(doc.count, Math.max(end, start + queryWords)); w += 2) {
      const score = windowScore(doc, w, queryWords);
      if (score >= 0) windows.push({ start: w, score });
    }
    windows.sort((a, b) => b.score - a.score || a.start - b.start);
    const primary = windows[0] || null;
    const alternate = primary ? windows.find((win) => Math.abs(win.start - primary.start) >= queryWords - 2) || null : null;
    segments.push({ index: s, start, end, primary, alternate });
  }
  return segments;
}

export function queryForWindow(doc, window, queryWords = 12) {
  const words = [];
  for (let i = window.start; i < Math.min(doc.count, window.start + queryWords); i++) {
    words.push(doc.text.slice(doc.starts[i], doc.ends[i]));
  }
  return words.join(" ");
}

/** Order 0..n-1 so that any prefix is spread evenly through the range (van der Corput). */
export function spreadOrder(n) {
  const reverse = (value) => {
    let result = 0;
    let fraction = 0.5;
    let v = value;
    while (v > 0) {
      if (v & 1) result += fraction;
      fraction /= 2;
      v >>= 1;
    }
    return result;
  };
  return Array.from({ length: n }, (_, i) => i).sort((a, b) => reverse(a) - reverse(b) || a - b);
}

/**
 * Walk the submitted text left to right, emitting contiguous runs of unmatched or
 * matched characters. A matched run carries sourceId and a [n] marker to place after it.
 */
export function buildHighlightRuns(text, spans) {
  const source = String(text || "");
  const ordered = [...(spans || [])]
    .filter((span) => Number(span?.end) > Number(span?.start))
    .sort((a, b) => a.start - b.start || a.end - b.end);
  const runs = [];
  let cursor = 0;
  for (const span of ordered) {
    const start = Math.max(0, Math.min(source.length, span.start));
    const end = Math.max(start, Math.min(source.length, span.end));
    if (start > cursor) runs.push({ text: source.slice(cursor, start), sourceId: null, marker: null });
    if (end > start) {
      runs.push({ text: source.slice(start, end), sourceId: span.sourceId || null, marker: span.sourceId || null });
    }
    cursor = Math.max(cursor, end);
  }
  if (cursor < source.length) runs.push({ text: source.slice(cursor), sourceId: null, marker: null });
  return runs;
}

/** Split highlight runs on newlines so the report can keep the pasted paragraph breaks. */
export function runsToParagraphs(runs) {
  const paragraphs = [[]];
  for (const run of runs) {
    const parts = String(run.text || "").split("\n");
    parts.forEach((part, index) => {
      if (part) {
        paragraphs[paragraphs.length - 1].push({
          text: part,
          sourceId: run.sourceId,
          marker: index === parts.length - 1 ? run.marker : null,
        });
      }
      if (index < parts.length - 1) paragraphs.push([]);
    });
  }
  return paragraphs;
}
