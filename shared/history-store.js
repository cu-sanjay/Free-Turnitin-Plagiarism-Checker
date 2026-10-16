export const HISTORY_KEY = "plagcheck.history.v2";
const MAX_STORED_TEXT = 180_000;

function safeRead(storage) {
  try {
    const parsed = JSON.parse(storage.getItem(HISTORY_KEY) || "[]");
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function createHistoryStore(storage, { maxEntries = 20, maxChars = 900_000 } = {}) {
  if (!storage) throw new Error("History storage is required");

  const write = (entries) => {
    let next = entries.slice(0, maxEntries).map((entry) => ({ ...entry }));
    let payload = JSON.stringify(next);
    let trimIndex = next.length - 1;
    while (payload.length > maxChars && next.length) {
      const entry = next[trimIndex];
      if (entry?.text) {
        next[trimIndex] = { ...entry, text: "", textStored: false, spans: [] };
      } else if (next.length > 1) {
        next = next.slice(0, next.length - 1);
        trimIndex = next.length - 1;
      } else {
        break;
      }
      payload = JSON.stringify(next);
    }
    storage.setItem(HISTORY_KEY, payload);
    return next;
  };

  return {
    list() {
      return safeRead(storage);
    },
    save(record) {
      const entries = safeRead(storage).filter((entry) => entry.id !== record.id);
      entries.unshift(record);
      return write(entries);
    },
    remove(id) {
      return write(safeRead(storage).filter((entry) => entry.id !== id));
    },
    clear() {
      storage.removeItem(HISTORY_KEY);
      return [];
    },
  };
}

function overallOf(result) {
  if (Number.isFinite(result?.similarity?.overallPercent)) return result.similarity.overallPercent;
  if (Number.isFinite(result?.plagiarismPercentage)) return result.plagiarismPercentage;
  return 0;
}

export function toHistoryRecord({ text, result, excludeQuotes }) {
  const sourceText = String(text || "");
  const preview = sourceText.replace(/\s+/g, " ").trim().slice(0, 180);
  const textStored = sourceText.length > 0 && sourceText.length <= MAX_STORED_TEXT;
  return {
    id: `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`,
    createdAt: result?.checkedAt || new Date().toISOString(),
    preview,
    characterCount: result?.characterCount ?? sourceText.length,
    wordCount: result?.totalWords ?? result?.similarity?.totalWords ?? 0,
    excludeQuotes: Boolean(excludeQuotes || result?.settings?.excludeQuotes),
    overallPercent: overallOf(result),
    similarity: result?.similarity || { overallPercent: overallOf(result) },
    summary: result?.summary || "",
    coverage: result?.coverage || {},
    notes: result?.notes || [],
    sources: result?.sources || [],
    spans: textStored ? result?.spans || [] : [],
    failures: (result?.failures || []).slice(0, 12),
    settings: result?.settings || { excludeQuotes: Boolean(excludeQuotes) },
    text: textStored ? sourceText : "",
    textStored,
  };
}

export function recordToResult(record) {
  const overallPercent = Number.isFinite(record?.similarity?.overallPercent)
    ? record.similarity.overallPercent
    : record?.overallPercent ?? 0;
  return {
    checkedAt: record.createdAt,
    similarity: record.similarity || { overallPercent },
    summary: record.summary || "",
    sources: record.sources || [],
    spans: record.spans || [],
    coverage: record.coverage || {},
    notes: record.notes || [],
    failures: record.failures || [],
    settings: record.settings || { excludeQuotes: record.excludeQuotes },
    totalWords: record.wordCount,
    characterCount: record.characterCount,
    textStored: record.textStored !== false && Boolean(record.text),
  };
}

export function recordText(record) {
  return record?.textStored === false ? "" : String(record?.text || "");
}
