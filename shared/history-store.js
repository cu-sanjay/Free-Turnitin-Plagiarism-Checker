export const HISTORY_KEY = "plagcheck.history.v1";

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
    let next = entries.slice(0, maxEntries);
    let payload = JSON.stringify(next);
    while (payload.length > maxChars && next.length > 1) {
      next = next.slice(0, next.length - 1);
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

function trimSentence(sentence) {
  const clean = String(sentence || "").replace(/\s+/g, " ").trim();
  return clean.length > 500 ? `${clean.slice(0, 497)}...` : clean;
}

export function toHistoryRecord({ text, result, excludeQuotes }) {
  const sourceText = String(text || "");
  const preview = sourceText.replace(/\s+/g, " ").trim().slice(0, 180);
  return {
    id: `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`,
    createdAt: result?.checkedAt || new Date().toISOString(),
    preview,
    characterCount: result?.characterCount ?? sourceText.length,
    wordCount: result?.totalWords ?? 0,
    excludeQuotes: Boolean(excludeQuotes),
    plagiarismPercentage: result?.plagiarismPercentage ?? 0,
    overallScore: result?.overallScore ?? 0,
    totalSentences: result?.totalSentences ?? 0,
    plagiarizedSentences: result?.plagiarizedSentences ?? 0,
    passagesChecked: result?.passagesChecked ?? 0,
    passagesInDocument: result?.passagesInDocument ?? 0,
    coverageNote: result?.coverageNote || "",
    analysisSummary: result?.analysisSummary || "",
    sources: (result?.sources || []).slice(0, 12),
    results: (result?.results || []).map((item) => ({
      sentence: trimSentence(item.sentence),
      passageIndex: item.passageIndex,
      similarity: item.similarity,
      isPlagiarized: item.isPlagiarized,
      sources: (item.sources || []).slice(0, 3).map((source) => ({
        url: source.url,
        host: source.host,
        similarity: source.similarity,
        excerpt: source.excerpt ? String(source.excerpt).slice(0, 360) : "",
      })),
    })),
  };
}

export function recordToResult(record) {
  return {
    overallScore: record.overallScore,
    plagiarismPercentage: record.plagiarismPercentage,
    totalSentences: record.totalSentences,
    plagiarizedSentences: record.plagiarizedSentences,
    totalWords: record.wordCount,
    characterCount: record.characterCount,
    passagesChecked: record.passagesChecked,
    passagesInDocument: record.passagesInDocument,
    coverageNote: record.coverageNote,
    analysisSummary: record.analysisSummary,
    sources: record.sources || [],
    results: record.results || [],
    excludeQuotes: record.excludeQuotes,
    checkedAt: record.createdAt,
  };
}
