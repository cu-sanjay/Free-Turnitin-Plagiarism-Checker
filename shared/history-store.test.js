import assert from "node:assert/strict";
import test from "node:test";
import { createHistoryStore, recordText, recordToResult, toHistoryRecord } from "./history-store.js";

function memoryStorage() {
  const data = new Map();
  return {
    getItem(key) {
      return data.has(key) ? data.get(key) : null;
    },
    setItem(key, value) {
      data.set(key, String(value));
    },
    removeItem(key) {
      data.delete(key);
    },
  };
}

test("history keeps newest checks locally and drops the oldest when full", () => {
  const store = createHistoryStore(memoryStorage(), { maxEntries: 2 });
  store.save({ id: "a", overallPercent: 1 });
  store.save({ id: "b", overallPercent: 2 });
  store.save({ id: "c", overallPercent: 3 });
  assert.deepEqual(
    store.list().map((entry) => entry.id),
    ["c", "b"]
  );
  store.remove("b");
  assert.equal(store.list().length, 1);
  store.clear();
  assert.deepEqual(store.list(), []);
});

test("history stores the overall similarity and enough text to reopen highlights", () => {
  const text = "The harbor commission published the tidal chart for Monday morning after the storm.";
  const record = toHistoryRecord({
    text,
    excludeQuotes: true,
    result: {
      checkedAt: "2026-10-14T00:00:00.000Z",
      similarity: { overallPercent: 62.5, matchedWords: 10, countedWords: 16, longestMatchWords: 10 },
      summary: "63% of the document matches 1 public source.",
      sources: [{ id: 1, url: "https://example.test", host: "example.test", matchedWords: 10, sharePercent: 62.5 }],
      spans: [{ start: 0, end: 40, words: 8, sourceId: 1 }],
      coverage: { segmentsSearched: 2, segmentsTotal: 2 },
      totalWords: 16,
      characterCount: text.length,
    },
  });
  assert.equal(record.overallPercent, 62.5);
  assert.equal(record.textStored, true);
  assert.equal(record.text, text);
  const result = recordToResult(record);
  assert.equal(result.similarity.overallPercent, 62.5);
  assert.equal(recordText(record), text);
  assert.equal(result.spans[0].sourceId, 1);
});

test("history does not fake highlights when the document is too large to store", () => {
  const text = "alpha ".repeat(40_000);
  const record = toHistoryRecord({
    text,
    result: {
      similarity: { overallPercent: 10, matchedWords: 1, countedWords: 10 },
      spans: [{ start: 0, end: 5, words: 1, sourceId: 1 }],
      sources: [{ id: 1, url: "https://example.test", host: "example.test" }],
    },
  });
  assert.equal(record.textStored, false);
  assert.equal(record.text, "");
  assert.deepEqual(record.spans, []);
  assert.equal(record.overallPercent, 10);
});
