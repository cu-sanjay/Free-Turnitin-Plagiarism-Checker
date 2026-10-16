import assert from "node:assert/strict";
import test from "node:test";
import { createHistoryStore, toHistoryRecord } from "./history-store.js";

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
  store.save({ id: "a", plagiarismPercentage: 1 });
  store.save({ id: "b", plagiarismPercentage: 2 });
  store.save({ id: "c", plagiarismPercentage: 3 });
  assert.deepEqual(
    store.list().map((entry) => entry.id),
    ["c", "b"]
  );
  store.remove("b");
  assert.equal(store.list().length, 1);
  store.clear();
  assert.deepEqual(store.list(), []);
});

test("history records store scores and a short preview, not the full document", () => {
  const text = `${"alpha ".repeat(5000)}ending`;
  const record = toHistoryRecord({
    text,
    excludeQuotes: true,
    result: {
      plagiarismPercentage: 10,
      overallScore: 12,
      totalWords: 5001,
      characterCount: text.length,
      passagesChecked: 2,
      passagesInDocument: 40,
      checkedAt: "2026-10-14T00:00:00.000Z",
      results: [{ sentence: "alpha alpha", similarity: 10, isPlagiarized: false, sources: [] }],
    },
  });
  assert.ok(record.preview.length <= 180);
  assert.ok(record.preview.length < text.length);
  assert.equal(record.plagiarismPercentage, 10);
  assert.equal(record.excludeQuotes, true);
  assert.equal(JSON.stringify(record).includes(text), false);
});
