import assert from "node:assert/strict";
import test from "node:test";
import { checkPlagiarism, defaultQueryBudget, describeSimilarity } from "./plagiarism.js";
import { formatPercent } from "../shared/similarity.js";

const COPIED = "The harbor commission published the tidal chart for Monday morning after the storm.";
const ORIGINAL = "Unique classroom note about ceramic glazes and kiln schedules follows here in detail for the studio.";

test("checkPlagiarism reports overall word coverage, not a passage average or a 50% flag rate", async () => {
  const copiedBlock = Array.from({ length: 12 }, () => COPIED).join(" ");
  const originalBlock = Array.from({ length: 4 }, () => ORIGINAL).join(" ");
  const text = `${copiedBlock} ${originalBlock}`;
  const page = `Preface. ${copiedBlock} End.`;
  const progress = [];

  const result = await checkPlagiarism(text, {
    maxQueries: 6,
    onProgress: (event) => progress.push(event.percent),
    search: async () => ["https://example.test/tides"],
    fetchContent: async () => ({ ok: true, text: page, title: "Tides" }),
  });

  const copiedWords = copiedBlock.split(/\s+/).length;
  const totalWords = text.split(/\s+/).filter(Boolean).length;
  const expected = Math.round((copiedWords / totalWords) * 1000) / 10;

  assert.equal(result.plagiarismPercentage, undefined);
  assert.ok(Math.abs(result.similarity.overallPercent - expected) < 5);
  assert.ok(result.similarity.overallPercent > 50);
  assert.ok(result.similarity.matchedWords >= copiedWords - 5);
  assert.equal(result.sources[0].url, "https://example.test/tides");
  assert.ok(result.spans.length >= 1);
  assert.equal(result.spans[0].sourceId, 1);
  assert.match(text.slice(result.spans[0].start, result.spans[0].end), /harbor commission/i);
  assert.ok(progress.at(-1) === 100);
  assert.ok(progress.some((value, index) => index > 0 && value >= progress[index - 1]));
});

test("empty web results do not invent a score or a source", async () => {
  const text = `${ORIGINAL} ${ORIGINAL}`;
  const result = await checkPlagiarism(text, {
    search: async () => [],
    fetchContent: async () => {
      throw new Error("should not fetch");
    },
  });
  assert.equal(result.similarity.overallPercent, 0);
  assert.equal(result.similarity.matchedWords, 0);
  assert.equal(result.sources.length, 0);
  assert.equal(result.spans.length, 0);
});

test("a fetch that fails is reported and is not treated as a match", async () => {
  const text = `${COPIED} ${COPIED} ${ORIGINAL}`;
  const result = await checkPlagiarism(text, {
    search: async () => ["https://example.test/blocked"],
    fetchContent: async () => ({ ok: false, reason: "HTTP 403" }),
  });
  assert.equal(result.similarity.overallPercent, 0);
  assert.equal(result.sources.length, 0);
  assert.ok(result.failures.some((item) => item.reason === "HTTP 403"));
  assert.ok(result.notes.some((note) => /could not be read/i.test(note.text)));
});

test("too-short text returns a zero score instead of dividing by zero", async () => {
  const result = await checkPlagiarism("Too short.", {
    search: async () => ["https://example.test"],
  });
  assert.equal(result.similarity.overallPercent, 0);
  assert.equal(result.similarity.countedWords > 0 ? result.spans.length : 0, result.spans.length);
  assert.equal(result.sources.length, 0);
});

test("coverage notes say how many segments were searched", async () => {
  const text = `${COPIED} ${ORIGINAL} ${COPIED}`;
  const result = await checkPlagiarism(text, {
    maxQueries: 2,
    search: async () => ["https://example.test/tides"],
    fetchContent: async () => ({ ok: true, text: COPIED.repeat(4), title: "Tides" }),
  });
  assert.ok(result.coverage.segmentsTotal >= 1);
  assert.ok(result.coverage.queriesRun >= 1);
  assert.ok(result.notes.some((note) => /Searched \d+ of \d+ text segments/.test(note.text)));
});

test("describeSimilarity uses the overall word share", () => {
  const stats = { overallPercent: 31.2, matchedWords: 312, countedWords: 1000 };
  assert.match(describeSimilarity(stats, 2), /31%/);
  assert.equal(formatPercent(0), "0%");
  assert.equal(formatPercent(0.4), "<1%");
});

test("query budget grows with document size but stays capped", () => {
  assert.equal(defaultQueryBudget(100), 25);
  assert.ok(defaultQueryBudget(20_000) <= 60);
});
