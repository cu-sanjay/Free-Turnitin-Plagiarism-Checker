import assert from "node:assert/strict";
import test from "node:test";
import {
  calculateSimilarity,
  checkPlagiarism,
  defaultMaxPassages,
  nGramSimilarity,
  preparePassages,
  selectPassages,
} from "./plagiarism.js";

test("cosine similarity is 1 for the same text and 0 for disjoint text", () => {
  const sentence = "The committee published the annual rainfall measurements on Tuesday morning";
  assert.equal(calculateSimilarity(sentence, sentence), 1);
  assert.equal(calculateSimilarity(sentence, "quark gluon plasma lattice"), 0);
  assert.equal(calculateSimilarity("", "hello world"), 0);
});

test("n-gram similarity detects a shared phrase", () => {
  const left = "alpha beta gamma delta epsilon zeta eta theta";
  const right = "alpha beta gamma delta epsilon zeta eta theta and more words follow here";
  assert.ok(nGramSimilarity(left, right, 5) > 0.4);
  assert.equal(nGramSimilarity("one two", "three four", 5), 0);
});

test("long unpunctuated text is chunked and a spread is selected", () => {
  const words = Array.from({ length: 4000 }, (_, index) => `word${index}`);
  const passages = preparePassages(words.join(" "));
  assert.ok(passages.length > 50);
  assert.ok(passages.every((passage) => passage.split(/\s+/).length <= 80));

  const selected = selectPassages(passages, defaultMaxPassages(passages.length));
  assert.equal(selected.length, 24);
  assert.equal(selected[0].index, 0);
  assert.equal(selected.at(-1).index, passages.length - 1);
});

test("checkPlagiarism reports only real overlap from fetched pages", async () => {
  const text = `${"The harbor commission published the tidal chart for Monday morning after the storm. ".repeat(3)}Unique classroom note about ceramic glazes and kiln schedules follows here in detail.`;
  const progress = [];
  const page = `Preface. ${"The harbor commission published the tidal chart for Monday morning after the storm. ".repeat(4)} End.`;

  const result = await checkPlagiarism(text, {
    maxPassages: 2,
    onProgress: (event) => progress.push(event.percent),
    search: async () => ["https://example.test/tides"],
    fetchContent: async () => page,
  });

  assert.ok(result.results.some((item) => item.isPlagiarized));
  assert.ok(result.results.some((item) => item.similarity === 0 || !item.isPlagiarized));
  assert.equal(result.sources[0].url, "https://example.test/tides");
  assert.ok(result.sources[0].maxSimilarity > 50);
  assert.match(result.results.find((item) => item.sources.length).sources[0].excerpt, /harbor commission/i);
  assert.equal(Number.isNaN(result.overallScore), false);
  assert.ok(progress.at(-1) === 100);
  assert.ok(progress.some((value, index) => index > 0 && value >= progress[index - 1]));
});

test("empty web results do not invent a plagiarism score", async () => {
  const text = "This classroom reflection on local river stones is original phrasing for a short methods note and should not match.";
  const result = await checkPlagiarism(`${text} ${text}`, {
    search: async () => [],
    fetchContent: async () => {
      throw new Error("should not fetch");
    },
  });

  assert.equal(result.plagiarismPercentage, 0);
  assert.equal(result.overallScore, 0);
  assert.equal(result.plagiarizedSentences, 0);
  assert.equal(result.sources.length, 0);
  assert.equal(result.totalSentences, result.results.length);
});

test("too-short text returns a zero score instead of dividing by zero", async () => {
  const result = await checkPlagiarism("Too short.", {
    search: async () => ["https://example.test"],
  });
  assert.equal(result.totalSentences, 0);
  assert.equal(result.overallScore, 0);
  assert.equal(result.plagiarismPercentage, 0);
});
