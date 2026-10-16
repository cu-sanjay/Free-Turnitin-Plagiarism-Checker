import assert from "node:assert/strict";
import test from "node:test";
import {
  buildDocIndex,
  buildHighlightRuns,
  computeOverall,
  findSourceRuns,
  formatPercent,
  resolveSpans,
  runsToParagraphs,
  tokenizeDocument,
  tokenizeSource,
} from "./similarity.js";

function words(count, prefix = "word") {
  return Array.from({ length: count }, (_, i) => `${prefix}${i}`).join(" ");
}

function matchAll(doc, sourceText) {
  const index = buildDocIndex(doc);
  const ids = tokenizeSource(sourceText, doc);
  return findSourceRuns(doc, index, ids);
}

test("overall similarity is the share of the document's words covered by matched spans", () => {
  const source = words(20, "alpha");
  const doc = tokenizeDocument(`${source} ${words(20, "beta")}`);
  const runs = matchAll(doc, source);
  const { spans, stats } = resolveSpans(doc, [{ key: "a", url: "https://example.test/a", runs }]);
  assert.equal(doc.counted, 40);
  assert.equal(stats.matchedWords, 20);
  assert.equal(stats.overallPercent, 50);
  assert.equal(spans.length, 1);
  assert.equal(formatPercent(stats.overallPercent), "50%");
});

test("overall similarity is not an average of per-passage scores", () => {
  const copied = words(90, "copy");
  const original = words(10, "orig");
  const doc = tokenizeDocument(`${copied} ${original}`);
  const { stats } = resolveSpans(doc, [{ key: "a", url: "https://example.test/a", runs: matchAll(doc, copied) }]);
  assert.equal(stats.matchedWords, 90);
  assert.equal(stats.overallPercent, 90);
  assert.notEqual(stats.overallPercent, 50);
});

test("span matching locates the longest common word run inside the submitted text", () => {
  const phrase = "the harbor commission published the tidal chart for monday morning";
  const doc = tokenizeDocument(`Preface. ${phrase} after the storm stayed local.`);
  const runs = matchAll(doc, `Unrelated intro. ${phrase} extra words on the page.`);
  assert.ok(runs.length >= 1);
  const run = runs.reduce((best, item) => (item.end - item.start > best.end - best.start ? item : best));
  assert.ok(run.end - run.start >= 10);
  const excerpt = doc.text.slice(doc.starts[run.start], doc.ends[run.end - 1]);
  assert.match(excerpt, /harbor commission published the tidal chart/i);
});

test("overlapping sources keep each word with the source that covers more of the document", () => {
  const text = `${words(12, "share")} ${words(8, "onlya")} ${words(4, "tail")}`;
  const doc = tokenizeDocument(text);
  const a = matchAll(doc, `${words(12, "share")} ${words(8, "onlya")}`);
  const b = matchAll(doc, words(12, "share"));
  const { sources, spans, stats } = resolveSpans(doc, [
    { key: "a", url: "https://example.test/long", runs: a },
    { key: "b", url: "https://example.test/short", runs: b },
  ]);
  assert.equal(sources[0].url, "https://example.test/long");
  assert.equal(sources[0].id, 1);
  assert.equal(sources.length, 1);
  assert.equal(spans.every((span) => span.sourceId === 1), true);
  assert.equal(stats.matchedWords, 20);
});

test("quoted words can be excluded from the overall percentage", () => {
  const quoted = "the harbor commission published the tidal chart for monday";
  const rest = words(10, "studio");
  const doc = tokenizeDocument(`"${quoted}" ${rest}`, { excludeQuotes: true });
  const { stats } = resolveSpans(doc, [
    { key: "a", url: "https://example.test/a", runs: matchAll(doc, `${quoted} ${rest}`) },
  ]);
  assert.equal(stats.excludedWords > 0, true);
  assert.equal(stats.matchedWords, 10);
  assert.equal(stats.overallPercent, 100);
});

test("highlight runs keep paragraph breaks and place a marker after each match", () => {
  const text = "Hello world.\n\nSecond paragraph here.";
  const runs = buildHighlightRuns(text, [{ start: 0, end: 11, sourceId: 1 }]);
  assert.equal(runs[0].text, "Hello world");
  assert.equal(runs[0].marker, 1);
  const paragraphs = runsToParagraphs(runs);
  assert.equal(paragraphs.length, 3);
  assert.equal(paragraphs[0][0].marker, 1);
  assert.equal(paragraphs[1].length, 0);
});
