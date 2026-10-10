import test from "node:test";
import assert from "node:assert/strict";
import { compareAgainstSources } from "./plagiarism.js";
import { checkTextSchema } from "../shared/schema.js";

const source = (id, name, text) => ({ id, name, text });

test("finds exact phrase overlap regardless of case and punctuation", () => {
  const draft = "A study found that public green spaces can improve mental health in crowded cities.";
  const references = [
    source(
      "ref-1",
      "Green cities",
      "Recent research found that public green spaces can improve mental health in crowded cities."
    ),
  ];

  const result = compareAgainstSources(draft, references);

  assert.equal(result.wordCount, 14);
  assert.equal(result.matchedWords, 12);
  assert.equal(result.sources[0].matches[0].text, "found that public green spaces can improve mental health in crowded cities");
  assert.equal(result.sources[0].matches[0].referenceText, "found that public green spaces can improve mental health in crowded cities");
  assert.equal(result.highlightRanges.length, 1);
});

test("does not count unrelated text or matches shorter than five words", () => {
  const result = compareAgainstSources(
    "The study found public gardens improve overall wellness for people living nearby.",
    [source("ref-1", "Reference", "Public gardens improve health for residents in neighborhoods.")]
  );

  assert.equal(result.matchedWords, 0);
  assert.equal(result.similarityPercentage, 0);
  assert.deepEqual(result.sources, []);
  assert.deepEqual(result.highlightRanges, []);
});

test("supports a stricter minimum phrase length to reduce short common matches", () => {
  const phrase = "shared wording can appear in many everyday";
  const references = [source("threshold", "Reference", phrase)];
  const draft = `${phrase} but the remaining explanation is unique.`;

  const broadResult = compareAgainstSources(draft, references, { minMatchWords: 5 });
  const strictResult = compareAgainstSources(draft, references, { minMatchWords: 8 });

  assert.equal(broadResult.minMatchWords, 5);
  assert.ok(broadResult.matchedWords > 0);
  assert.equal(strictResult.minMatchWords, 8);
  assert.equal(strictResult.matchedWords, 0);
});

test("counts overlapping matches only once and preserves matching source names", () => {
  const phrase = "local communities need reliable access to clean drinking water";
  const result = compareAgainstSources(phrase, [
    source("a", "Report A", phrase),
    source("b", "Report B", `Research shows ${phrase}.`),
  ]);

  assert.equal(result.matchedWords, 9);
  assert.equal(result.similarityPercentage, 100);
  assert.equal(result.highlightRanges.length, 1);
  assert.deepEqual(result.highlightRanges[0].sources.sort(), ["Report A", "Report B"]);
});

test("keeps Unicode offsets aligned for annotated draft text", () => {
  const draft = "Résumé: cafés improve a city's social life and local economy.";
  const phrase = "cafés improve a city's social life and local economy";
  const result = compareAgainstSources(draft, [
    source("unicode", "Local study", `The study notes ${phrase}.`),
  ]);
  const highlight = result.highlightRanges[0];

  assert.equal(draft.slice(highlight.start, highlight.end), phrase);
});

test("excludes quoted phrases from the score when requested", () => {
  const quoted = "A careful local analysis found that adding native trees reduces summer heat.";
  const draft = `The report states, "${quoted}" This independent conclusion remains under review.`;
  const references = [source("quote", "Quoted research", quoted)];

  assert.ok(compareAgainstSources(draft, references).matchedWords > 0);
  const result = compareAgainstSources(draft, references, {
    excludeQuotes: true,
    excludeReferences: false,
  });
  assert.equal(result.matchedWords, 0);
  assert.equal(result.excludedWords, 12);
});

test("excludes bibliography text without excluding the main draft", () => {
  const mainDraft = "Reliable public transit helps residents reach work and school more easily.";
  const citation = "Local communities benefit when city parks provide shade and safe walking paths.";
  const draft = `${mainDraft}\n\nReferences\n${citation}`;
  const result = compareAgainstSources(draft, [source("ref", "Reference", citation)], {
    excludeQuotes: false,
    excludeReferences: true,
  });

  assert.equal(result.matchedWords, 0);
  assert.equal(result.excludedWords, 13);
  assert.equal(result.wordCount, 11);
});

test("skips a manually excluded reference source", () => {
  const phrase = "original phrase should only match the first source";
  const result = compareAgainstSources(phrase, [
    source("included", "Included", phrase),
    { ...source("excluded", "Excluded", phrase), exclude: true },
  ]);

  assert.equal(result.similarityPercentage, 100);
  assert.equal(result.sources.length, 1);
  assert.equal(result.sources[0].name, "Included");
  assert.equal(result.excludedSourceCount, 1);
});

test("validates draft and user-provided reference limits", () => {
  const validText = "A sufficiently long draft text for a valid check. ".repeat(3);
  const validSource = source("ref", "Reference", "A reference body.");

  assert.equal(checkTextSchema.safeParse({ text: validText, sources: [validSource] }).success, true);
  assert.equal(
    checkTextSchema.safeParse({
      text: validText,
      sources: [validSource],
      options: { minMatchWords: 8 },
    }).success,
    true
  );
  assert.equal(checkTextSchema.safeParse({ text: "too short", sources: [validSource] }).success, false);
  assert.equal(checkTextSchema.safeParse({ text: validText, sources: [] }).success, false);
  assert.equal(
    checkTextSchema.safeParse({
      text: validText,
      sources: Array.from({ length: 25 }, (_, index) =>
        source(`ref-${index}`, `Reference ${index}`, "Reference body.")
      ),
    }).success,
    true
  );
  assert.equal(
    checkTextSchema.safeParse({
      text: validText,
      sources: Array.from({ length: 26 }, (_, index) =>
        source(`ref-${index}`, `Reference ${index}`, "Reference body.")
      ),
    }).success,
    false
  );
  assert.equal(
    checkTextSchema.safeParse({
      text: validText,
      sources: [{ ...validSource, url: "javascript:alert(1)" }],
    }).success,
    false
  );
  assert.equal(
    checkTextSchema.safeParse({
      text: validText,
      sources: [validSource],
      options: { minMatchWords: 7 },
    }).success,
    false
  );
});
