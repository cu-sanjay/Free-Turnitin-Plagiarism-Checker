import assert from "node:assert/strict";
import test from "node:test";
import { buildPlagiarismPdf } from "./pdf-report.js";

test("pdf report is paginated and contains only supplied figures", () => {
  const results = Array.from({ length: 18 }, (_, index) => ({
    sentence: `Passage ${index + 1} discusses river stone glazes and kiln schedules in the studio notes.`,
    similarity: index === 0 ? 82 : 4,
    isPlagiarized: index === 0,
    sources:
      index === 0
        ? [{ url: "https://example.test/source", similarity: 82, excerpt: "river stone glazes and kiln schedules" }]
        : [],
  }));

  const pdf = buildPlagiarismPdf({
    checkedAt: "2026-10-14T12:00:00.000Z",
    plagiarismPercentage: 6,
    overallScore: 8,
    totalWords: 2400,
    characterCount: 14000,
    passagesChecked: 18,
    passagesInDocument: 120,
    plagiarizedSentences: 1,
    excludeQuotes: true,
    coverageNote: "Compared 18 of 120 passages, spaced from the start, middle, and end.",
    analysisSummary: "Scores cover 18 of 120 passages.",
    sources: [{ url: "https://example.test/source", host: "example.test", maxSimilarity: 82, matchCount: 1 }],
    results,
  });

  const text = Array.from(pdf, (byte) => String.fromCharCode(byte)).join("");
  assert.match(text, /^%PDF-1\.4/);
  assert.match(text, /%%EOF$/);
  assert.equal((text.match(/\/Type \/Page /g) || []).length, (text.match(/\/Type \/Page /g) || []).length);
  assert.ok((text.match(/\/Type \/Page /g) || []).length >= 2);
  assert.match(text, /1 \/ \d+/);
  assert.match(text, /Passages over 50% similarity: 6%/);
  assert.match(text, /example\.test\/source/);
  assert.doesNotMatch(text, /Turnitin similarity index/i);
  assert.equal(text.includes("99%"), false);
});
