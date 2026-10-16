import assert from "node:assert/strict";
import test from "node:test";
import { buildPlagiarismPdf } from "./pdf-report.js";

function pdfText(bytes) {
  return Array.from(bytes, (byte) => String.fromCharCode(byte)).join("");
}

test("pdf report is centred, paginated, and uses the overall similarity", () => {
  const text = Array.from(
    { length: 40 },
    (_, index) => `Paragraph ${index + 1} discusses river stone glazes and kiln schedules in the studio notes.`
  ).join("\n\n");
  const copied = text.slice(0, 180);
  const pdf = buildPlagiarismPdf({
    checkedAt: "2026-10-14T12:00:00.000Z",
    text,
    summary: "31% of the document (80 of 260 words) matches 1 public source.",
    similarity: {
      overallPercent: 31.2,
      matchedWords: 80,
      countedWords: 260,
      longestMatchWords: 42,
      sourceCount: 1,
    },
    coverage: { segmentsSearched: 8, segmentsTotal: 12, queriesRun: 8 },
    notes: [{ level: "info", text: "Searched 8 of 12 text segments with 8 search phrases." }],
    sources: [
      {
        id: 1,
        url: "https://example.test/source",
        host: "example.test",
        title: "Studio notes",
        matchedWords: 80,
        sharePercent: 31.2,
      },
    ],
    spans: [{ start: 0, end: copied.length, words: 20, sourceId: 1 }],
  });

  const content = pdfText(pdf);
  assert.match(content, /^%PDF-1\.4/);
  assert.match(content, /%%EOF$/);
  assert.ok((content.match(/\/Type \/Page /g) || []).length >= 2);
  assert.match(content, /1 of /);
  assert.match(content, /Overall similarity/);
  assert.match(content, /31%/);
  assert.match(content, /example\.test\/source/);
  assert.match(content, /\[1\]/);
  assert.doesNotMatch(content, /Passages over 50% similarity/);
  assert.doesNotMatch(content, /Turnitin similarity index/i);
  assert.equal(content.includes("99%"), false);
});

test("pdf says when a saved report cannot redraw highlights", () => {
  const content = pdfText(
    buildPlagiarismPdf({
      text: "",
      textStored: false,
      similarity: { overallPercent: 12, matchedWords: 10, countedWords: 80 },
      sources: [],
      spans: [],
    })
  );
  assert.match(content, /original document was not stored/);
});
