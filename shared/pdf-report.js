import { buildHighlightRuns, formatPercent, runsToParagraphs } from "./similarity.js";
import { sourceColor } from "./source-colors.js";

const PAGE_WIDTH = 595.28;
const PAGE_HEIGHT = 841.89;
const MARGIN_X = 54;
const MARGIN_TOP = 62;
const MARGIN_BOTTOM = 50;
const CONTENT_WIDTH = PAGE_WIDTH - MARGIN_X * 2;
const LINE_GAP = 3.2;
const PARAGRAPH_GAP = 10;

const HELVETICA = [
  278, 278, 355, 556, 556, 889, 667, 191, 333, 333, 389, 584, 278, 333, 278, 278, 556, 556, 556, 556, 556, 556, 556,
  556, 556, 556, 278, 278, 584, 584, 584, 556, 1015, 667, 667, 722, 722, 667, 611, 778, 722, 278, 500, 667, 556, 833,
  722, 778, 667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 278, 278, 278, 469, 556, 333, 556, 556, 500, 556,
  556, 278, 556, 556, 222, 222, 500, 222, 833, 556, 556, 556, 556, 333, 500, 278, 556, 500, 722, 500, 500, 500, 334,
  260, 334, 584,
];

const WINANSI = new Map([
  [0x20ac, 128],
  [0x201a, 130],
  [0x192, 131],
  [0x201e, 132],
  [0x2026, 133],
  [0x2020, 134],
  [0x2021, 135],
  [0x2c6, 136],
  [0x2030, 137],
  [0x160, 138],
  [0x2039, 139],
  [0x152, 140],
  [0x17d, 142],
  [0x2018, 145],
  [0x2019, 146],
  [0x201c, 147],
  [0x201d, 148],
  [0x2022, 149],
  [0x2013, 150],
  [0x2014, 151],
  [0x2dc, 152],
  [0x2122, 153],
  [0x161, 154],
  [0x203a, 155],
  [0x153, 156],
  [0x17e, 158],
  [0x178, 159],
  [0xa0, 32],
]);

function toWinAnsi(text) {
  let out = "";
  for (const char of String(text || "")) {
    const code = char.codePointAt(0);
    if (code === 10 || code === 13) {
      out += " ";
      continue;
    }
    if (code >= 32 && code <= 126) {
      out += char;
      continue;
    }
    const mapped = WINANSI.get(code);
    if (mapped) {
      out += String.fromCharCode(mapped);
      continue;
    }
    if (code > 126 && code < 256) {
      out += char;
      continue;
    }
    out += "?";
  }
  return out;
}

function pdfEscape(value) {
  return toWinAnsi(value)
    .split("")
    .map((char) => {
      if (char === "\\") return "\\\\";
      if (char === "(") return "\\(";
      if (char === ")") return "\\)";
      const code = char.charCodeAt(0);
      if (code < 32 || code > 126) return `\\${code.toString(8).padStart(3, "0")}`;
      return char;
    })
    .join("");
}

function glyphWidth(char, bold) {
  const code = char.charCodeAt(0);
  const index = code - 32;
  const base = index >= 0 && index < HELVETICA.length ? HELVETICA[index] : 556;
  return bold ? Math.round(base * 1.045) : base;
}

function textWidth(text, size, bold = false) {
  let width = 0;
  for (const char of toWinAnsi(text)) width += glyphWidth(char, bold);
  return (width * size) / 1000;
}

function formatWhen(value) {
  const date = value ? new Date(value) : new Date();
  if (Number.isNaN(date.getTime())) return "Unknown time";
  return date.toISOString().replace("T", " ").replace(/\.\d{3}Z$/, " UTC");
}

function wrapPieces(pieces, size, width, bold = false) {
  const lines = [];
  let current = [];
  let currentWidth = 0;
  const commit = () => {
    if (current.length) lines.push(current);
    current = [];
    currentWidth = 0;
  };
  const pushToken = (token) => {
    const extra = current.length ? textWidth(" ", size, bold) : 0;
    if (current.length && currentWidth + extra + token.width > width) commit();
    else if (current.length) currentWidth += extra;
    current.push(token);
    currentWidth += token.width;
  };
  for (const piece of pieces) {
    const words = String(piece.text || "").split(/\s+/).filter((word) => word.length);
    words.forEach((word, index) => {
      const marker = index === words.length - 1 ? piece.marker : null;
      const labelled = marker ? `${word} [${marker}]` : word;
      let remaining = labelled;
      while (remaining.length) {
        let fit = remaining;
        while (fit.length > 1 && textWidth(fit, size, bold) > width) fit = fit.slice(0, -1);
        const token = {
          text: fit,
          width: textWidth(fit, size, bold),
          sourceId: piece.sourceId || null,
          marker: fit === labelled ? marker : null,
        };
        pushToken(token);
        remaining = remaining.slice(fit.length);
      }
    });
  }
  commit();
  return lines.length ? lines : [[]];
}

function wrapPlain(text, size, width, bold = false) {
  return wrapPieces([{ text, sourceId: null, marker: null }], size, width, bold);
}

class Document {
  constructor() {
    this.pages = [];
    this.newPage();
  }

  newPage() {
    this.ops = [];
    this.pages.push(this.ops);
    this.y = PAGE_HEIGHT - MARGIN_TOP;
    this.pageHasBody = false;
  }

  remaining() {
    return this.y - MARGIN_BOTTOM;
  }

  need(height) {
    if (this.pageHasBody && this.remaining() < height) this.newPage();
  }

  gap(height) {
    this.need(height);
    this.y -= height;
  }

  line(color = [0.82, 0.84, 0.86], width = CONTENT_WIDTH) {
    this.need(8);
    const y = this.y - 2;
    this.ops.push(`${color.join(" ")} RG 0.6 w ${MARGIN_X} ${y.toFixed(2)} m ${MARGIN_X + width} ${y.toFixed(2)} l S`);
    this.y -= 10;
  }

  rect(x, y, w, h, rgb, alpha = 1) {
    if (alpha < 1) {
      this.ops.push(`/GS1 gs ${rgb.join(" ")} rg ${x.toFixed(2)} ${y.toFixed(2)} ${w.toFixed(2)} ${h.toFixed(2)} re f`);
      this.ops.push("/GS0 gs");
      return;
    }
    this.ops.push(`${rgb.join(" ")} rg ${x.toFixed(2)} ${y.toFixed(2)} ${w.toFixed(2)} ${h.toFixed(2)} re f`);
  }

  textAt(text, x, y, { size = 11, bold = false, color = [0.1, 0.1, 0.12] } = {}) {
    this.ops.push("BT");
    this.ops.push(`/${bold ? "F2" : "F1"} ${size} Tf`);
    this.ops.push(`${color.join(" ")} rg`);
    this.ops.push(`1 0 0 1 ${x.toFixed(2)} ${y.toFixed(2)} Tm`);
    this.ops.push(`(${pdfEscape(text)}) Tj`);
    this.ops.push("ET");
  }

  centered(text, size, bold, color, y) {
    const x = (PAGE_WIDTH - textWidth(text, size, bold)) / 2;
    this.textAt(text, Math.max(MARGIN_X, x), y, { size, bold, color });
  }

  flowPlain(text, { size = 11, bold = false, color = [0.12, 0.12, 0.14], width = CONTENT_WIDTH, indent = 0, gap = LINE_GAP } = {}) {
    const lineHeight = size + gap;
    const lines = wrapPlain(text, size, width, bold);
    for (const line of lines) {
      this.need(lineHeight + 1);
      const y = this.y - size;
      const joined = line.map((token) => token.text).join(" ");
      this.textAt(joined, MARGIN_X + indent, y, { size, bold, color });
      this.y -= lineHeight;
      this.pageHasBody = true;
    }
  }

  flowRuns(pieces, { size = 10.5, width = CONTENT_WIDTH } = {}) {
    const lineHeight = size + LINE_GAP + 1.5;
    const lines = wrapPieces(pieces, size, width, false);
    if (lines.length > 2 && this.remaining() < lineHeight * 2 && this.pageHasBody) this.newPage();
    for (const line of lines) {
      this.need(lineHeight + 1);
      const baseline = this.y - size;
      let x = MARGIN_X;
      line.forEach((token, index) => {
        if (index) x += textWidth(" ", size, false);
        if (token.sourceId) {
          const color = sourceColor(token.sourceId);
          this.rect(x - 0.6, baseline - 2.2, token.width + 1.2, size + 3.4, color.rgb, 0.22);
          this.ops.push(`${color.rgb.join(" ")} RG 1 w ${x.toFixed(2)} ${(baseline - 1.4).toFixed(2)} m ${(x + token.width).toFixed(2)} ${(baseline - 1.4).toFixed(2)} l S`);
        }
        this.textAt(token.text, x, baseline, { size, color: [0.08, 0.08, 0.1] });
        x += token.width;
      });
      this.y -= lineHeight;
      this.pageHasBody = true;
    }
    this.y -= PARAGRAPH_GAP;
  }
}

function headerAndFooter(pageOps, pageNumber, pageCount) {
  const header = [
    "0.88 0.9 0.92 RG 0.6 w",
    `${MARGIN_X} ${PAGE_HEIGHT - 36} m ${PAGE_WIDTH - MARGIN_X} ${PAGE_HEIGHT - 36} l S`,
    "BT /F2 9 Tf 0.35 0.38 0.42 rg",
    `1 0 0 1 ${MARGIN_X} ${PAGE_HEIGHT - 28} Tm`,
    "(Turnitout) Tj",
    "ET",
    "BT /F1 9 Tf 0.35 0.38 0.42 rg",
    `1 0 0 1 ${PAGE_WIDTH - MARGIN_X - textWidth("Similarity report", 9)} ${PAGE_HEIGHT - 28} Tm`,
    "(Similarity report) Tj",
    "ET",
    "0.88 0.9 0.92 RG 0.6 w",
    `${MARGIN_X} 38 m ${PAGE_WIDTH - MARGIN_X} 38 l S`,
    "BT /F1 8 Tf 0.4 0.42 0.45 rg",
    `1 0 0 1 ${MARGIN_X} 24 Tm`,
    "(Public-web overlap only. Not a Turnitin report.) Tj",
    "ET",
    "BT /F1 8 Tf 0.4 0.42 0.45 rg",
    `1 0 0 1 ${PAGE_WIDTH - MARGIN_X - textWidth(`${pageNumber} of ${pageCount}`, 8)} 24 Tm`,
    `(${pdfEscape(`${pageNumber} of ${pageCount}`)}) Tj`,
    "ET",
  ];
  return `${header.join("\n")}\n${pageOps.join("\n")}\n`;
}

function byteLength(value) {
  return new TextEncoder().encode(value).length;
}

function overallOf(report) {
  return report?.similarity?.overallPercent ?? report?.plagiarismPercentage ?? 0;
}

export function buildPlagiarismPdf(report = {}) {
  const doc = new Document();
  const similarity = report.similarity || {};
  const sources = Array.isArray(report.sources) ? report.sources : [];
  const text = String(report.text || "");
  const spans = Array.isArray(report.spans) ? report.spans : [];
  const coverage = report.coverage || {};
  const percentLabel = formatPercent(overallOf(report));

  doc.gap(18);
  doc.centered("Turnitout", 11, true, [0.2, 0.32, 0.42], doc.y);
  doc.gap(26);
  doc.centered("Similarity report", 22, true, [0.07, 0.09, 0.12], doc.y);
  doc.gap(28);
  doc.centered("Public-web overlap only. Not a Turnitin report.", 10, false, [0.36, 0.38, 0.4], doc.y);
  doc.gap(18);
  doc.centered(`Generated ${formatWhen(report.checkedAt)}`, 10, false, [0.36, 0.38, 0.4], doc.y);
  doc.gap(16);
  doc.line();
  doc.gap(46);
  doc.centered(percentLabel, 36, true, [0.07, 0.09, 0.12], doc.y);
  doc.gap(48);
  doc.centered("Overall similarity", 11, false, [0.32, 0.34, 0.36], doc.y);
  doc.gap(22);
  if (report.summary) {
    doc.flowPlain(report.summary, { size: 11, color: [0.16, 0.18, 0.2] });
    doc.gap(6);
  }
  doc.gap(4);

  const stats = [
    `Matched words  ${Number(similarity.matchedWords || 0).toLocaleString("en-US")} of ${Number(similarity.countedWords || report.totalWords || 0).toLocaleString("en-US")}`,
    `Sources  ${sources.length}`,
    `Longest match  ${Number(similarity.longestMatchWords || 0).toLocaleString("en-US")} words`,
    `Coverage  searched ${coverage.segmentsSearched ?? "—"} of ${coverage.segmentsTotal ?? "—"} segments`,
  ];
  for (const line of stats) doc.flowPlain(line, { size: 11, color: [0.14, 0.16, 0.18] });
  doc.gap(8);
  for (const note of report.notes || []) {
    doc.flowPlain(note.text, { size: 9.5, color: [0.32, 0.34, 0.36] });
    doc.gap(2);
  }
  if (report.settings?.excludeQuotes) {
    doc.flowPlain("Quotations and citation markers were excluded before comparison.", { size: 9.5, color: [0.32, 0.34, 0.36] });
  }

  doc.gap(10);
  doc.flowPlain("Sources", { size: 14, bold: true });
  doc.gap(6);
  if (!sources.length) {
    doc.flowPlain("No public page produced a word run long enough to count as a match.", { size: 11 });
  } else {
    const col = [36, 250, 90, 90];
    const headers = ["#", "Source", "Domain", "Matched words"];
    doc.need(18);
    let x = MARGIN_X;
    headers.forEach((header, index) => {
      doc.textAt(header, x, doc.y - 10, { size: 8, bold: true, color: [0.35, 0.37, 0.4] });
      x += col[index];
    });
    doc.y -= 16;
    doc.line([0.9, 0.91, 0.93]);
    for (const source of sources) {
      const color = sourceColor(source.id);
      const title = source.title || source.host || source.url || "Source";
      const titleLines = wrapPlain(title, 9, col[1] - 8, true);
      const urlLines = wrapPlain(source.url || "", 8, col[1] - 8);
      const rowHeight = 14 + titleLines.length * 11 + urlLines.length * 10;
      doc.need(rowHeight);
      doc.rect(MARGIN_X, doc.y - 11, 8, 8, color.rgb);
      doc.textAt(String(source.id), MARGIN_X + 12, doc.y - 10, { size: 10, bold: true });
      let ty = doc.y - 10;
      for (const line of titleLines) {
        doc.textAt(line.map((token) => token.text).join(" "), MARGIN_X + col[0], ty, { size: 9, bold: true });
        ty -= 11;
      }
      for (const line of urlLines) {
        doc.textAt(line.map((token) => token.text).join(" "), MARGIN_X + col[0], ty, { size: 8, color: [0.35, 0.37, 0.4] });
        ty -= 10;
      }
      doc.textAt(source.host || "", MARGIN_X + col[0] + col[1], doc.y - 10, { size: 9 });
      const share = `${Number(source.matchedWords || 0).toLocaleString("en-US")}  ${formatPercent(source.sharePercent || 0)}`;
      doc.textAt(share, MARGIN_X + col[0] + col[1] + col[2], doc.y - 10, { size: 9 });
      doc.y -= Math.max(22, rowHeight - 4);
      doc.pageHasBody = true;
    }
  }

  doc.gap(14);
  doc.flowPlain("Document", { size: 14, bold: true });
  doc.gap(8);
  if (!text) {
    doc.flowPlain(
      report.textStored === false
        ? "The original document was not stored with this saved report, so the highlighted text cannot be redrawn. The overall similarity and sources above are still the recorded figures."
        : "No document text was supplied to this PDF.",
      { size: 11, color: [0.3, 0.32, 0.34] }
    );
  } else {
    const paragraphs = runsToParagraphs(buildHighlightRuns(text, spans));
    for (const paragraph of paragraphs) {
      if (!paragraph.length) {
        doc.gap(6);
        continue;
      }
      doc.flowRuns(paragraph);
    }
  }

  const pageCount = doc.pages.length;
  const objects = [];
  const add = (body) => {
    objects.push(body);
    return objects.length;
  };

  const contentIds = doc.pages.map((ops, index) => {
    const data = headerAndFooter(ops, index + 1, pageCount);
    return add(`<< /Length ${byteLength(data)} >>\nstream\n${data}endstream`);
  });
  const fontRegular = add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>");
  const fontBold = add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>");
  const gsOpaque = add("<< /Type /ExtGState /ca 1 /CA 1 >>");
  const gsSoft = add("<< /Type /ExtGState /ca 0.22 /CA 0.22 >>");
  const pagesId = objects.length + 1 + doc.pages.length;
  const pageIds = [];
  doc.pages.forEach((_, index) => {
    pageIds[index] = add(
      `<< /Type /Page /Parent ${pagesId} 0 R /MediaBox [0 0 ${PAGE_WIDTH} ${PAGE_HEIGHT}] /Resources << /Font << /F1 ${fontRegular} 0 R /F2 ${fontBold} 0 R >> /ExtGState << /GS0 ${gsOpaque} 0 R /GS1 ${gsSoft} 0 R >> >> /Contents ${contentIds[index]} 0 R >>`
    );
  });
  const kids = pageIds.map((id) => `${id} 0 R`).join(" ");
  const storedPagesId = add(`<< /Type /Pages /Count ${pageCount} /Kids [${kids}] >>`);
  if (storedPagesId !== pagesId) throw new Error("PDF page tree id mismatch");
  const catalogId = add(`<< /Type /Catalog /Pages ${pagesId} 0 R >>`);

  let pdf = "%PDF-1.4\n";
  const offsets = [0];
  objects.forEach((body, index) => {
    offsets.push(byteLength(pdf));
    pdf += `${index + 1} 0 obj\n${body}\nendobj\n`;
  });
  const xref = byteLength(pdf);
  pdf += `xref\n0 ${objects.length + 1}\n`;
  pdf += "0000000000 65535 f \n";
  for (let i = 1; i < offsets.length; i++) pdf += `${String(offsets[i]).padStart(10, "0")} 00000 n \n`;
  pdf += `trailer << /Size ${objects.length + 1} /Root ${catalogId} 0 R >>\nstartxref\n${xref}\n%%EOF`;
  const bytes = new Uint8Array(pdf.length);
  for (let index = 0; index < pdf.length; index++) bytes[index] = pdf.charCodeAt(index) & 255;
  return bytes;
}
