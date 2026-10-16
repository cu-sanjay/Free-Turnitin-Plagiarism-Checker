const PAGE_WIDTH = 595.28;
const PAGE_HEIGHT = 841.89;
const MARGIN_X = 48;
const TOP = 786;
const BOTTOM = 56;
const CONTENT_WIDTH = PAGE_WIDTH - MARGIN_X * 2;

function pdfEscape(value) {
  return String(value)
    .replace(/\s+/g, " ")
    .split("")
    .map((char) => {
      const code = char.charCodeAt(0);
      if (char === "\\") return "\\\\";
      if (char === "(") return "\\(";
      if (char === ")") return "\\)";
      if (code < 32 || code > 255) return "?";
      if (code > 126) return `\\${code.toString(8).padStart(3, "0")}`;
      return char;
    })
    .join("");
}

function wrapText(text, size, width = CONTENT_WIDTH) {
  const maxChars = Math.max(12, Math.floor(width / (size * 0.5)));
  const words = String(text || "").replace(/\s+/g, " ").trim().split(" ");
  if (!words[0]) return [""];
  const lines = [];
  let current = "";
  for (const word of words) {
    const pieces = word.length > maxChars ? word.match(new RegExp(`.{1,${maxChars}}`, "g")) : [word];
    for (const piece of pieces) {
      const next = current ? `${current} ${piece}` : piece;
      if (next.length > maxChars && current) {
        lines.push(current);
        current = piece;
      } else {
        current = next;
      }
    }
  }
  if (current) lines.push(current);
  return lines;
}

function formatWhen(value) {
  const date = value ? new Date(value) : new Date();
  if (Number.isNaN(date.getTime())) return "Unknown time";
  return date.toISOString().replace("T", " ").replace(/\.\d{3}Z$/, " UTC");
}

function buildLines(report) {
  const lines = [];
  const add = (text, { size = 11, color = [0.1, 0.1, 0.1], gap = 16, font = "F1" } = {}) => {
    for (const line of wrapText(text, size)) {
      lines.push({ text: line, size, color, gap, font });
    }
  };

  add("Similarity report", { size: 22, font: "F2", gap: 28, color: [0, 0, 0] });
  add("Public-web overlap only. Not a Turnitin report.", { size: 10, color: [0.3, 0.3, 0.3], gap: 18 });
  add(`Generated ${formatWhen(report.checkedAt)}`, { size: 10, color: [0.3, 0.3, 0.3], gap: 22 });

  add("Score summary", { size: 14, font: "F2", gap: 20 });
  add(`Passages over 50% similarity: ${report.plagiarismPercentage ?? 0}%`, { size: 12, gap: 16 });
  add(`Average similarity of compared passages: ${report.overallScore ?? 0}%`, { size: 12, gap: 16 });
  add(`Words in submission: ${report.totalWords ?? 0}`, { size: 11, gap: 15 });
  add(`Characters in submission: ${report.characterCount ?? 0}`, { size: 11, gap: 15 });
  add(`Passages compared: ${report.passagesChecked ?? report.totalSentences ?? 0}`, { size: 11, gap: 15 });
  add(`Passages in document: ${report.passagesInDocument ?? report.totalSentences ?? 0}`, { size: 11, gap: 15 });
  add(`Passages flagged above 50%: ${report.plagiarizedSentences ?? 0}`, { size: 11, gap: 18 });
  if (report.excludeQuotes) add("Quotations and citation markers were excluded before comparison.", { size: 10, gap: 16 });
  add(report.coverageNote || "", { size: 10, color: [0.25, 0.25, 0.25], gap: 16 });
  add(report.analysisSummary || "", { size: 10, color: [0.25, 0.25, 0.25], gap: 24 });

  const sources = Array.isArray(report.sources) ? report.sources : [];
  add("Sources", { size: 14, font: "F2", gap: 20 });
  if (!sources.length) {
    add("No public page passed the similarity threshold for the compared passages.", { size: 11, gap: 20 });
  } else {
    sources.forEach((source, index) => {
      add(`${index + 1}. ${source.host || source.url}`, { size: 11, font: "F2", gap: 14 });
      add(source.url, { size: 9, color: [0.2, 0.2, 0.2], gap: 13 });
      add(
        `Highest similarity ${source.maxSimilarity ?? source.similarity ?? 0}% · ${source.matchCount || 1} passage match${(source.matchCount || 1) === 1 ? "" : "es"}`,
        { size: 10, gap: 16 }
      );
    });
  }

  add("Compared passages", { size: 14, font: "F2", gap: 20 });
  const results = Array.isArray(report.results) ? report.results : [];
  if (!results.length) {
    add("No passage was long enough to compare.", { size: 11, gap: 16 });
  }
  results.forEach((item, index) => {
    add(`Passage ${index + 1} · ${item.similarity ?? 0}% similarity`, { size: 12, font: "F2", gap: 16 });
    add(item.sentence || "", { size: 10, gap: 14 });
    if (!item.sources?.length) {
      add("No source passed the 15% similarity threshold.", { size: 10, color: [0.3, 0.3, 0.3], gap: 18 });
      return;
    }
    for (const source of item.sources) {
      add(`${source.similarity}% · ${source.url}`, { size: 9, color: [0.15, 0.15, 0.15], gap: 12 });
      if (source.excerpt) add(`Excerpt: ${source.excerpt}`, { size: 9, color: [0.28, 0.28, 0.28], gap: 14 });
    }
    lines.push({ text: "", size: 8, color: [0, 0, 0], gap: 8, font: "F1" });
  });

  return lines;
}

function paginate(lines) {
  const pages = [];
  let current = [];
  let y = TOP;
  const commit = () => {
    pages.push(current);
    current = [];
    y = TOP;
  };

  for (const line of lines) {
    if (y - line.gap < BOTTOM) commit();
    current.push({ ...line, y });
    y -= line.gap;
  }
  if (current.length || !pages.length) pages.push(current);
  return pages;
}

function byteLength(value) {
  return new TextEncoder().encode(value).length;
}

function pageStream(pageLines, pageNumber, pageCount) {
  const commands = [
    "0.08 0.08 0.08 rg",
    `${MARGIN_X} 812 m ${PAGE_WIDTH - MARGIN_X} 812 l S`,
    "BT",
    "/F2 9 Tf",
    "0 0 0 rg",
    `1 0 0 1 ${MARGIN_X} 820 Tm`,
    "(Turnitout similarity report) Tj",
    "ET",
    "0.35 0.35 0.35 rg",
    "BT",
    "/F1 9 Tf",
    `1 0 0 1 ${PAGE_WIDTH - MARGIN_X - 70} 820 Tm`,
    `(${pdfEscape(`${pageNumber} / ${pageCount}`)}) Tj`,
    "ET",
    "0.75 0.75 0.75 RG",
    `${MARGIN_X} 40 m ${PAGE_WIDTH - MARGIN_X} 40 l S`,
    "BT",
    "/F1 8 Tf",
    "0.35 0.35 0.35 rg",
    `1 0 0 1 ${MARGIN_X} 28 Tm`,
    "(Internet similarity only. Scores describe compared passages. Not a Turnitin or publisher-database report.) Tj",
    "ET",
  ];

  for (const line of pageLines) {
    const [r, g, b] = line.color;
    commands.push("BT");
    commands.push(`/${line.font} ${line.size} Tf`);
    commands.push(`${r} ${g} ${b} rg`);
    commands.push(`1 0 0 1 ${MARGIN_X} ${line.y.toFixed(2)} Tm`);
    commands.push(`(${pdfEscape(line.text)}) Tj`);
    commands.push("ET");
  }

  return commands.join("\n");
}

export function buildPlagiarismPdf(report) {
  const pages = paginate(buildLines(report || {}));
  const objects = [];
  const add = (body) => {
    objects.push(body);
    return objects.length;
  };

  const pageIds = pages.map(() => null);
  const contentIds = pages.map((pageLines, index) => {
    const data = `${pageStream(pageLines, index + 1, pages.length)}\n`;
    return add(`<< /Length ${byteLength(data)} >>\nstream\n${data}endstream`);
  });

  const fontRegular = add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>");
  const fontBold = add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>");
  const pagesId = objects.length + 1 + pages.length;
  pages.forEach((_, index) => {
    pageIds[index] = add(
      `<< /Type /Page /Parent ${pagesId} 0 R /MediaBox [0 0 ${PAGE_WIDTH} ${PAGE_HEIGHT}] /Resources << /Font << /F1 ${fontRegular} 0 R /F2 ${fontBold} 0 R >> >> /Contents ${contentIds[index]} 0 R >>`
    );
  });
  const kids = pageIds.map((id) => `${id} 0 R`).join(" ");
  const storedPagesId = add(`<< /Type /Pages /Count ${pages.length} /Kids [${kids}] >>`);
  if (storedPagesId !== pagesId) {
    throw new Error("PDF page tree id mismatch");
  }
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
  for (let i = 1; i < offsets.length; i++) {
    pdf += `${String(offsets[i]).padStart(10, "0")} 00000 n \n`;
  }
  pdf += `trailer << /Size ${objects.length + 1} /Root ${catalogId} 0 R >>\nstartxref\n${xref}\n%%EOF`;
  const bytes = new Uint8Array(pdf.length);
  for (let index = 0; index < pdf.length; index++) bytes[index] = pdf.charCodeAt(index) & 255;
  return bytes;
}
