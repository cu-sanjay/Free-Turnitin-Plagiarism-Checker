import { buildPlagiarismPdf } from "@shared/pdf-report.js";

export function downloadReport(report, text = "") {
  const bytes = buildPlagiarismPdf({ ...report, text: text || report.text || "" });
  const blob = new Blob([bytes], { type: "application/pdf" });
  const url = URL.createObjectURL(blob);
  const stamp = (report?.checkedAt || new Date().toISOString()).slice(0, 10);
  const link = document.createElement("a");
  link.href = url;
  link.download = `turnitout-report-${stamp}.pdf`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
