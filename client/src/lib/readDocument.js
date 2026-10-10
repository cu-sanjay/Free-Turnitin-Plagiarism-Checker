const MAX_FILE_BYTES = 5 * 1024 * 1024;

export async function readDocumentFile(file, maxCharacters) {
  if (file.size > MAX_FILE_BYTES) {
    throw new Error("Documents must be 5 MB or smaller.");
  }

  const extension = file.name.split(".").at(-1)?.toLowerCase();
  let text;

  if (extension === "txt" || extension === "md") {
    text = await file.text();
  } else if (extension === "docx") {
    const mammoth = await import("mammoth/mammoth.browser");
    const result = await mammoth.extractRawText({
      arrayBuffer: await file.arrayBuffer(),
    });
    text = result.value;
  } else if (extension === "pdf") {
    const [{ default: pdfjs }, { default: pdfWorkerUrl }] = await Promise.all([
      import("pdfjs-dist"),
      import("pdfjs-dist/build/pdf.worker.min.mjs?url"),
    ]);
    pdfjs.GlobalWorkerOptions.workerSrc = pdfWorkerUrl;
    const document = await pdfjs.getDocument({
      data: new Uint8Array(await file.arrayBuffer()),
      useSystemFonts: true,
    }).promise;
    try {
      if (document.numPages > 200) {
        throw new Error("PDF files are limited to 200 pages.");
      }
      const pageText = [];
      for (let pageNumber = 1; pageNumber <= document.numPages; pageNumber++) {
        const page = await document.getPage(pageNumber);
        const content = await page.getTextContent();
        pageText.push(
          content.items
            .filter((item) => "str" in item)
            .map((item) => item.str)
            .join(" ")
        );
      }
      text = pageText.join("\n\n");
    } finally {
      await document.destroy();
    }
  } else {
    throw new Error("Choose a .txt, .md, .docx, or text-based .pdf file.");
  }

  if (text.length > maxCharacters) {
    throw new Error(`Extracted text must be ${maxCharacters.toLocaleString()} characters or fewer.`);
  }
  if (!text.trim()) {
    throw new Error("No selectable text was found. Scanned PDFs need OCR and are not supported.");
  }

  return text;
}
