# ClearMark

ClearMark is a lightweight text-similarity checker for reviewing a draft against reference documents you choose. It is an independent project and is not affiliated with Turnitin or any other academic-integrity service.

## What it does

- Compares a draft with up to 25 reference documents that you paste or upload.
- Finds exact, case-insensitive runs of five or more words, with punctuation and Unicode-aware tokenization.
- Shows matching passages side by side, highlights them in the draft, and summarizes matched-word coverage.
- Optionally excludes quoted passages and a detected bibliography from the score.
- Lets you choose a 5-, 8-, or 12-word exact-match threshold to tune the balance between shorter overlaps and longer, more distinctive phrases.
- Records reference titles and optional source links, and exports JSON or a print-ready report that can be saved as PDF.
- Imports `.txt`, `.md`, `.docx`, and text-based `.pdf` files. Scanned PDFs need OCR and are not supported.

Turnitin's public product pages describe reports over its institutional collection of student submissions, web pages, and licensed publications, with match filters and side-by-side evidence. ClearMark follows useful review patterns such as source-by-source evidence, exclusions, and downloadable reports, but is independent and does not have access to Turnitin's private repositories, the web, or subscription databases. It compares only text you provide, does not fetch URLs (links are attribution only), and does not determine whether a match constitutes plagiarism. Common phrases and properly cited writing may match; treat results as a review aid, not a verdict.

### Feature cross-check

| Review need | ClearMark |
| --- | --- |
| Similarity overview | Percentage of analyzed draft words in exact matching phrases |
| Source details | Per-reference score and matched text side by side; optional source links |
| Visual review | Matching passages highlighted in the draft |
| Exclusions | Optionally skip quoted passages, bibliography text, or selected sources |
| Match filtering | Choose a minimum exact phrase length of 5, 8, or 12 words |
| File support | `.txt`, `.md`, `.docx`, and text-based `.pdf` (no OCR) |
| Export | JSON data or print-to-PDF through the desktop app |
| Larger source sets | Compare with up to 25 references |
| Turnitin / open-web corpus search | Not provided; only references supplied by the user are checked |

The review patterns are informed by public product descriptions from [Turnitin Similarity](https://www.turnitin.com/products/similarity), [Turnitin Feedback Studio](https://www.turnitin.com/products/feedback-studio), [Grammarly's plagiarism checker](https://www.grammarly.com/plagiarism-checker), [Quetext](https://www.quetext.com/), and [Copyscape](https://www.copyscape.com/). These services describe capabilities such as originality reports, contextual source review, and comparisons against online or licensed source collections. ClearMark is not a replacement for those corpora, institutional workflows, writing feedback, or AI-writing indicators; its results remain limited to reference text supplied by the user.

## Run locally

Requirements: Node.js 18 or newer.

```sh
npm install
npm run dev
```

Open `http://localhost:5000`. For a production build, run `npm run build`, then run `npm start`.

## Windows desktop app

Install dependencies and build the Windows installer:

```sh
npm install
npm run desktop:dist
```

The installer is written to `release/`. It installs ClearMark as a standalone Windows desktop application with Start Menu and desktop shortcuts. For local desktop development, use `npm run desktop`.

## Limits and privacy

A draft can contain up to 50,000 characters; each of up to 25 references can contain up to 100,000 characters. Uploaded files are read in the browser as plain text and sent to the app's comparison endpoint. The comparison has no database persistence and does not send document text to a search provider. As with any locally hosted web app, the operator of the server can access requests sent to that server.

## Development

Run the algorithm and input-validation tests with:

```sh
npm test
```

The app is built with React, Vite, Express, and Zod.
