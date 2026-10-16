import { createServer } from "http";
import { ZodError } from "zod";
import { checkTextSchema } from "../shared/schema.js";
import { checkPlagiarism } from "./plagiarism.js";

const MAX_TEXT_CHARS = 8_000_000;

function readInput(body) {
  const { text } = checkTextSchema.parse(body ?? {});
  if (text.length > MAX_TEXT_CHARS) {
    const error = new Error(
      "This document is larger than this server can compare in one request. Split it into parts under 8 million characters."
    );
    error.status = 413;
    throw error;
  }
  return {
    text,
    excludeQuotes: Boolean(body?.excludeQuotes),
  };
}

function errorPayload(error) {
  if (error instanceof ZodError) {
    return {
      status: 400,
      body: { error: error.errors[0]?.message || "Validation failed" },
    };
  }
  return {
    status: error.status || 500,
    body: { error: error instanceof Error ? error.message : "An unknown error occurred" },
  };
}

function writeEvent(res, payload) {
  res.write(`data: ${JSON.stringify(payload)}\n\n`);
}

export function registerRoutes(app) {
  app.post("/api/plagiarism-check", async (req, res) => {
    try {
      const { text, excludeQuotes } = readInput(req.body);
      console.log("Starting plagiarism check for text length:", text.length);
      const checkResult = await checkPlagiarism(text, { excludeQuotes });
      console.log(
        "Plagiarism check complete. Overall:",
        checkResult.similarity?.overallPercent,
        "sources:",
        checkResult.sources?.length,
        "segments:",
        checkResult.coverage?.segmentsSearched,
        "/",
        checkResult.coverage?.segmentsTotal
      );
      res.json(checkResult);
    } catch (error) {
      const payload = errorPayload(error);
      console.error("Error in plagiarism check:", error);
      res.status(payload.status).json(payload.body);
    }
  });

  app.post("/api/plagiarism-check/stream", async (req, res) => {
    res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
    res.setHeader("Cache-Control", "no-cache, no-transform");
    res.setHeader("Connection", "keep-alive");
    res.setHeader("X-Accel-Buffering", "no");
    if (typeof res.flushHeaders === "function") res.flushHeaders();

    const abortController = new AbortController();
    const onClose = () => {
      if (!res.writableEnded) abortController.abort();
    };
    res.on("close", onClose);

    try {
      const { text, excludeQuotes } = readInput(req.body);
      console.log("Starting streamed plagiarism check for text length:", text.length);
      const checkResult = await checkPlagiarism(text, {
        excludeQuotes,
        signal: abortController.signal,
        onProgress: (progress) => {
          if (!res.writableEnded) writeEvent(res, { type: "progress", ...progress });
        },
      });
      if (!res.writableEnded) writeEvent(res, { type: "result", result: checkResult });
    } catch (error) {
      if (error?.name === "AbortError") {
        if (!res.writableEnded) writeEvent(res, { type: "error", message: "Check cancelled" });
      } else {
        const payload = errorPayload(error);
        console.error("Error in streamed plagiarism check:", error);
        if (!res.writableEnded) {
          writeEvent(res, { type: "error", status: payload.status, message: payload.body.error });
        }
      }
    } finally {
      res.off("close", onClose);
      if (!res.writableEnded) res.end();
    }
  });

  return createServer(app);
}
