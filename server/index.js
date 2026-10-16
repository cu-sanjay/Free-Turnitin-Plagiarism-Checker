import express from "express";
import { registerRoutes } from "./routes.js";
import { setupVite, serveStatic, log } from "./vite.js";

const app = express();
app.use(express.json({ limit: "12mb" }));
app.use(express.urlencoded({ extended: false, limit: "12mb" }));

app.use((req, res, next) => {
  const start = Date.now();
  const path = req.path;
  let capturedJsonResponse = undefined;

  const originalResJson = res.json;
  res.json = function (bodyJson, ...args) {
    capturedJsonResponse = bodyJson;
    return originalResJson.apply(res, [bodyJson, ...args]);
  };

  res.on("finish", () => {
    const duration = Date.now() - start;
    if (path.startsWith("/api")) {
      let logLine = `${req.method} ${path} ${res.statusCode} in ${duration}ms`;
      if (capturedJsonResponse && typeof capturedJsonResponse === "object") {
        const summary = {
          error: capturedJsonResponse.error,
          overallPercent: capturedJsonResponse.similarity?.overallPercent,
          sources: capturedJsonResponse.sources?.length,
        };
        const compact = JSON.stringify(summary);
        if (compact !== "{}") logLine += ` :: ${compact}`;
      }

      if (logLine.length > 180) {
        logLine = logLine.slice(0, 179) + "…";
      }

      log(logLine);
    }
  });

  next();
});

(async () => {
  const server = registerRoutes(app);

  app.use((err, _req, res, _next) => {
    const tooLarge = err?.type === "entity.too.large" || err?.status === 413;
    const status = tooLarge ? 413 : err.status || err.statusCode || 500;
    const message = tooLarge
      ? "This document is larger than the server can accept in one request. Split it into a smaller part."
      : err.message || "Internal Server Error";

    console.error("Error:", err);
    res.status(status).json({ error: message, message });
  });

  if (app.get("env") === "development") {
    await setupVite(app, server);
  } else {
    serveStatic(app);
  }

  const PORT = 5000;
  server.listen(PORT, "0.0.0.0", () => {
    log(`Server running on port ${PORT}`);
  });
})();
