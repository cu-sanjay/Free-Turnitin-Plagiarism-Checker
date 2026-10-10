import { createServer } from "http";
import { checkTextSchema } from "../shared/schema.js";
import { compareAgainstSources } from "./plagiarism.js";

export function registerRoutes(app) {
  app.post("/api/plagiarism-check", async (req, res) => {
    const validation = checkTextSchema.safeParse(req.body);
    if (!validation.success) {
      return res.status(400).json({
        error: validation.error.issues[0]?.message ?? "Invalid comparison request",
      });
    }

    try {
      return res.json(
        compareAgainstSources(
          validation.data.text,
          validation.data.sources,
          validation.data.options
        )
      );
    } catch (error) {
      console.error("Comparison failed:", error);
      return res.status(500).json({ error: "The comparison could not be completed." });
    }
  });

  const httpServer = createServer(app);
  return httpServer;
}
