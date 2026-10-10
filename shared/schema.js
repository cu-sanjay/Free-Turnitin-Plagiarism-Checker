import { z } from "zod";

export const checkTextSchema = z.object({
  text: z
    .string()
    .min(100, "Your draft must be at least 100 characters long")
    .max(50000, "Your draft must be 50,000 characters or fewer")
    .refine((value) => value.trim().length >= 100, "Your draft must be at least 100 characters long"),
  sources: z
    .array(
      z.object({
        id: z.string().min(1).max(80),
        name: z.string().trim().min(1).max(80, "Source names must be 80 characters or fewer"),
        exclude: z.boolean().default(false),
        url: z
          .string()
          .max(2048)
          .refine((value) => {
            if (!value) return true;
            try {
              return ["http:", "https:"].includes(new URL(value).protocol);
            } catch {
              return false;
            }
          }, "Source links must use http or https")
          .optional()
          .default(""),
        text: z
          .string()
          .min(1, "Each reference needs text")
          .max(100000, "Each reference must be 100,000 characters or fewer")
          .refine((value) => value.trim().length > 0, "Each reference needs text"),
      })
    )
    .min(1, "Add at least one reference document")
    .max(25, "You can compare against up to 25 reference documents"),
  options: z
    .object({
      excludeQuotes: z.boolean().default(false),
      excludeReferences: z.boolean().default(false),
      minMatchWords: z.union([z.literal(5), z.literal(8), z.literal(12)]).default(5),
    })
    .optional()
    .default({ excludeQuotes: false, excludeReferences: false, minMatchWords: 5 }),
});
