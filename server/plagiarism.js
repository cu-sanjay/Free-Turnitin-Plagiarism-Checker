const PLAGIARISM_THRESHOLD = 0.5;
const SOURCE_THRESHOLD = 0.15;
const DEFAULT_MAX_PASSAGES = 24;
const URLS_PER_PASSAGE = 4;

function yieldToEventLoop() {
  return new Promise((resolve) => setImmediate(resolve));
}

function clampPercent(value) {
  return Math.max(0, Math.min(100, Math.round(value)));
}

function wordList(text) {
  return String(text || "")
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean);
}

function termFrequency(words) {
  const counts = new Map();
  for (const word of words) {
    counts.set(word, (counts.get(word) || 0) + 1);
  }
  return counts;
}

export function calculateSimilarity(text1, text2) {
  const words1 = String(text1 || "").toLowerCase().split(/\s+/).filter(Boolean);
  const words2 = String(text2 || "").toLowerCase().split(/\s+/).filter(Boolean);
  if (!words1.length || !words2.length) return 0;

  const freq1 = termFrequency(words1);
  const freq2 = termFrequency(words2);

  let dot = 0;
  let mag1 = 0;
  let mag2 = 0;

  for (const count of freq1.values()) mag1 += count * count;
  for (const count of freq2.values()) mag2 += count * count;
  for (const [word, count] of freq1) {
    const other = freq2.get(word);
    if (other) dot += count * other;
  }

  if (mag1 === 0 || mag2 === 0) return 0;
  return Math.min(1, dot / (Math.sqrt(mag1) * Math.sqrt(mag2)));
}

export function nGramSimilarity(text1, text2, n = 5) {
  const createNGrams = (text) => {
    const words = String(text || "")
      .toLowerCase()
      .replace(/[^\w\s]/g, "")
      .split(/\s+/)
      .filter(Boolean);
    const ngrams = new Set();
    for (let i = 0; i <= words.length - n; i++) {
      ngrams.add(words.slice(i, i + n).join(" "));
    }
    return ngrams;
  };

  const ngrams1 = createNGrams(text1);
  const ngrams2 = createNGrams(text2);
  if (ngrams1.size === 0 || ngrams2.size === 0) return 0;

  let matches = 0;
  const [smaller, larger] = ngrams1.size <= ngrams2.size ? [ngrams1, ngrams2] : [ngrams2, ngrams1];
  for (const gram of smaller) {
    if (larger.has(gram)) matches++;
  }

  return matches / Math.max(ngrams1.size, ngrams2.size);
}

export function stripQuotedAndCitations(text) {
  return String(text || "")
    .replace(/"[^"\n]{0,800}"/g, " ")
    .replace(/“[^”\n]{0,800}”/g, " ")
    .replace(/\[[0-9,\s-]{1,24}\]/g, " ")
    .replace(/\((?:[^()]{0,160})\)/g, (match) => (/\d{4}|et al\./i.test(match) ? " " : match));
}

export function preparePassages(text, { excludeQuotes = false } = {}) {
  let source = String(text || "");
  if (excludeQuotes) source = stripQuotedAndCitations(source);

  const sentences = source
    .split(/[.!?]+/)
    .map((sentence) => sentence.replace(/\s+/g, " ").trim())
    .filter((sentence) => sentence.length > 20);

  const passages = [];
  for (const sentence of sentences) {
    const words = sentence.split(/\s+/).filter(Boolean);
    if (words.length <= 80) {
      passages.push(sentence);
      continue;
    }
    for (let i = 0; i < words.length; i += 40) {
      const slice = words.slice(i, i + 40).join(" ");
      if (slice.length > 20) passages.push(slice);
    }
  }
  return passages;
}

export function defaultMaxPassages(count) {
  if (count <= 20) return count;
  return DEFAULT_MAX_PASSAGES;
}

export function selectPassages(passages, maxPassages = DEFAULT_MAX_PASSAGES) {
  const list = Array.isArray(passages) ? passages : [];
  const max = Math.max(0, maxPassages);
  if (!list.length || max === 0) return [];
  if (list.length <= max) {
    return list.map((text, index) => ({ text, index }));
  }

  const chosen = [];
  const seen = new Set();
  const slots = Math.max(1, max);
  for (let i = 0; i < slots; i++) {
    const index = Math.round((i * (list.length - 1)) / Math.max(1, slots - 1));
    if (seen.has(index)) continue;
    seen.add(index);
    chosen.push({ text: list[index], index });
  }
  return chosen;
}

export function extractExcerpt(passage, content) {
  const passageWords = wordList(passage).slice(0, 24);
  const contentWords = String(content || "").split(/\s+/).filter(Boolean);
  if (!passageWords.length || contentWords.length < 6) {
    return String(content || "").replace(/\s+/g, " ").trim().slice(0, 280);
  }

  const contentLower = contentWords.map((word) => word.toLowerCase());
  const step = Math.max(1, Math.floor(contentLower.length / 500));
  const windowSize = passageWords.length;
  let bestStart = 0;
  let bestScore = 0;

  for (let i = 0; i <= contentLower.length - Math.min(6, windowSize); i += step) {
    let score = 0;
    const limit = Math.min(windowSize, contentLower.length - i);
    for (let j = 0; j < limit; j++) {
      if (contentLower[i + j] === passageWords[j]) score++;
    }
    if (score > bestScore) {
      bestScore = score;
      bestStart = i;
    }
  }

  const start = Math.max(0, bestStart - 6);
  const end = Math.min(contentWords.length, bestStart + windowSize + 10);
  const excerpt = contentWords.slice(start, end).join(" ");
  return excerpt.length > 360 ? `${excerpt.slice(0, 357)}...` : excerpt;
}

function hostnameOf(url) {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
}

function summarize(plagiarismPercentage, sourceCount, checked, total) {
  const scope =
    checked < total
      ? `Scores cover ${checked} of ${total} passages, spaced through the document.`
      : `Scores cover every passage long enough to compare.`;
  const limit =
    "This is a public-web similarity check. It is not a Turnitin report and it does not search private research or IEEE databases.";

  if (!checked) {
    return `Not enough prose to compare. Add at least one passage longer than 20 characters. ${limit}`;
  }
  if (plagiarismPercentage === 0 && sourceCount === 0) {
    return `${scope} No compared passage cleared the match threshold on the pages this check could fetch. ${limit}`;
  }
  if (plagiarismPercentage < 20) {
    return `${scope} ${sourceCount} source${sourceCount === 1 ? "" : "s"} overlapped with the compared passages. ${limit}`;
  }
  if (plagiarismPercentage < 50) {
    return `${scope} A noticeable share of compared passages overlapped public pages (${sourceCount} source${sourceCount === 1 ? "" : "s"}). ${limit}`;
  }
  return `${scope} Many compared passages closely matched public pages (${sourceCount} source${sourceCount === 1 ? "" : "s"}). ${limit}`;
}

export async function searchWeb(query) {
  const urls = [];

  try {
    const searchQuery = encodeURIComponent(String(query || "").slice(0, 200));
    const ddgUrl = `https://html.duckduckgo.com/html/?q=${searchQuery}`;
    const response = await fetch(ddgUrl, {
      headers: {
        "User-Agent":
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
      },
    });
    const html = await response.text();
    const resultMatches = html.match(/uddg=([^"&]+)/g) || [];
    const ddgUrls = resultMatches
      .map((match) => {
        const encoded = match.replace("uddg=", "");
        try {
          return decodeURIComponent(encoded);
        } catch {
          return null;
        }
      })
      .filter((url) => url && url.startsWith("http") && !url.includes("duckduckgo.com"))
      .slice(0, 8);
    urls.push(...ddgUrls.filter(Boolean));
  } catch (error) {
    console.error("DuckDuckGo search error:", error.message || error);
  }

  try {
    const searchQuery = encodeURIComponent(String(query || "").slice(0, 150));
    const crossrefUrl = `https://api.crossref.org/works?query=${searchQuery}&rows=5`;
    const response = await fetch(crossrefUrl, {
      headers: { "User-Agent": "plagiarism-checker/0.0.4 (academic similarity check)" },
    });
    const data = await response.json();
    if (data.message?.items) {
      for (const item of data.message.items) {
        if (item.URL) urls.push(item.URL);
      }
    }
  } catch (error) {
    console.error("CrossRef search error:", error.message || error);
  }

  return [...new Set(urls)].slice(0, 10);
}

export async function fetchPageContent(url) {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 8000);
    const response = await fetch(url, {
      headers: {
        "User-Agent":
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        Accept: "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
      },
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (!response.ok) return "";

    const html = await response.text();
    const text = html
      .replace(/<script[^>]*>.*?<\/script>/gis, "")
      .replace(/<style[^>]*>.*?<\/style>/gis, "")
      .replace(/<nav[^>]*>.*?<\/nav>/gis, "")
      .replace(/<header[^>]*>.*?<\/header>/gis, "")
      .replace(/<footer[^>]*>.*?<\/footer>/gis, "")
      .replace(/<aside[^>]*>.*?<\/aside>/gis, "")
      .replace(/<[^>]+>/g, " ")
      .replace(/&[a-z]+;/gi, " ")
      .replace(/\s+/g, " ")
      .trim();

    return text.slice(0, 5000);
  } catch (error) {
    console.error(`Error fetching ${url}:`, error.message || error);
    return "";
  }
}

function groupSources(results) {
  const grouped = new Map();
  for (const item of results) {
    for (const source of item.sources || []) {
      const current = grouped.get(source.url) || {
        url: source.url,
        host: hostnameOf(source.url),
        matchCount: 0,
        maxSimilarity: 0,
        similarityTotal: 0,
      };
      current.matchCount += 1;
      current.similarityTotal += source.similarity;
      current.maxSimilarity = Math.max(current.maxSimilarity, source.similarity);
      grouped.set(source.url, current);
    }
  }
  return [...grouped.values()]
    .map((source) => ({
      url: source.url,
      host: source.host,
      matchCount: source.matchCount,
      maxSimilarity: source.maxSimilarity,
      avgSimilarity: Math.round(source.similarityTotal / source.matchCount),
    }))
    .sort((a, b) => b.maxSimilarity - a.maxSimilarity || b.matchCount - a.matchCount);
}

export async function checkPlagiarism(text, options = {}) {
  const {
    excludeQuotes = false,
    onProgress = () => {},
    signal,
    search = searchWeb,
    fetchContent = fetchPageContent,
    maxPassages,
  } = options;

  const report = (phase, percent, message, extra = {}) => {
    onProgress({
      phase,
      percent: clampPercent(percent),
      message,
      ...extra,
    });
  };

  const throwIfAborted = () => {
    if (signal?.aborted) {
      const error = new Error("Check cancelled");
      error.name = "AbortError";
      throw error;
    }
  };

  report("prepare", 2, "Reading the document");
  await yieldToEventLoop();
  throwIfAborted();

  const passages = preparePassages(text, { excludeQuotes });
  const limit = maxPassages ?? defaultMaxPassages(passages.length);
  const selected = selectPassages(passages, limit);
  const wordCount = String(text || "")
    .trim()
    .split(/\s+/)
    .filter(Boolean).length;

  report("prepare", 8, `Prepared ${passages.length} passages`, {
    completed: 0,
    total: selected.length,
  });

  const results = [];

  for (let i = 0; i < selected.length; i++) {
    throwIfAborted();
    const passage = selected[i];
    const percent = 10 + Math.round((i / selected.length) * 80);
    report("compare", percent, `Comparing passage ${i + 1} of ${selected.length}`, {
      completed: i,
      total: selected.length,
    });

    const urls = (await search(passage.text)).slice(0, URLS_PER_PASSAGE);
    let maxSimilarity = 0;
    const matchedSources = [];

    for (const url of urls) {
      throwIfAborted();
      const content = await fetchContent(url);
      if (!content || content.length <= 100) continue;

      const cosineSim = calculateSimilarity(passage.text, content);
      const ngramSim = nGramSimilarity(passage.text, content, 5);
      const similarity = Math.max(cosineSim, ngramSim);
      if (similarity > maxSimilarity) maxSimilarity = similarity;

      if (similarity > SOURCE_THRESHOLD) {
        matchedSources.push({
          url,
          host: hostnameOf(url),
          similarity: Math.round(similarity * 100),
          excerpt: extractExcerpt(passage.text, content),
        });
      }
      await yieldToEventLoop();
    }

    matchedSources.sort((a, b) => b.similarity - a.similarity);
    results.push({
      sentence: passage.text,
      passageIndex: passage.index,
      similarity: Math.round(maxSimilarity * 100),
      sources: matchedSources,
      isPlagiarized: maxSimilarity > PLAGIARISM_THRESHOLD,
    });
  }

  report("summarize", 94, "Summarizing matches", {
    completed: selected.length,
    total: selected.length,
  });
  await yieldToEventLoop();

  const totalSimilarity = results.reduce((sum, item) => sum + item.similarity, 0);
  const overallScore = results.length ? Math.round(totalSimilarity / results.length) : 0;
  const plagiarizedCount = results.filter((item) => item.isPlagiarized).length;
  const plagiarismPercentage = results.length ? Math.round((plagiarizedCount / results.length) * 100) : 0;
  const sources = groupSources(results);
  const coverageNote =
    selected.length < passages.length
      ? `Compared ${selected.length} of ${passages.length} passages, spaced from the start, middle, and end. Uncompared passages are not included in the score.`
      : passages.length
        ? `Compared all ${passages.length} passages that were long enough.`
        : "No comparable passages were found.";

  const checkResult = {
    overallScore,
    plagiarismPercentage,
    totalSentences: results.length,
    plagiarizedSentences: plagiarizedCount,
    results,
    totalWords: wordCount,
    characterCount: String(text || "").length,
    passagesInDocument: passages.length,
    passagesChecked: results.length,
    sources,
    coverageNote,
    analysisSummary: summarize(plagiarismPercentage, sources.length, results.length, passages.length),
    excludeQuotes: Boolean(excludeQuotes),
    checkedAt: new Date().toISOString(),
  };

  report("done", 100, "Check complete", {
    completed: selected.length,
    total: selected.length,
  });

  return checkResult;
}
