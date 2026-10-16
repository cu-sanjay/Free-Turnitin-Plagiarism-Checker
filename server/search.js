import dns from "node:dns/promises";
import net from "node:net";

const BROWSER_UA =
  "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36";
const BOT_UA = "TurnitoutSimilarityCheck/1.0 (+https://github.com/S4M3Y/Free-Turnitin-Plagiarism-Checker)";

const SKIP_HOSTS = [
  "youtube.com",
  "youtu.be",
  "facebook.com",
  "instagram.com",
  "tiktok.com",
  "pinterest.com",
  "twitter.com",
  "x.com",
  "linkedin.com",
  "reddit.com",
  "translate.google.com",
];

const MAX_PAGE_BYTES = 4_000_000;
const MAX_TEXT_CHARS = 3_000_000;

export function sleep(ms, signal) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(abortError());
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener(
      "abort",
      () => {
        clearTimeout(timer);
        reject(abortError());
      },
      { once: true }
    );
  });
}

export function abortError() {
  const error = new Error("Check cancelled");
  error.name = "AbortError";
  return error;
}

function withTimeout(signal, ms) {
  const timeout = AbortSignal.timeout(ms);
  return signal ? AbortSignal.any([signal, timeout]) : timeout;
}

export function hostnameOf(url) {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return String(url);
  }
}

export function isPrivateAddress(address) {
  if (net.isIPv4(address)) {
    const [a, b] = address.split(".").map(Number);
    return (
      a === 10 ||
      a === 127 ||
      a === 0 ||
      (a === 169 && b === 254) ||
      (a === 172 && b >= 16 && b <= 31) ||
      (a === 192 && b === 168) ||
      (a === 100 && b >= 64 && b <= 127)
    );
  }
  if (net.isIPv6(address)) {
    const value = address.toLowerCase();
    return value === "::1" || value === "::" || value.startsWith("fc") || value.startsWith("fd") || value.startsWith("fe80") || value.startsWith("::ffff:");
  }
  return true;
}

async function assertPublicUrl(url) {
  const parsed = new URL(url);
  if (!/^https?:$/.test(parsed.protocol)) throw new Error("Only http and https pages can be fetched");
  const host = parsed.hostname;
  if (host === "localhost" || host.endsWith(".local") || host.endsWith(".internal")) {
    throw new Error("Private addresses are not fetched");
  }
  if (net.isIP(host)) {
    if (isPrivateAddress(host)) throw new Error("Private addresses are not fetched");
    return parsed;
  }
  const records = await dns.lookup(host, { all: true });
  if (!records.length || records.some((record) => isPrivateAddress(record.address))) {
    throw new Error("Private addresses are not fetched");
  }
  return parsed;
}

export function shouldFetchUrl(url) {
  try {
    const parsed = new URL(url);
    if (!/^https?:$/.test(parsed.protocol)) return false;
    const host = parsed.hostname.replace(/^www\./, "");
    if (SKIP_HOSTS.some((blocked) => host === blocked || host.endsWith(`.${blocked}`))) return false;
    if (/\.(pdf|docx?|pptx?|xlsx?|zip|png|jpe?g|gif|mp4|mp3)(\?|#|$)/i.test(parsed.pathname)) return false;
    return true;
  } catch {
    return false;
  }
}

// HTML to text ------------------------------------------------------------------------------

const NAMED_ENTITIES = {
  amp: "&",
  lt: "<",
  gt: ">",
  quot: '"',
  apos: "'",
  nbsp: " ",
  ndash: "–",
  mdash: "—",
  hellip: "…",
  lsquo: "‘",
  rsquo: "’",
  ldquo: "“",
  rdquo: "”",
  copy: "©",
  reg: "®",
  trade: "™",
  eacute: "é",
  egrave: "è",
  agrave: "à",
  aacute: "á",
  uuml: "ü",
  ouml: "ö",
  auml: "ä",
  ccedil: "ç",
  ntilde: "ñ",
  bull: "•",
  middot: "·",
  deg: "°",
  times: "×",
  minus: "−",
};

function decodeEntities(text) {
  return text.replace(/&(#x[0-9a-f]+|#\d+|[a-z][a-z0-9]+);/gi, (whole, body) => {
    if (body[0] === "#") {
      const code = body[1].toLowerCase() === "x" ? parseInt(body.slice(2), 16) : parseInt(body.slice(1), 10);
      try {
        return Number.isFinite(code) && code > 0 && code < 0x110000 ? String.fromCodePoint(code) : " ";
      } catch {
        return " ";
      }
    }
    return NAMED_ENTITIES[body.toLowerCase()] ?? " ";
  });
}

export function htmlToText(html) {
  const source = String(html || "");
  const titleMatch = source.match(/<title[^>]*>([\s\S]{0,300}?)<\/title>/i);
  const title = titleMatch ? decodeEntities(titleMatch[1]).replace(/\s+/g, " ").trim() : "";
  const text = source
    .replace(/<!--[\s\S]*?-->/g, " ")
    .replace(/<(script|style|noscript|svg|template|nav|footer|aside|form|iframe|select|button)\b[\s\S]*?<\/\1>/gi, " ")
    .replace(/<sup\b[^>]*class="[^"]*\breference\b[^"]*"[\s\S]*?<\/sup>/gi, " ")
    .replace(/<span\b[^>]*class="[^"]*mw-editsection[^"]*"[\s\S]*?<\/span>\s*<\/span>|<span\b[^>]*class="[^"]*mw-editsection[^"]*"[\s\S]*?<\/span>/gi, " ")
    .replace(/<\/?(?:p|div|br|li|ul|ol|h[1-6]|tr|td|th|section|article|header|blockquote|pre|table|figure|figcaption|dd|dt)\b[^>]*>/gi, "\n")
    .replace(/<[^>]+>/g, " ");
  const decoded = decodeEntities(text)
    .replace(/[ \t\f\v\u00a0]+/g, " ")
    .replace(/ *\n[ \n]*/g, "\n")
    .trim();
  return { title, text: decoded.slice(0, MAX_TEXT_CHARS) };
}

async function readBody(response, limit) {
  const reader = response.body?.getReader();
  if (!reader) return { text: await response.text(), truncated: false };
  const chunks = [];
  let total = 0;
  let truncated = false;
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    chunks.push(value);
    total += value.length;
    if (total >= limit) {
      truncated = true;
      await reader.cancel().catch(() => {});
      break;
    }
  }
  const buffer = Buffer.concat(chunks.map((chunk) => Buffer.from(chunk)));
  return { text: buffer.toString("utf8"), truncated };
}

/**
 * Fetch a page and extract its readable text.
 * Resolves to { ok: true, text, title } or { ok: false, reason }; it never invents content.
 */
export async function fetchPageContent(url, { signal, timeoutMs = 12_000 } = {}) {
  try {
    let current = url;
    for (let hop = 0; hop < 5; hop++) {
      await assertPublicUrl(current);
      const response = await fetch(current, {
        redirect: "manual",
        signal: withTimeout(signal, timeoutMs),
        headers: {
          "User-Agent": current.includes("wikipedia.org") ? BOT_UA : BROWSER_UA,
          Accept: "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5",
          "Accept-Language": "en-US,en;q=0.8",
        },
      });
      if (response.status >= 300 && response.status < 400 && response.headers.get("location")) {
        current = new URL(response.headers.get("location"), current).toString();
        await response.body?.cancel().catch(() => {});
        continue;
      }
      if (!response.ok) {
        await response.body?.cancel().catch(() => {});
        return { ok: false, reason: `HTTP ${response.status}` };
      }
      const type = (response.headers.get("content-type") || "").toLowerCase();
      if (type && !/(text\/|application\/xhtml|application\/xml)/.test(type)) {
        await response.body?.cancel().catch(() => {});
        return { ok: false, reason: `Unsupported content type (${type.split(";")[0]})` };
      }
      const { text: body, truncated } = await readBody(response, MAX_PAGE_BYTES);
      const extracted = type.includes("text/plain") ? { title: "", text: body.slice(0, MAX_TEXT_CHARS) } : htmlToText(body);
      if (extracted.text.length < 200) return { ok: false, reason: "No readable text on the page" };
      return { ok: true, text: extracted.text, title: extracted.title, truncated, finalUrl: current };
    }
    return { ok: false, reason: "Too many redirects" };
  } catch (error) {
    if (signal?.aborted) throw abortError();
    const reason = error?.name === "TimeoutError" ? "Timed out" : error?.cause?.code || error?.message || "Fetch failed";
    return { ok: false, reason: String(reason).slice(0, 120) };
  }
}

// Search engines ----------------------------------------------------------------------------

function decodeBingRedirect(href) {
  const url = href.replace(/&amp;/g, "&");
  try {
    const parsed = new URL(url);
    if (parsed.hostname.endsWith("bing.com") && parsed.pathname.startsWith("/ck/a")) {
      const encoded = parsed.searchParams.get("u");
      if (encoded?.startsWith("a1")) {
        const decoded = Buffer.from(encoded.slice(2).replace(/-/g, "+").replace(/_/g, "/"), "base64").toString("utf8");
        if (/^https?:\/\//.test(decoded)) return decoded;
      }
      return null;
    }
    return url;
  } catch {
    return null;
  }
}

export function parseBingResults(html) {
  const urls = [];
  for (const block of String(html).matchAll(/<li class="b_algo"[\s\S]*?<\/li>/g)) {
    const link = block[0].match(/<h2[^>]*>\s*<a[^>]*href="([^"]+)"/);
    const url = link ? decodeBingRedirect(link[1]) : null;
    if (url && !hostnameOf(url).endsWith("bing.com")) urls.push(url);
  }
  return urls;
}

export function parseDuckDuckGoResults(html) {
  const urls = [];
  for (const match of String(html).matchAll(/uddg=([^"&]+)/g)) {
    try {
      const url = decodeURIComponent(match[1]);
      if (/^https?:\/\//.test(url) && !url.includes("duckduckgo.com")) urls.push(url);
    } catch {
      /* ignore malformed redirect */
    }
  }
  return urls;
}

async function bingSearch(query, signal) {
  const response = await fetch(`https://www.bing.com/search?setlang=en&cc=us&q=${encodeURIComponent(query)}`, {
    headers: { "User-Agent": BROWSER_UA, "Accept-Language": "en-US,en;q=0.9" },
    signal: withTimeout(signal, 12_000),
  });
  if (response.status === 429 || response.status === 403) {
    await response.body?.cancel().catch(() => {});
    return { blocked: true, urls: [], message: `Bing refused automated requests (HTTP ${response.status})` };
  }
  if (!response.ok) {
    await response.body?.cancel().catch(() => {});
    throw new Error(`Bing returned HTTP ${response.status}`);
  }
  const html = await response.text();
  const urls = parseBingResults(html);
  if (!urls.length && /captcha|unusual traffic|are you a robot/i.test(html)) {
    return { blocked: true, urls: [], message: "Bing asked for a human check" };
  }
  return { urls };
}

async function duckDuckGoSearch(query, signal) {
  const response = await fetch(`https://html.duckduckgo.com/html/?q=${encodeURIComponent(query)}`, {
    headers: { "User-Agent": BROWSER_UA, "Accept-Language": "en-US,en;q=0.9" },
    signal: withTimeout(signal, 12_000),
  });
  const html = await response.text();
  if (response.status === 202 || /anomaly|Unfortunately, bots use DuckDuckGo too|challenge/i.test(html.slice(0, 20_000))) {
    return { blocked: true, urls: [], message: "DuckDuckGo showed a bot challenge" };
  }
  if (!response.ok) throw new Error(`DuckDuckGo returned HTTP ${response.status}`);
  return { urls: parseDuckDuckGoResults(html) };
}

async function wikipediaSearch(query, signal) {
  const params = new URLSearchParams({
    action: "query",
    list: "search",
    srsearch: query,
    srlimit: "4",
    srprop: "",
    format: "json",
    origin: "*",
  });
  const response = await fetch(`https://en.wikipedia.org/w/api.php?${params}`, {
    headers: { "User-Agent": BOT_UA, Accept: "application/json" },
    signal: withTimeout(signal, 12_000),
  });
  if (response.status === 429) {
    await response.body?.cancel().catch(() => {});
    return { blocked: true, urls: [], message: "Wikipedia search is rate limiting this server" };
  }
  if (!response.ok) {
    await response.body?.cancel().catch(() => {});
    throw new Error(`Wikipedia search returned HTTP ${response.status}`);
  }
  const data = await response.json();
  const urls = (data?.query?.search || []).map(
    (item) => `https://en.wikipedia.org/wiki/${encodeURIComponent(String(item.title).replace(/ /g, "_"))}`
  );
  return { urls };
}

const ENGINES = [
  { name: "Bing", run: bingSearch, gapMs: 350 },
  { name: "DuckDuckGo", run: duckDuckGoSearch, gapMs: 700 },
  { name: "Wikipedia", run: wikipediaSearch, gapMs: 120 },
];

/**
 * A polite search client: paces each engine, retries once on transient failures, and turns
 * off an engine for the rest of the run after repeated blocks. Failures are reported, never
 * hidden, so a blocked search cannot look like "no matches".
 */
export function createSearchProvider({ engines = ENGINES, failureLimit = 3 } = {}) {
  const state = engines.map((engine) => ({
    engine,
    name: engine.name,
    queries: 0,
    succeeded: 0,
    failed: 0,
    blocked: 0,
    consecutive: 0,
    disabled: false,
    lastError: "",
    chain: Promise.resolve(),
    nextAt: 0,
  }));

  const schedule = (entry, task) => {
    const run = entry.chain.then(async () => {
      const wait = entry.nextAt - Date.now();
      if (wait > 0) await sleep(wait);
      entry.nextAt = Date.now() + entry.engine.gapMs;
      return task();
    });
    entry.chain = run.catch(() => {});
    return run;
  };

  async function queryEngine(entry, query, signal) {
    if (entry.disabled) return { urls: [], skipped: true };
    entry.queries++;
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        const outcome = await schedule(entry, () => entry.engine.run(query, signal));
        if (outcome.blocked) {
          entry.blocked++;
          entry.consecutive++;
          entry.lastError = outcome.message;
          if (entry.consecutive >= failureLimit) entry.disabled = true;
          return { urls: [], error: outcome.message };
        }
        entry.succeeded++;
        entry.consecutive = 0;
        return { urls: outcome.urls };
      } catch (error) {
        if (signal?.aborted) throw abortError();
        entry.lastError = error?.name === "TimeoutError" ? `${entry.name} timed out` : error?.message || "Search failed";
        if (attempt === 1) {
          entry.failed++;
          entry.consecutive++;
          if (entry.consecutive >= failureLimit) entry.disabled = true;
          return { urls: [], error: entry.lastError };
        }
        await sleep(600, signal);
      }
    }
    return { urls: [] };
  }

  return {
    async search(query, signal) {
      const settled = await Promise.all(state.map((entry) => queryEngine(entry, query, signal)));
      const urls = [];
      const longest = Math.max(0, ...settled.map((item) => item.urls.length));
      for (let rank = 0; rank < longest; rank++) {
        for (const item of settled) if (item.urls[rank]) urls.push(item.urls[rank]);
      }
      const answered = settled.some((item, index) => !item.skipped && !item.error && state[index].queries > 0);
      return {
        urls: [...new Set(urls)],
        answered,
        errors: settled.map((item, index) => (item.error ? { engine: state[index].name, message: item.error } : null)).filter(Boolean),
      };
    },
    stats() {
      return state.map(({ name, queries, succeeded, failed, blocked, disabled, lastError }) => ({
        name,
        queries,
        succeeded,
        failed,
        blocked,
        disabled,
        lastError,
      }));
    },
  };
}
