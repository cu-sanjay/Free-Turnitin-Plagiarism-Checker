import assert from "node:assert/strict";
import test from "node:test";
import { htmlToText, isPrivateAddress, parseBingResults, parseDuckDuckGoResults, shouldFetchUrl } from "./search.js";

test("Bing result parser reads the visible result URLs, including ck/a redirects", () => {
  const html = `
    <li class="b_algo"><h2><a href="https://en.wikipedia.org/wiki/Lighthouse">Lighthouse</a></h2></li>
    <li class="b_algo"><h2><a href="https://www.bing.com/ck/a?!&&amp;p=abc&amp;u=a1aHR0cHM6Ly9zY2llbmNlLm5hc2EuZ292L21pc3Npb24vaHViYmxlLw&amp;ntb=1">NASA</a></h2></li>
  `;
  const urls = parseBingResults(html);
  assert.equal(urls[0], "https://en.wikipedia.org/wiki/Lighthouse");
  assert.equal(urls[1], "https://science.nasa.gov/mission/hubble/");
});

test("DuckDuckGo parser ignores the search host itself", () => {
  const html = `uddg=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FHubble_Space_Telescope&amp; uddg=https%3A%2F%2Fduckduckgo.com%2F`;
  assert.deepEqual(parseDuckDuckGoResults(html), ["https://en.wikipedia.org/wiki/Hubble_Space_Telescope"]);
});

test("htmlToText strips scripts and keeps readable words", () => {
  const { title, text } = htmlToText(
    `<html><head><title>Lighthouse - Wikipedia</title><script>secret()</script></head><body><p>A lighthouse is a tower.</p></body></html>`
  );
  assert.equal(title, "Lighthouse - Wikipedia");
  assert.match(text, /A lighthouse is a tower/);
  assert.doesNotMatch(text, /secret/);
});

test("private addresses and skipped hosts are not fetched", () => {
  assert.equal(isPrivateAddress("127.0.0.1"), true);
  assert.equal(isPrivateAddress("10.0.0.8"), true);
  assert.equal(isPrivateAddress("1.1.1.1"), false);
  assert.equal(shouldFetchUrl("https://en.wikipedia.org/wiki/Lighthouse"), true);
  assert.equal(shouldFetchUrl("https://www.youtube.com/watch?v=1"), false);
  assert.equal(shouldFetchUrl("https://example.test/paper.pdf"), false);
});
