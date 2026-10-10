const DEFAULT_MIN_MATCH_WORDS = 5;

function getExcludedRanges(text, options) {
  const ranges = [];
  if (options.excludeQuotes) {
    for (const match of text.matchAll(/"[^"\r\n]{2,}"|“[^”\r\n]{2,}”|^\s*>[^\r\n]*/gm)) {
      ranges.push({ start: match.index, end: match.index + match[0].length });
    }
  }
  if (options.excludeReferences) {
    const heading =
      /^[ \t]*(?:#{1,6}[ \t]*)?(?:references|works cited|bibliography|sources)[ \t]*:?[ \t]*$/im.exec(text);
    if (heading) {
      ranges.push({ start: heading.index, end: text.length });
    }
  }
  return ranges;
}

function getTokens(text, excludedRanges = []) {
  return Array.from(text.matchAll(/[\p{L}\p{N}]+(?:['’][\p{L}\p{N}]+)*/gu), (match) => ({
    word: match[0].toLowerCase().replaceAll("’", "'"),
    start: match.index,
    end: match.index + match[0].length,
  })).filter(({ start }) =>
    !excludedRanges.some((range) => start >= range.start && start < range.end)
  );
}

function createNgramIndex(tokens, minMatchWords) {
  const index = new Map();
  for (let position = 0; position <= tokens.length - minMatchWords; position++) {
    const key = tokens
      .slice(position, position + minMatchWords)
      .map(({ word }) => word)
      .join(" ");
    const positions = index.get(key) ?? [];
    positions.push(position);
    index.set(key, positions);
  }
  return index;
}

function countCoveredTokens(matches) {
  const positions = new Set();
  for (const match of matches) {
    for (let position = match.tokenStart; position < match.tokenEnd; position++) {
      positions.add(position);
    }
  }
  return positions.size;
}

function mergeHighlightRanges(sourceResults) {
  const ranges = sourceResults
    .flatMap(({ name, matches }) =>
      matches.map(({ start, end }) => ({ start, end, sources: [name] }))
    )
    .sort((left, right) => left.start - right.start || left.end - right.end);
  const merged = [];

  for (const range of ranges) {
    const previous = merged.at(-1);
    if (previous && range.start <= previous.end) {
      previous.end = Math.max(previous.end, range.end);
      previous.sources = [...new Set([...previous.sources, ...range.sources])];
    } else {
      merged.push(range);
    }
  }

  return merged;
}

export function compareAgainstSources(
  text,
  sources,
  options = {
    excludeQuotes: false,
    excludeReferences: false,
    minMatchWords: DEFAULT_MIN_MATCH_WORDS,
  }
) {
  const minMatchWords = options.minMatchWords ?? DEFAULT_MIN_MATCH_WORDS;
  const allSubmissionTokens = getTokens(text);
  const submissionTokens = getTokens(text, getExcludedRanges(text, options));
  const submissionWords = submissionTokens.map(({ word }) => word);
  const sourceResults = sources.map((source) => {
    if (source.exclude) {
      return {
        id: source.id,
        name: source.name,
        url: source.url ?? "",
        matches: [],
        matchedWords: 0,
        similarityPercentage: 0,
        excluded: true,
      };
    }
    const referenceTokens = getTokens(source.text);
    const index = createNgramIndex(referenceTokens, minMatchWords);
    const matches = [];

    for (
      let submissionStart = 0;
      submissionStart <= submissionTokens.length - minMatchWords;
    ) {
      const key = submissionWords
        .slice(submissionStart, submissionStart + minMatchWords)
        .join(" ");
      const referencePositions = index.get(key);
      if (!referencePositions) {
        submissionStart++;
        continue;
      }

      let bestLength = 0;
      let bestReferenceStart = 0;
      const maxPossibleLength = Math.min(
        submissionTokens.length - submissionStart,
        referenceTokens.length
      );
      for (const referenceStart of referencePositions) {
        let length = minMatchWords;
        while (
          submissionStart + length < submissionTokens.length &&
          referenceStart + length < referenceTokens.length &&
          submissionWords[submissionStart + length] ===
            referenceTokens[referenceStart + length].word
        ) {
          length++;
        }
        if (length > bestLength) {
          bestLength = length;
          bestReferenceStart = referenceStart;
          if (bestLength === maxPossibleLength) break;
        }
      }

      if (bestLength < minMatchWords) {
        submissionStart++;
        continue;
      }

      const submissionEnd = submissionStart + bestLength;
      const referenceEnd = bestReferenceStart + bestLength;
      matches.push({
        tokenStart: submissionStart,
        tokenEnd: submissionEnd,
        start: submissionTokens[submissionStart].start,
        end: submissionTokens[submissionEnd - 1].end,
        text: text.slice(
          submissionTokens[submissionStart].start,
          submissionTokens[submissionEnd - 1].end
        ),
        referenceText: source.text.slice(
          referenceTokens[bestReferenceStart].start,
          referenceTokens[referenceEnd - 1].end
        ),
        matchedWords: bestLength,
      });
      submissionStart = submissionEnd;
    }

    return {
      id: source.id,
      name: source.name,
      url: source.url ?? "",
      matches,
      matchedWords: countCoveredTokens(matches),
      similarityPercentage: 0,
    };
  });

  const matchedWords = countCoveredTokens(sourceResults.flatMap(({ matches }) => matches));
  const similarityPercentage = submissionTokens.length
    ? Math.round((matchedWords / submissionTokens.length) * 100)
    : 0;

  for (const source of sourceResults) {
    source.similarityPercentage = submissionTokens.length
      ? Math.round((source.matchedWords / submissionTokens.length) * 100)
      : 0;
  }

  return {
    minMatchWords,
    wordCount: submissionTokens.length,
    excludedWords: allSubmissionTokens.length - submissionTokens.length,
    matchedWords,
    similarityPercentage,
    sources: sourceResults
      .filter(({ matches }) => matches.length > 0)
      .sort((left, right) => right.matchedWords - left.matchedWords),
    excludedSourceCount: sourceResults.filter(({ excluded }) => excluded).length,
    highlightRanges: mergeHighlightRanges(sourceResults),
  };
}
