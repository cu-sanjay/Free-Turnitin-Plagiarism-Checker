export const verifiedOn = "2026-10-14";

export const plagiarismTools = [
  {
    name: "Plagiarism.Free",
    url: "https://plagiarism.free/",
    logo: "/logos/plagiarism-free.png",
    kind: "Plagiarism",
    account: "No account for the free tier.",
    limit:
      "The site states 2,500 words per check and 3 checks per day without an account. A free account is listed at 10,000 words and 8 checks per day.",
  },
  {
    name: "Search Engine Reports",
    url: "https://searchenginereports.net/plagiarism-checker",
    logo: "/logos/searchenginereports.webp",
    kind: "Plagiarism",
    account: "The page states that no sign-up and no sign-in are required.",
    limit: "The checker page shows a 1,000-word counter and describes checks of up to 1,000 words.",
  },
  {
    name: "SmallSEOTools",
    url: "https://smallseotools.com/plagiarism-checker/",
    logo: "/logos/smallseotools.svg",
    kind: "Plagiarism",
    account:
      "The checker is presented as a free paste-and-scan page. A testimonial on that page says no sign-up is required; the site’s own FAQ does not separately spell out an account rule.",
    limit: "The page counter and FAQ both state up to 1,000 words per free scan.",
  },
  {
    name: "PrePostSEO",
    url: "https://www.prepostseo.com/plagiarism-checker",
    logo: "/logos/prepostseo.svg",
    kind: "Plagiarism",
    account: "The pricing table includes visitors who are free and unregistered, so a check can start without an account.",
    limit: "Unregistered use is listed only as “Limited words.” The page does not publish the exact free word count.",
  },
  {
    name: "DupliChecker",
    url: "https://www.duplichecker.com/",
    logo: "/logos/duplichecker.png",
    kind: "Plagiarism",
    account: "The page says the tool does not ask for registration to start a check.",
    limit:
      "A free word cap is not clearly separated from a “25,000 words” line shown next to a premium prompt, so that figure is not treated as the free limit.",
  },
  {
    name: "Plagiarism Detector",
    url: "https://plagiarismdetector.net/",
    logo: "/logos/plagiarismdetector.png",
    kind: "Plagiarism",
    account: "The page describes pasting text and checking for free. It does not say an account is required.",
    limit: "It describes unlimited checks. No numeric word cap is published on the page reviewed.",
  },
];

export const aiTools = [
  {
    name: "Scribbr AI Detector",
    url: "https://www.scribbr.com/ai-detector/",
    logo: "/logos/scribbr.ico",
    kind: "AI detection",
    account: "The page states that no sign-up is required.",
    limit: "The page states unlimited free checks, with up to 1,200 words per submission.",
  },
  {
    name: "GPTZero",
    url: "https://gptzero.me/",
    logo: "/logos/gptzero.ico",
    kind: "AI detection",
    account: "The FAQ says scans over 10,000 characters require a free account first, which means shorter scans can start without one.",
    limit: "10,000 characters before an account is required, as stated in the FAQ.",
  },
  {
    name: "Sapling AI Detector",
    url: "https://sapling.ai/ai-content-detector",
    logo: "/logos/sapling.png",
    kind: "AI detection",
    account:
      "The FAQ describes a free version. The same page also refers to signing in for dashboard history, so an account may be involved beyond the basic paste box.",
    limit: "The FAQ states the free version is truncated to 2,000 characters. Paid plans are listed up to 100,000 characters.",
  },
];
