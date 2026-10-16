import { Link } from "wouter";

export default function AboutPage() {
  return (
    <div className="page">
      <header className="page-intro">
        <p className="kicker">A note from me</p>
        <h1>What I can still check for you</h1>
      </header>
      <article className="letter">
        <p>
          Hey. For a while I was the one emailing you a Turnitin report, no charge, straight from a class I had. That
          class subscription is over, so I can’t do that anymore.
        </p>
        <p>
          Turnitin also shut the old endpoints this site used. Their team removed or fixed them. I don’t have a way
          into the research-paper or IEEE databases either, so I can’t hand you an exact Turnitin report from here.
        </p>
        <p>
          What you can still do on this page is a simpler check: your writing against public pages I can open. The
          percentage is the share of your words that actually appear on those pages. It is not a Turnitin similarity
          index, and a low score does not mean the work is original.
        </p>
        <p>
          I’m looking for a way to get you those free reports again. If that access comes back, I’ll put it here.
        </p>
        <p>
          Your text stays on this server for the check. I don’t add a new analytics service, and I don’t send the whole
          document to anyone else. I do send short phrases from your text to public search engines (Bing, DuckDuckGo
          when it answers, and Wikipedia search) so I can find pages to compare. If you keep a result, it stays in this
          browser.
        </p>
        <p>
          <Link href="/tools">Other free checkers</Link>
          {" · "}
          <Link href="/">Back to your document</Link>
        </p>
      </article>
    </div>
  );
}
