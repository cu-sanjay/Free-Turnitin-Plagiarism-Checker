import { aiTools, plagiarismTools, verifiedOn } from "@/data/free-tools";

function ToolCard({ tool }) {
  return (
    <article className="tool-card">
      <p className="kicker">{tool.kind}</p>
      <h2 className="tool-heading">
        <img className="tool-logo" src={tool.logo} alt="" />
        <a href={tool.url} target="_blank" rel="noopener noreferrer">
          {tool.name}
        </a>
      </h2>
      <p>{tool.account}</p>
      <p className="meta" style={{ marginTop: "0.45rem" }}>
        {tool.limit}
      </p>
    </article>
  );
}

export default function ToolsPage() {
  return (
    <div className="page">
      <header className="page-intro">
        <p className="kicker">Free tools</p>
        <h1>Other checkers you can open yourself</h1>
        <p className="lede" style={{ maxWidth: "42rem", marginTop: "0.8rem" }}>
          These open in a new tab. I don’t send your document to them. I read each limit on the site itself on{" "}
          {verifiedOn}. If they didn’t publish a number, I left that gap.
        </p>
      </header>
      <h2 style={{ marginTop: "1.6rem", letterSpacing: "-0.03em" }}>Plagiarism</h2>
      <div className="tool-grid">
        {plagiarismTools.map((tool) => (
          <ToolCard key={tool.url} tool={tool} />
        ))}
      </div>
      <h2 style={{ marginTop: "1.8rem", letterSpacing: "-0.03em" }}>AI detection</h2>
      <div className="tool-grid">
        {aiTools.map((tool) => (
          <ToolCard key={tool.url} tool={tool} />
        ))}
      </div>
    </div>
  );
}
