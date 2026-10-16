import { useEffect, useState } from "react";
import { Link, useLocation } from "wouter";
import { Moon, Sun } from "lucide-react";

const LINKS = [
  { href: "/", label: "Check" },
  { href: "/history", label: "History" },
  { href: "/tools", label: "Free tools" },
  { href: "/about", label: "About" },
];

function readTheme() {
  if (typeof document === "undefined") return "light";
  return document.documentElement.classList.contains("dark") ? "dark" : "light";
}

export default function AppShell({ children }) {
  const [location] = useLocation();
  const [theme, setTheme] = useState(readTheme);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    try {
      localStorage.setItem("turnitout-theme", theme);
    } catch {
      /* storage can be unavailable in private modes */
    }
  }, [theme]);

  return (
    <div className="app-shell">
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <header className="topbar">
        <Link href="/" className="brand">
          <span className="brand-mark" aria-hidden="true">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <rect x="1" y="1" width="9" height="11" rx="1.5" stroke="currentColor" strokeWidth="1.4" />
              <path d="M5 6.5h6.5M5 9h4" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
            </svg>
          </span>
          Turnitout
        </Link>
        <nav className="nav-links" aria-label="Primary">
          {LINKS.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              aria-current={location === link.href ? "page" : undefined}
            >
              {link.label}
            </Link>
          ))}
        </nav>
        <button
          type="button"
          className="theme-toggle"
          aria-pressed={theme === "dark"}
          aria-label={theme === "dark" ? "Switch to light appearance" : "Switch to dark appearance"}
          onClick={() => setTheme((current) => (current === "dark" ? "light" : "dark"))}
        >
          {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
        </button>
      </header>
      <main id="main">{children}</main>
    </div>
  );
}
