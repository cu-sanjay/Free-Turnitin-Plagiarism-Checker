import { Link, useLocation } from "wouter";
import { useEffect } from "react";

const NotFound = () => {
  const [location] = useLocation();
  useEffect(() => {
    console.error("404 Error: User attempted to access non-existent route:", location);
  }, [location]);
  return (
    <div className="page">
      <p className="kicker">404</p>
      <h1 data-testid="text-404" style={{ fontSize: "4rem", letterSpacing: "-0.05em" }}>
        Page not found
      </h1>
      <p className="lede" data-testid="text-not-found" style={{ margin: "0.8rem 0 1.2rem" }}>
        That address is not part of this checker.
      </p>
      <Link href="/" className="primary-button" data-testid="button-home" style={{ textDecoration: "none" }}>
        Return to Check
      </Link>
    </div>
  );
};

export default NotFound;
