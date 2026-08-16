"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Incident Analyzer" },
  { href: "/comparison", label: "Model Comparison" },
  { href: "/experiments", label: "Experiment Explorer" },
  { href: "/architecture", label: "Architecture" },
];

export function Nav() {
  const path = usePathname();
  return (
    <header className="nav">
      <div>
        <strong>ResolveLM</strong>
        <div className="muted">Incident post-training platform</div>
      </div>
      <nav>
        {LINKS.map((l) => (
          <Link key={l.href} href={l.href} className={path === l.href ? "active" : ""}>
            {l.label}
          </Link>
        ))}
      </nav>
    </header>
  );
}
