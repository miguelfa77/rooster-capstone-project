"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { COPY } from "@/lib/copy";
import styles from "./TopBar.module.css";

const LINKS = [
  { href: "/", label: COPY.navChat },
  { href: "/intel", label: COPY.navIntel },
];

export default function TopBar() {
  const pathname = usePathname();
  return (
    <header className={styles.bar}>
      <div className={styles.brand}>
        <span className={styles.wordmark}>{COPY.wordmark}</span>
        <span className={styles.tagline}>{COPY.tagline}</span>
      </div>
      <nav className={styles.nav}>
        {LINKS.map(({ href, label }) => {
          const active = href === "/" ? pathname === "/" : pathname === href || pathname.startsWith(href + "/");
          return (
            <Link key={href} href={href} className={styles.link} aria-current={active ? "page" : undefined}>
              {label}
            </Link>
          );
        })}
      </nav>
    </header>
  );
}
