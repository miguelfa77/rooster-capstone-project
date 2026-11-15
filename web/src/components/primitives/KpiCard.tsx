import type { KpiPayload } from "@/lib/types";
import styles from "./KpiCard.module.css";

type Trend = "up" | "down" | "neutral";

/** The delta's direction from its sign: "+" or "▲" up, "−", "-" or "▼" down, anything else neutral. */
function trendOf(delta: string): Trend {
  const t = delta.trimStart();
  if (t.startsWith("+") || t.startsWith("▲")) return "up";
  if (t.startsWith("−") || t.startsWith("-") || t.startsWith("▼")) return "down";
  return "neutral";
}

/** `sub` is an optional muted line under the value (used by the Intel framing cards). */
export default function KpiCard({ p, sub }: { p: KpiPayload; sub?: string }) {
  return (
    <div className={styles.card}>
      <div className={styles.label}>{p.label}</div>
      <div className={styles.value}>
        <span>{p.value}</span>
        {p.unit ? <span className={styles.unit}>{p.unit}</span> : null}
      </div>
      {p.delta ? <Delta text={p.delta} /> : null}
      {sub ? <div className={styles.sub}>{sub}</div> : null}
    </div>
  );
}

function Delta({ text }: { text: string }) {
  const trend = trendOf(text);
  const hasArrow = /^\s*[▲▼]/.test(text);
  const arrow = hasArrow ? "" : trend === "up" ? "▲ " : trend === "down" ? "▼ " : "";
  return (
    <div className={styles.delta} data-trend={trend}>
      {arrow}
      {text}
    </div>
  );
}
