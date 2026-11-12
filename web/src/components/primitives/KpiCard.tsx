import type { KpiPayload } from "@/lib/types";
import styles from "./KpiCard.module.css";

/** `sub` is an optional muted line under the value (used by the Intel framing cards). */
export default function KpiCard({ p, sub }: { p: KpiPayload; sub?: string }) {
  return (
    <div className={styles.card}>
      <div className={styles.label}>{p.label}</div>
      <div className={styles.value}>
        <span>{p.value}</span>
        {p.unit ? <span className={styles.unit}>{p.unit}</span> : null}
      </div>
      {p.delta ? <div className={styles.delta}>{p.delta}</div> : null}
      {sub ? <div className={styles.sub}>{sub}</div> : null}
    </div>
  );
}
