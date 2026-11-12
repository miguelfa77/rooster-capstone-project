import type { KpiPayload } from "@/lib/types";
import styles from "./KpiCard.module.css";

export default function KpiCard({ p }: { p: KpiPayload }) {
  return (
    <div className={styles.card}>
      <div className={styles.label}>{p.label}</div>
      <div className={styles.value}>
        <span>{p.value}</span>
        {p.unit ? <span className={styles.unit}>{p.unit}</span> : null}
      </div>
      {p.delta ? <div className={styles.delta}>{p.delta}</div> : null}
    </div>
  );
}
