import { formatNumber } from "@/lib/format";
import styles from "./Map.module.css";

/** Gradient legend: title, color ramp and its min/max. */
export default function Legend({ title, ramp, range }: { title: string; ramp: string[]; range: [number, number] }) {
  return (
    <div className={styles.legend}>
      <div className={styles.legendTitle}>{title}</div>
      <div className={styles.legendBar} style={{ background: `linear-gradient(to right, ${ramp.join(", ")})` }} />
      <div className={styles.legendRange}>
        <span>{formatNumber(range[0])}</span>
        <span>{formatNumber(range[1])}</span>
      </div>
    </div>
  );
}
