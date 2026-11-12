import { useId, type ReactNode } from "react";
import styles from "./Intel.module.css";

/** A titled page region; the heading names it for assistive tech. */
export default function Section({ title, children }: { title: string; children: ReactNode }) {
  const id = useId();
  return (
    <section className={styles.section} aria-labelledby={id}>
      <h2 id={id} className={styles.heading}>
        {title}
      </h2>
      {children}
    </section>
  );
}
