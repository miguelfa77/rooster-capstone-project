import type { CompositePayload } from "@/lib/types";
import Primitive from "./Primitive";
import styles from "./Composite.module.css";

export default function Composite({ p }: { p: CompositePayload }) {
  return (
    <section className={styles.composite}>
      {p.heading ? <h3 className={styles.heading}>{p.heading}</h3> : null}
      {p.blocks.map((block, i) => (
        <Primitive key={i} p={block} />
      ))}
    </section>
  );
}
