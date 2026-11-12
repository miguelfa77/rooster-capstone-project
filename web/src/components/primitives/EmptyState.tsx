import { COPY } from "@/lib/copy";
import styles from "./EmptyState.module.css";

export default function EmptyState({ message = COPY.emptyChart }: { message?: string }) {
  return <p className={styles.empty}>{message}</p>;
}
