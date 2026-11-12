import styles from "./FallbackCard.module.css";

/** The graceful-fallback card: why the turn could not answer and what to try. */
export default function FallbackCard({ reason, suggestion }: { reason: string; suggestion: string }) {
  return (
    <div className={styles.card} role="alert">
      <p className={styles.reason}>{reason}</p>
      {suggestion ? <p className={styles.suggestion}>{suggestion}</p> : null}
    </div>
  );
}
