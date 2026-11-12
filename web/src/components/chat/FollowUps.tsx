import { COPY } from "@/lib/copy";
import styles from "./FollowUps.module.css";

/** Clickable pills for the latest assistant message; a muted plain line otherwise. */
export default function FollowUps({
  items,
  interactive,
  onPick,
}: {
  items: string[];
  interactive: boolean;
  onPick: (text: string) => void;
}) {
  if (items.length === 0) return null;
  if (!interactive) {
    return (
      <p className={styles.plain}>
        {COPY.chatFollowUpsLabel} {items.join(" · ")}
      </p>
    );
  }
  return (
    <div className={styles.pills}>
      {items.map((item) => (
        <button key={item} type="button" className={styles.pill} onClick={() => onPick(item)}>
          {item}
        </button>
      ))}
    </div>
  );
}
