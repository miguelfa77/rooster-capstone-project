import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { TextPayload } from "@/lib/types";
import styles from "./TextBlock.module.css";

export default function TextBlock({ p }: { p: TextPayload }) {
  return (
    <div className={`${styles.text} ${styles[p.role] ?? ""}`}>
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{p.content}</ReactMarkdown>
    </div>
  );
}
