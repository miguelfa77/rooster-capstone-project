import type { TurnEvent } from "./types";

/**
 * Parse a `text/event-stream` body into turn events.
 * The event type lives in the JSON payload, so only `data:` fields are used;
 * comments and other fields are ignored. An unterminated event at EOF is dropped.
 */
export async function* parseSSE(stream: ReadableStream<Uint8Array>): AsyncGenerator<TurnEvent> {
  const reader = stream.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n?/g, "\n");
      let sep: number;
      while ((sep = buffer.indexOf("\n\n")) !== -1) {
        const block = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        const data = dataOf(block);
        if (data !== null) yield JSON.parse(data) as TurnEvent;
      }
    }
  } finally {
    reader.releaseLock();
  }
}

function dataOf(block: string): string | null {
  const lines: string[] = [];
  for (const line of block.split("\n")) {
    if (line.startsWith(":")) continue;
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    if (field !== "data") continue;
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    lines.push(value);
  }
  return lines.length ? lines.join("\n") : null;
}
