import { describe, expect, it } from "vitest";
import { parseSSE } from "@/lib/sse";
import type { TurnEvent } from "@/lib/types";

function streamOf(chunks: (string | Uint8Array)[]): ReadableStream<Uint8Array> {
  const enc = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const c of chunks) controller.enqueue(typeof c === "string" ? enc.encode(c) : c);
      controller.close();
    },
  });
}

async function collect(stream: ReadableStream<Uint8Array>): Promise<TurnEvent[]> {
  const out: TurnEvent[] = [];
  for await (const ev of parseSSE(stream)) out.push(ev);
  return out;
}

describe("parseSSE", () => {
  it("single_event", async () => {
    const events = await collect(streamOf(['event: thought\ndata: {"type":"thought","text":"hola"}\n\n']));
    expect(events).toEqual([{ type: "thought", text: "hola" }]);
  });

  it("multiple_events_one_chunk", async () => {
    const events = await collect(
      streamOf([
        'event: text_delta\ndata: {"type":"text_delta","text":"a"}\n\n' +
          'event: text_delta\ndata: {"type":"text_delta","text":"b"}\n\n' +
          'event: done\ndata: {"type":"done"}\n\n',
      ]),
    );
    expect(events).toEqual([
      { type: "text_delta", text: "a" },
      { type: "text_delta", text: "b" },
      { type: "done" },
    ]);
  });

  it("event_split_across_chunks", async () => {
    const events = await collect(
      streamOf(['event: follow_ups\nda', 'ta: {"type":"follow_ups",', '"items":["x"]}\n', "\n"]),
    );
    expect(events).toEqual([{ type: "follow_ups", items: ["x"] }]);
  });

  it("utf8_char_split_across_chunks", async () => {
    const text = "Ruzafa · 1.200 €á";
    const bytes = new TextEncoder().encode(`data: ${JSON.stringify({ type: "text_delta", text })}\n\n`);
    const euro = bytes.indexOf(0xe2); // first byte of "€" (E2 82 AC)
    const aAcute = bytes.indexOf(0xc3, euro); // first byte of "á" (C3 A1)
    const chunks = [bytes.slice(0, euro + 1), bytes.slice(euro + 1, aAcute + 1), bytes.slice(aAcute + 1)];
    const events = await collect(streamOf(chunks));
    expect(events).toEqual([{ type: "text_delta", text }]);
  });

  it("multiline_data_joined", async () => {
    const events = await collect(streamOf(['data: {"type":"thought",\ndata: "text":"x"}\n\n']));
    expect(events).toEqual([{ type: "thought", text: "x" }]);
  });

  it("ignores_comment_lines", async () => {
    const events = await collect(
      streamOf([': keep-alive\n\nid: 1\nretry: 10\n: note\ndata: {"type":"done"}\n\n']),
    );
    expect(events).toEqual([{ type: "done" }]);
  });
});
