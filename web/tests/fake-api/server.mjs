// A tiny stand-in for the Rooster FastAPI, used only by the Playwright tests.
// Plain Node `http`, no dependencies. Scripted turns are chosen by the message text:
//   "todo"  -> one primitive of every kind (repo-root tests/fixtures/primitive_payloads.json,
//              its `payloads` plus `folium_code`)
//   "error" -> an error event, then done (nothing stored, as in the real API)
//   other   -> a short text turn
import { randomUUID } from "node:crypto";
import { readFileSync } from "node:fs";
import { createServer } from "node:http";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.FAKE_API_PORT ?? 8787);
const KEY = "test";
const STEP_MS = 30;

const readJson = (path) => JSON.parse(readFileSync(path, "utf8"));
const SHARED = readJson(join(HERE, "../../../tests/fixtures/primitive_payloads.json"));
// Every kind once, plus the folium `code` variant (rendered as a sandboxed iframe).
const PAYLOADS = [...SHARED.payloads, SHARED.folium_code];
const fixture = (name) => readJson(join(HERE, "fixtures", `${name}.json`));

const INTEL = {
  "/intel/framing": fixture("framing"),
  "/intel/neighborhoods": fixture("neighborhoods"),
  "/intel/overlays/transit": fixture("transit"),
  "/intel/overlays/tourism": fixture("tourism"),
  "/intel/listings": fixture("listings"),
  "/intel/parcels": fixture("parcels"),
  "/intel/ranking": fixture("ranking"),
  "/intel/yield-liquidity": fixture("yield-liquidity"),
};

/** conversation id -> stored messages */
const conversations = new Map();

const FOLLOW_UPS = ["¿Y en Russafa?", "Compara Benimaclet con Arrancapins"];

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

/** Split prose into a few word-boundary chunks so it streams as several text_delta events. */
function chunks(text, n = 4) {
  const words = text.split(" ");
  const size = Math.ceil(words.length / n);
  const out = [];
  for (let i = 0; i < words.length; i += size) out.push(words.slice(i, i + size).join(" ") + (i + size < words.length ? " " : ""));
  return out;
}

function script(text) {
  if (text === "error") {
    return {
      events: [
        { type: "thought", text: "Consultando los datos…" },
        { type: "error", reason: "No se pudo completar la consulta.", suggestion: "Prueba a reformular la pregunta." },
      ],
      stored: null,
    };
  }
  if (text === "todo") {
    const prose = PAYLOADS.filter((p) => p.kind === "text").map((p) => p.content).join("\n\n");
    const rest = PAYLOADS.filter((p) => p.kind !== "text");
    return {
      events: [
        { type: "thought", text: "Buscando la rentabilidad por barrio." },
        { type: "thought", text: "Preparando tablas, gráficos y mapas." },
        ...chunks(prose).map((t) => ({ type: "text_delta", text: t })),
        ...rest.map((p) => ({ type: "primitive", primitive: p })),
        { type: "follow_ups", items: FOLLOW_UPS },
      ],
      stored: { blocks: PAYLOADS, summary: prose, primitive_payloads: rest, follow_ups: FOLLOW_UPS },
    };
  }
  const prose = `Respuesta breve sobre: ${text}.`;
  return {
    events: [
      { type: "thought", text: "Pensando la respuesta." },
      ...chunks(prose, 2).map((t) => ({ type: "text_delta", text: t })),
      { type: "follow_ups", items: FOLLOW_UPS },
    ],
    stored: {
      blocks: [{ kind: "text", content: prose, role: "analysis", cited_data: [] }],
      summary: prose,
      primitive_payloads: [],
      follow_ups: FOLLOW_UPS,
    },
  };
}

function send(res, status, body) {
  res.writeHead(status, { "content-type": "application/json" });
  res.end(JSON.stringify(body));
}

async function readBody(req) {
  let raw = "";
  for await (const chunk of req) raw += chunk;
  return raw ? JSON.parse(raw) : {};
}

async function streamTurn(res, id, text) {
  const { events, stored } = script(text);
  res.writeHead(200, { "content-type": "text/event-stream", "cache-control": "no-cache" });
  for (const event of events) {
    res.write(`event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`);
    await sleep(STEP_MS);
  }
  // Saved before `done`, as the real API does, so a refetch on done sees the turn.
  if (stored) {
    conversations
      .get(id)
      .push({ role: "user", content: text }, { role: "assistant", graceful_fallback: null, ...stored });
  }
  res.end(`event: done\ndata: ${JSON.stringify({ type: "done" })}\n\n`);
}

const server = createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);
  const path = url.pathname;
  if (path === "/health") return send(res, 200, { ok: true });
  if (req.headers["x-rooster-key"] !== KEY) return send(res, 401, { detail: "invalid api key" });

  try {
    if (req.method === "GET" && path in INTEL) return send(res, 200, INTEL[path]);
    if (req.method === "POST" && path === "/conversations") {
      const id = randomUUID();
      conversations.set(id, []);
      return send(res, 201, { id });
    }
    const conv = path.match(/^\/conversations\/([^/]+)$/);
    if (req.method === "GET" && conv) {
      const messages = conversations.get(conv[1]);
      return messages ? send(res, 200, { id: conv[1], messages }) : send(res, 404, { detail: "not found" });
    }
    const msg = path.match(/^\/conversations\/([^/]+)\/messages$/);
    if (req.method === "POST" && msg) {
      if (!conversations.has(msg[1])) return send(res, 404, { detail: "not found" });
      const { text } = await readBody(req);
      if (typeof text !== "string" || !text.trim()) return send(res, 422, { detail: "text must not be blank" });
      return await streamTurn(res, msg[1], text.trim());
    }
    return send(res, 404, { detail: "not found" });
  } catch (err) {
    console.error("fake api error", err);
    if (!res.headersSent) send(res, 500, { detail: String(err) });
    else res.end();
  }
});

server.listen(PORT, () => console.log(`fake rooster api on http://localhost:${PORT}`));
