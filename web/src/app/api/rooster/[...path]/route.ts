// Server-side proxy: /api/rooster/{conversations,intel,me}/... -> ${ROOSTER_API_URL}/...
// Adds the shared API key and the signed-in user's id (from the session, never from the
// request); the browser never sees the key or talks to FastAPI directly.
import { currentUserId } from "@/auth";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const ALLOWED_PREFIXES = new Set(["conversations", "intel", "me"]);
const FORWARD_REQUEST_HEADERS = ["content-type", "accept", "accept-language"];
// fetch() already decoded the body, so length/encoding headers no longer apply.
const DROP_RESPONSE_HEADERS = ["content-encoding", "content-length", "transfer-encoding", "connection"];

type Context = { params: Promise<{ path: string[] }> };

let misconfigLogged = false;

function json(status: number, detail: string): Response {
  return Response.json({ detail }, { status });
}

async function proxy(request: Request, { params }: Context): Promise<Response> {
  const userId = await currentUserId();
  if (!userId) return json(401, "signed_out");

  const { path } = await params;
  const valid =
    path.length > 0 && ALLOWED_PREFIXES.has(path[0]) && path.every((s) => s !== "" && s !== "." && s !== "..");
  if (!valid) return json(404, "not_found");

  const base = process.env.ROOSTER_API_URL?.replace(/\/+$/, "");
  const key = process.env.ROOSTER_API_KEY;
  if (!base || !key) {
    if (!misconfigLogged) {
      misconfigLogged = true;
      console.error("rooster proxy: ROOSTER_API_URL and ROOSTER_API_KEY must both be set");
    }
    return json(500, "proxy_misconfigured");
  }

  const headers = new Headers();
  for (const name of FORWARD_REQUEST_HEADERS) {
    const value = request.headers.get(name);
    if (value !== null) headers.set(name, value);
  }
  headers.set("x-rooster-key", key);
  headers.set("x-rooster-user", userId);

  let upstream: Response;
  try {
    const url = `${base}/${path.map(encodeURIComponent).join("/")}${new URL(request.url).search}`;
    upstream = await fetch(url, {
      method: request.method,
      headers,
      body: request.method === "GET" ? undefined : await request.arrayBuffer(),
      signal: request.signal,
      cache: "no-store",
    });
  } catch (err) {
    if (!request.signal.aborted) console.error("rooster proxy: upstream fetch failed", err);
    return json(502, "upstream_unavailable");
  }

  const out = new Headers(upstream.headers);
  for (const name of DROP_RESPONSE_HEADERS) out.delete(name);
  if (out.get("content-type")?.startsWith("text/event-stream")) out.set("cache-control", "no-cache");
  return new Response(upstream.body, { status: upstream.status, statusText: upstream.statusText, headers: out });
}

export const GET = proxy;
export const POST = proxy;
export const PATCH = proxy;
export const DELETE = proxy;
