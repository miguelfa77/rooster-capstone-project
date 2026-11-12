// Server-side proxy: /api/rooster/{conversations,intel}/... -> ${ROOSTER_API_URL}/...
// Adds the shared API key; the browser never sees it or talks to FastAPI directly.
export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const ALLOWED_PREFIXES = new Set(["conversations", "intel"]);
const DROP_REQUEST_HEADERS = ["host", "cookie", "x-rooster-key", "connection", "content-length"];
// fetch() already decoded the body, so length/encoding headers no longer apply.
const DROP_RESPONSE_HEADERS = ["content-encoding", "content-length", "transfer-encoding", "connection"];

type Context = { params: Promise<{ path: string[] }> };

function json(status: number, detail: string): Response {
  return Response.json({ detail }, { status });
}

async function proxy(request: Request, { params }: Context): Promise<Response> {
  const { path } = await params;
  const valid =
    path.length > 0 && ALLOWED_PREFIXES.has(path[0]) && path.every((s) => s !== "" && s !== "." && s !== "..");
  if (!valid) return json(404, "not_found");

  const headers = new Headers(request.headers);
  for (const name of DROP_REQUEST_HEADERS) headers.delete(name);
  headers.set("x-rooster-key", process.env.ROOSTER_API_KEY ?? "");

  let upstream: Response;
  try {
    const base = (process.env.ROOSTER_API_URL ?? "").replace(/\/+$/, "");
    const url = `${base}/${path.map(encodeURIComponent).join("/")}${new URL(request.url).search}`;
    upstream = await fetch(url, {
      method: request.method,
      headers,
      body: request.method === "GET" ? undefined : await request.arrayBuffer(),
      signal: request.signal,
      cache: "no-store",
    });
  } catch {
    return json(502, "upstream_unavailable");
  }

  const out = new Headers(upstream.headers);
  for (const name of DROP_RESPONSE_HEADERS) out.delete(name);
  if (out.get("content-type")?.startsWith("text/event-stream")) out.set("cache-control", "no-cache");
  return new Response(upstream.body, { status: upstream.status, statusText: upstream.statusText, headers: out });
}

export const GET = proxy;
export const POST = proxy;
