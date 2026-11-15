import type { Mock } from "vitest";

export type Route = (url: string, init?: RequestInit) => Response | Promise<Response>;

/** Answers `fetchMock` by "<METHOD> <url>" key; an unlisted request throws so a test never hits a silent default. */
export function routes(fetchMock: Mock, handlers: Record<string, Route>): void {
  fetchMock.mockImplementation(async (url: string, init?: RequestInit) => {
    const key = `${init?.method ?? "GET"} ${url}`;
    const handler = handlers[key];
    if (!handler) throw new Error(`unexpected fetch ${key}`);
    return handler(url, init);
  });
}
