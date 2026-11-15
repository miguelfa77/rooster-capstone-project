// The one place that fixes the e2e environment: ports, secrets and the session cookie. Read by
// playwright.config.ts (to start the servers) and tests/e2e/helpers.ts (to talk to them), so the
// two cannot drift apart.

/** Ports can be overridden (E2E_WEB_PORT / E2E_API_PORT) when 3100/8787 are taken by another run. */
export const WEB_PORT = Number(process.env.E2E_WEB_PORT ?? 3100);
export const API_PORT = Number(process.env.E2E_API_PORT ?? 8787);
export const BASE_URL = `http://127.0.0.1:${WEB_PORT}`;
export const FAKE_API = `http://localhost:${API_PORT}`;

/** Sign-in is never exercised against Google: specs mint a session cookie with this secret. */
export const AUTH_SECRET = "test-secret";
/** Auth.js on http; the cookie name is also its JWT salt. */
export const SESSION_COOKIE = "authjs.session-token";
/** Both test users (see tests/fake-api/server.mjs) are on the allowlist. */
export const ALLOWED_EMAILS = "ana@fund.com,luis@fund.com";
