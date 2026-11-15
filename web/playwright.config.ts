import { defineConfig, devices } from "@playwright/test";

import { ALLOWED_EMAILS, API_PORT, AUTH_SECRET, BASE_URL, FAKE_API, WEB_PORT } from "./tests/e2e/env";

// A server already on these ports is a failure, never a shortcut: a stale build would be tested
// instead of the tree. Set E2E_REUSE_SERVERS=1 to reuse servers you started yourself.
const reuseExistingServer = process.env.E2E_REUSE_SERVERS === "1";

// Optional: point at an existing Chrome/Chromium when `npx playwright install chromium` is not possible.
const executablePath = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE || undefined;

export default defineConfig({
  testDir: "./tests/e2e",
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: "list",
  timeout: 60_000,
  expect: { timeout: 15_000 },
  use: {
    baseURL: BASE_URL,
    trace: "retain-on-failure",
    launchOptions: {
      executablePath,
      // Software WebGL so MapLibre can render in headless runs.
      args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"],
    },
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: [
    {
      command: "node tests/fake-api/server.mjs",
      url: `${FAKE_API}/health`,
      env: { FAKE_API_PORT: String(API_PORT) },
      reuseExistingServer,
      timeout: 30_000,
    },
    {
      // The standalone server (as in web/Dockerfile) needs .next/static and public/ copied next to it.
      command: [
        "npx next build",
        "rm -rf .next/standalone/.next/static .next/standalone/public",
        "cp -r .next/static .next/standalone/.next/static",
        "if [ -d public ]; then cp -r public .next/standalone/public; fi",
        "node .next/standalone/server.js",
      ].join(" && "),
      url: BASE_URL,
      env: {
        PORT: String(WEB_PORT),
        HOSTNAME: "127.0.0.1",
        ROOSTER_API_URL: FAKE_API,
        ROOSTER_API_KEY: "test",
        AUTH_SECRET,
        AUTH_URL: BASE_URL,
        AUTH_GOOGLE_ID: "e2e-google-id",
        AUTH_GOOGLE_SECRET: "e2e-google-secret",
        ROOSTER_ALLOWED_EMAILS: ALLOWED_EMAILS,
      },
      reuseExistingServer,
      timeout: 300_000,
    },
  ],
});
