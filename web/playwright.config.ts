import { defineConfig, devices } from "@playwright/test";

// Ports can be overridden (E2E_WEB_PORT / E2E_API_PORT) when 3100/8787 are taken by another run.
const WEB_PORT = Number(process.env.E2E_WEB_PORT ?? 3100);
const API_PORT = Number(process.env.E2E_API_PORT ?? 8787);
const AUTH_SECRET = "test-secret";

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
    baseURL: `http://127.0.0.1:${WEB_PORT}`,
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
      url: `http://localhost:${API_PORT}/health`,
      env: { FAKE_API_PORT: String(API_PORT) },
      reuseExistingServer: !process.env.CI,
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
      url: `http://127.0.0.1:${WEB_PORT}`,
      env: {
        PORT: String(WEB_PORT),
        HOSTNAME: "127.0.0.1",
        ROOSTER_API_URL: `http://localhost:${API_PORT}`,
        ROOSTER_API_KEY: "test",
        // Sign-in is never exercised against Google: specs mint a session cookie with this secret.
        AUTH_SECRET: AUTH_SECRET,
        AUTH_URL: `http://127.0.0.1:${WEB_PORT}`,
        AUTH_GOOGLE_ID: "e2e-google-id",
        AUTH_GOOGLE_SECRET: "e2e-google-secret",
        ROOSTER_ALLOWED_EMAILS: "ana@fund.com,luis@fund.com",
      },
      reuseExistingServer: !process.env.CI,
      timeout: 300_000,
    },
  ],
});
