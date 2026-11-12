import { defineConfig, devices } from "@playwright/test";

const WEB_PORT = 3100;
const API_PORT = 8787;

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
    baseURL: `http://localhost:${WEB_PORT}`,
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
      command: `npx next build && npx next start -p ${WEB_PORT}`,
      url: `http://localhost:${WEB_PORT}`,
      env: { ROOSTER_API_URL: `http://localhost:${API_PORT}`, ROOSTER_API_KEY: "test" },
      reuseExistingServer: !process.env.CI,
      timeout: 300_000,
    },
  ],
});
