import { act, render, screen } from "@testing-library/react";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { ThemeProvider } from "@/components/theme-provider";
import { PreferencesProvider, usePreferences } from "@/lib/use-preferences";
import type { Preferences } from "@/lib/types";

vi.mock("sonner", () => ({ toast: { error: vi.fn() } }));

const fetchMock = vi.fn();

beforeAll(() => {
  window.matchMedia = ((query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addEventListener: () => {},
    removeEventListener: () => {},
    addListener: () => {},
    removeListener: () => {},
    dispatchEvent: () => false,
  })) as typeof window.matchMedia;
});

beforeEach(() => {
  vi.stubGlobal("fetch", fetchMock);
  localStorage.clear();
  document.documentElement.className = "";
});
afterEach(() => {
  fetchMock.mockReset();
  vi.unstubAllGlobals();
});

function Probe() {
  const { prefs, set } = usePreferences();
  return (
    <>
      <span data-testid="chat_open">{String(prefs.chat_open)}</span>
      <span data-testid="theme">{String(prefs.theme)}</span>
      <button onClick={() => set({ chat_open: true })}>open</button>
      <button onClick={() => set({ theme: "dark" })}>dark</button>
    </>
  );
}

function renderWith(initial: Preferences) {
  return render(
    <ThemeProvider>
      <PreferencesProvider initial={initial}>
        <Probe />
      </PreferencesProvider>
    </ThemeProvider>,
  );
}

function deferred() {
  let resolve!: (r: Response) => void;
  const promise = new Promise<Response>((r) => (resolve = r));
  return { promise, resolve };
}

describe("preferences", () => {
  it("preferences_optimistic_then_rollback_on_failure", async () => {
    const pending = deferred();
    fetchMock.mockReturnValue(pending.promise);
    renderWith({ chat_open: false });

    act(() => screen.getByRole("button", { name: "open" }).click());
    expect(screen.getByTestId("chat_open")).toHaveTextContent("true");
    expect(fetchMock).toHaveBeenCalledWith("/api/rooster/me/preferences", {
      method: "PATCH",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ chat_open: true }),
    });

    await act(async () => pending.resolve(new Response("", { status: 500 })));
    expect(screen.getByTestId("chat_open")).toHaveTextContent("false");
  });

  it("keeps_optimistic_value_on_success", async () => {
    fetchMock.mockResolvedValue(Response.json({ chat_open: true }));
    renderWith({ chat_open: false });
    await act(async () => screen.getByRole("button", { name: "open" }).click());
    expect(screen.getByTestId("chat_open")).toHaveTextContent("true");
  });

  it("theme_rollback_restores_previous_theme", async () => {
    const pending = deferred();
    fetchMock.mockReturnValue(pending.promise);
    renderWith({ theme: "light" });
    act(() => screen.getByRole("button", { name: "dark" }).click());
    expect(document.documentElement).toHaveClass("dark");
    await act(async () => pending.resolve(new Response("", { status: 500 })));
    expect(screen.getByTestId("theme")).toHaveTextContent("light");
    expect(document.documentElement).not.toHaveClass("dark");
  });

  it("stored_theme_applied_on_load", () => {
    renderWith({ theme: "dark" });
    expect(document.documentElement).toHaveClass("dark");
  });

  it("no_stored_theme_keeps_system", () => {
    renderWith({});
    expect(document.documentElement).not.toHaveClass("dark");
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
