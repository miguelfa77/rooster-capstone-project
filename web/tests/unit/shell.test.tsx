import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { ThemeProvider } from "@/components/theme-provider";
import { AppShell } from "@/components/shell/AppShell";
import { LeftNav } from "@/components/shell/LeftNav";
import { UserMenu } from "@/components/shell/UserMenu";
import { PreferencesProvider } from "@/lib/use-preferences";
import type { UserOut } from "@/lib/types";

const nav = vi.hoisted(() => ({ pathname: "/" }));
vi.mock("next/navigation", () => ({ usePathname: () => nav.pathname }));

const actions = vi.hoisted(() => ({ signOutAction: vi.fn() }));
vi.mock("@/lib/session-actions", () => actions);
vi.mock("sonner", () => ({ toast: { error: vi.fn() } }));

const USER: UserOut = {
  id: "u1",
  email: "maria@example.es",
  name: "María Fernández",
  image: null,
  preferences: {},
};

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
  // Radix popper/menus need these in jsdom.
  globalThis.ResizeObserver ??= class {
    observe() {}
    unobserve() {}
    disconnect() {}
  } as unknown as typeof ResizeObserver;
  Element.prototype.scrollIntoView ??= () => {};
  Element.prototype.hasPointerCapture ??= () => false;
});

beforeEach(() => {
  nav.pathname = "/";
  vi.stubGlobal("fetch", fetchMock);
  localStorage.clear();
  document.documentElement.className = "";
});
afterEach(() => {
  fetchMock.mockReset();
  actions.signOutAction.mockReset();
  vi.unstubAllGlobals();
});

function withProviders(ui: React.ReactNode, initial = USER.preferences) {
  return render(
    <ThemeProvider>
      <PreferencesProvider initial={initial}>{ui}</PreferencesProvider>
    </ThemeProvider>,
  );
}

function openMenu() {
  const trigger = screen.getByRole("button", { name: "Menú de usuario" });
  fireEvent.keyDown(trigger, { key: "Enter" });
  return screen.getByRole("menu");
}

describe("LeftNav", () => {
  it("nav_lists_items_in_order_with_future_ones_disabled", () => {
    render(<LeftNav conversations={<p>slot-conversaciones</p>} />);
    const navEl = screen.getByRole("navigation", { name: "Navegación principal" });
    const links = within(navEl).getAllByRole("link");
    expect(links.map((l) => l.textContent?.replace("Pronto", "").trim())).toEqual([
      "Inicio",
      "Mapa",
      "Barrios",
      "Anuncios",
      "Comparar",
      "Calculadora",
      "Seguimiento",
    ]);
    expect(links[0]).toHaveAttribute("href", "/");
    expect(links[0]).not.toHaveAttribute("aria-disabled");
    for (const link of links.slice(1)) {
      expect(link).toHaveAttribute("aria-disabled", "true");
      expect(link).not.toHaveAttribute("href");
      expect(link).not.toHaveAttribute("tabindex");
      expect(link).toHaveTextContent("Pronto");
      expect(link).toHaveAccessibleDescription("No disponible todavía");
    }
    expect(screen.getByRole("heading", { name: "Conversaciones" })).toBeInTheDocument();
    expect(screen.getByText("slot-conversaciones")).toBeInTheDocument();
    expect(screen.queryByText("Ajustes")).not.toBeInTheDocument();
  });

  it("active_item_marked_aria_current", () => {
    const { unmount } = render(<LeftNav />);
    expect(screen.getByRole("link", { name: "Inicio" })).toHaveAttribute("aria-current", "page");
    unmount();
    nav.pathname = "/chat";
    render(<LeftNav />);
    expect(screen.getByRole("link", { name: "Inicio" })).not.toHaveAttribute("aria-current");
  });
});

describe("UserMenu", () => {
  it("user_menu_shows_name_email_and_signs_out", () => {
    withProviders(<UserMenu user={USER} />);
    const trigger = screen.getByRole("button", { name: "Menú de usuario" });
    expect(trigger).toHaveTextContent("MF");
    // Dark fill with light text in light mode; a subtle raised fill in dark mode (mockup .L/.B).
    expect(within(trigger).getByText("MF")).toHaveClass("bg-foreground", "text-background", "dark:bg-input", "dark:text-foreground");
    const menu = openMenu();
    expect(within(menu).getByText("María Fernández")).toBeInTheDocument();
    expect(within(menu).getByText("maria@example.es")).toBeInTheDocument();
    for (const label of ["Claro", "Oscuro", "Sistema"]) {
      expect(within(menu).getByRole("menuitemradio", { name: label })).toBeInTheDocument();
    }
    act(() => within(menu).getByRole("menuitem", { name: "Cerrar sesión" }).click());
    expect(actions.signOutAction).toHaveBeenCalledTimes(1);
  });

  it("theme_choice_patches_preferences_and_sets_theme", async () => {
    fetchMock.mockResolvedValue(Response.json({ theme: "dark" }));
    withProviders(<UserMenu user={USER} />);
    const menu = openMenu();
    expect(within(menu).getByRole("menuitemradio", { name: "Claro" })).toHaveAttribute("aria-checked", "true");
    await act(async () => within(menu).getByRole("menuitemradio", { name: "Oscuro" }).click());
    expect(fetchMock).toHaveBeenCalledWith("/api/rooster/me/preferences", {
      method: "PATCH",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ theme: "dark" }),
    });
    expect(document.documentElement).toHaveClass("dark");
  });
});

describe("AppShell", () => {
  it("mobile_nav_opens_in_sheet", () => {
    withProviders(
      <AppShell user={USER}>
        <p>contenido</p>
      </AppShell>,
    );
    expect(screen.getByText("contenido")).toBeInTheDocument();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    act(() => screen.getByRole("button", { name: "Abrir menú" }).click());
    const dialog = screen.getByRole("dialog");
    // Focus goes to the sheet itself, not the close button (no ring on pointer opens).
    expect(dialog).toHaveFocus();
    expect(within(dialog).getByRole("link", { name: "Inicio" })).toHaveAttribute("href", "/");
    act(() => within(dialog).getByRole("link", { name: "Inicio" }).click());
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("top_bar_has_disabled_search_and_chat_slot", () => {
    withProviders(
      <AppShell user={USER} chat={<p>slot-chat</p>}>
        <p>contenido</p>
      </AppShell>,
    );
    const search = screen.getByRole("searchbox", { name: "Buscar" });
    expect(search).toBeDisabled();
    // Short hint below sm, the full one from sm up (never clipped mid-word).
    const form = search.closest("[role=search]")!;
    expect(within(form as HTMLElement).getByText("Buscar…")).toHaveClass("sm:hidden");
    expect(within(form as HTMLElement).getByText("Buscar barrio, dirección o anuncio…")).toHaveClass("hidden", "sm:inline");
    expect(screen.getByRole("button", { name: "Pregunta a Rooster" })).toBeInTheDocument();
    expect(screen.getByText("slot-chat")).toBeInTheDocument();
  });
});
