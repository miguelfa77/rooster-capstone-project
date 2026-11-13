import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import EntrarPage from "@/app/entrar/page";
import InvitacionPage from "@/app/entrar/invitacion/page";
import { COPY } from "@/lib/copy";

vi.mock("@/auth", () => ({ signIn: vi.fn() }));

afterEach(() => vi.clearAllMocks());

async function renderEntrar(params: { next?: string; error?: string; callbackUrl?: string }) {
  render(await EntrarPage({ searchParams: Promise.resolve(params) }));
}

describe("/entrar", () => {
  it("shows_google_button_and_service_error", async () => {
    await renderEntrar({ error: "servicio", next: "/inteligencia" });
    expect(screen.getByText(COPY.wordmark)).toBeInTheDocument();
    expect(screen.getByText("Tu copiloto inmobiliario para Valencia")).toBeInTheDocument();
    const button = screen.getByRole("button", { name: "Continuar con Google" });
    expect(button).toHaveAttribute("type", "submit");
    expect(screen.getByRole("alert")).toHaveTextContent(COPY.signInServiceError);
    const next = button.closest("form")?.querySelector('input[name="next"]');
    expect(next).toHaveValue("/inteligencia");
  });

  it("no_error_line_without_error_param", async () => {
    await renderEntrar({});
    expect(screen.getByRole("button", { name: "Continuar con Google" })).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("unsafe_next_is_dropped", async () => {
    await renderEntrar({ next: "//evil.test" });
    const next = screen.getByRole("button", { name: "Continuar con Google" }).closest("form")?.querySelector('input[name="next"]');
    expect(next).toHaveValue("/");
  });
});

describe("/entrar/invitacion", () => {
  it("explains_invite_only_and_offers_another_account", () => {
    render(<InvitacionPage />);
    expect(screen.getByRole("heading", { name: "Rooster es solo por invitación" })).toBeInTheDocument();
    expect(screen.getByText(COPY.inviteHowToRequest)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Probar con otra cuenta" })).toBeInTheDocument();
  });
});

describe("/entrar errors and callbackUrl", () => {
  function nextValue() {
    return screen.getByRole("button", { name: "Continuar con Google" }).closest("form")?.querySelector('input[name="next"]');
  }

  it("any_error_value_shows_service_error", async () => {
    for (const error of ["Configuration", "AccessDenied", "servicio"]) {
      await renderEntrar({ error });
      expect(screen.getByRole("alert")).toHaveTextContent(COPY.signInServiceError);
      cleanup();
    }
  });

  it("safe_callback_url_is_fallback_for_next", async () => {
    await renderEntrar({ callbackUrl: "/conversaciones/1" });
    expect(nextValue()).toHaveValue("/conversaciones/1");
    cleanup();
    await renderEntrar({ callbackUrl: "https://evil.test/" });
    expect(nextValue()).toHaveValue("/");
    cleanup();
    await renderEntrar({ next: "/inteligencia", callbackUrl: "/conversaciones/1" });
    expect(nextValue()).toHaveValue("/inteligencia");
  });
});
