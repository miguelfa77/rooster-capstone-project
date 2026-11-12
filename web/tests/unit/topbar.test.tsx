import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const pathname = vi.hoisted(() => ({ value: "/" }));
vi.mock("next/navigation", () => ({ usePathname: () => pathname.value }));

import TopBar from "@/components/TopBar";

function links() {
  return {
    chat: screen.getByRole("link", { name: "Pregunta a Rooster" }),
    intel: screen.getByRole("link", { name: "Inteligencia" }),
  };
}

beforeEach(() => {
  pathname.value = "/";
});

describe("TopBar", () => {
  it("shows_wordmark_tagline_and_links", () => {
    render(<TopBar />);
    expect(screen.getByText("Rooster")).toBeInTheDocument();
    expect(screen.getByText("Tu copiloto inmobiliario para Valencia")).toBeInTheDocument();
    expect(links().chat).toHaveAttribute("href", "/");
    expect(links().intel).toHaveAttribute("href", "/intel");
    expect(links().chat).toHaveAttribute("aria-current", "page");
  });

  it("marks_active_link", () => {
    pathname.value = "/intel";
    render(<TopBar />);
    expect(links().intel).toHaveAttribute("aria-current", "page");
    expect(links().chat).not.toHaveAttribute("aria-current");
  });

  it("marks_nested_path_active", () => {
    pathname.value = "/intel/mapa";
    render(<TopBar />);
    expect(links().intel).toHaveAttribute("aria-current", "page");
  });

  it("prefix_lookalike_not_active", () => {
    pathname.value = "/intelligence";
    render(<TopBar />);
    expect(links().intel).not.toHaveAttribute("aria-current");
    expect(links().chat).not.toHaveAttribute("aria-current");
  });
});
