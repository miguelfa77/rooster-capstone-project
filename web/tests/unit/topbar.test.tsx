import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

const pathname = vi.hoisted(() => ({ value: "/" }));
vi.mock("next/navigation", () => ({ usePathname: () => pathname.value }));

import TopBar from "@/components/TopBar";

describe("TopBar", () => {
  it("shows_wordmark_tagline_and_links", () => {
    render(<TopBar />);
    expect(screen.getByText("Rooster")).toBeInTheDocument();
    expect(screen.getByText("Tu copiloto inmobiliario para Valencia")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Pregunta a Rooster" })).toHaveAttribute("href", "/");
    expect(screen.getByRole("link", { name: "Inteligencia" })).toHaveAttribute("href", "/intel");
  });

  it("marks_active_link", () => {
    pathname.value = "/intel";
    render(<TopBar />);
    expect(screen.getAllByRole("link", { name: "Inteligencia" }).at(-1)).toHaveAttribute("aria-current", "page");
    expect(screen.getAllByRole("link", { name: "Pregunta a Rooster" }).at(-1)).not.toHaveAttribute("aria-current");
  });
});
