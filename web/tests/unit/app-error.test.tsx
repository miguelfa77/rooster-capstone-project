import { act, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import AppError from "@/app/(app)/error";
import RootError from "@/app/error";

describe("error boundary", () => {
  it("shows_spanish_message_and_retries", () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    const retry = vi.fn();
    render(<AppError error={new Error("boom")} retry={retry} />);
    expect(screen.getByRole("alert")).toHaveTextContent("Algo ha fallado");
    act(() => screen.getByRole("button", { name: "Reintentar" }).click());
    expect(retry).toHaveBeenCalledTimes(1);
    expect(RootError).toBe(AppError);
  });
});
