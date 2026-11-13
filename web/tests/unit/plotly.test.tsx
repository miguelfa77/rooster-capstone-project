import { act, render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

const plotly = vi.hoisted(() => ({ newPlot: vi.fn(async () => {}), purge: vi.fn() }));
vi.mock("plotly.js-dist-min", () => ({ default: plotly }));
const theme = vi.hoisted(() => ({ resolved: "light" }));
vi.mock("next-themes", () => ({ useTheme: () => ({ resolvedTheme: theme.resolved }) }));

import PlotlyFigure from "@/components/primitives/PlotlyFigure";

const DATA = [{ type: "bar" as const, x: ["a"], y: [1] }];

describe("PlotlyFigure", () => {
  it("plotly_rerenders_when_theme_changes", async () => {
    const { rerender } = render(<PlotlyFigure data={DATA} />);
    await act(async () => {});
    expect(plotly.newPlot).toHaveBeenCalledTimes(1);

    theme.resolved = "dark";
    rerender(<PlotlyFigure data={DATA} />);
    await act(async () => {});
    expect(plotly.newPlot).toHaveBeenCalledTimes(2);
    expect(plotly.purge).toHaveBeenCalledTimes(1);
  });

  it("plotly_waits_for_resolved_theme", async () => {
    plotly.newPlot.mockClear();
    theme.resolved = undefined as unknown as string;
    render(<PlotlyFigure data={DATA} />);
    await act(async () => {});
    expect(plotly.newPlot).not.toHaveBeenCalled();
    theme.resolved = "light";
  });
});
