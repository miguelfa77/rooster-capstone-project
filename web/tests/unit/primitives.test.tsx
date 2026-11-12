import { fireEvent, render, screen, within } from "@testing-library/react";
import { readFileSync } from "node:fs";
import path from "node:path";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { PrimitivePayload } from "@/lib/types";

// Recharts measures its container; give it a fixed size in jsdom.
vi.mock("recharts", async (importOriginal) => {
  const mod = await importOriginal<typeof import("recharts")>();
  return {
    ...mod,
    ResponsiveContainer: ({ children }: { children: ReactNode }) => (
      <mod.ResponsiveContainer width={800} height={400}>
        {children}
      </mod.ResponsiveContainer>
    ),
  };
});

const plotlyProps = vi.hoisted(() => ({ calls: [] as unknown[] }));
vi.mock("@/components/primitives/PlotlyFigure", () => ({
  default: (props: { data: unknown }) => {
    plotlyProps.calls.push(props);
    return <div data-testid="plotly-mock" />;
  },
}));
vi.mock("@/components/map/ChoroplethMap", () => ({ default: () => <div data-testid="choropleth-mock" /> }));
vi.mock("@/components/map/PointMap", () => ({ default: () => <div data-testid="point-map-mock" /> }));

import Primitive from "@/components/primitives/Primitive";

const fixture = JSON.parse(
  readFileSync(path.resolve(__dirname, "../../../tests/fixtures/primitive_payloads.json"), "utf8"),
) as { payloads: PrimitivePayload[]; folium_code: PrimitivePayload };

function payload<K extends PrimitivePayload["kind"]>(kind: K, pred?: (p: PrimitivePayload) => boolean) {
  const found = [...fixture.payloads, fixture.folium_code].find((p) => p.kind === kind && (!pred || pred(p)));
  if (!found) throw new Error(`fixture has no ${kind}`);
  return structuredClone(found) as Extract<PrimitivePayload, { kind: K }>;
}

afterEach(() => {
  plotlyProps.calls.length = 0;
  vi.restoreAllMocks();
});

describe("Primitive", () => {
  it("renders_every_fixture_payload_without_throwing", () => {
    for (const p of [...fixture.payloads, fixture.folium_code]) {
      const { unmount } = render(<Primitive p={p} />);
      unmount();
    }
  });

  it("text_renders_markdown_bold", () => {
    const p = { ...payload("text"), content: "Benimaclet **lidera** la rentabilidad." };
    const { container } = render(<Primitive p={p} />);
    expect(container.querySelector("strong")).toHaveTextContent("lidera");
  });

  it("kpi_shows_label_and_value", () => {
    render(<Primitive p={payload("kpi")} />);
    expect(screen.getByText("Rentabilidad media en Valencia")).toBeInTheDocument();
    expect(screen.getByText("5,9")).toBeInTheDocument();
    expect(screen.getByText("+0,3 pp")).toBeInTheDocument();
  });

  it("table_shows_headers_and_sorts_numeric_column", () => {
    render(<Primitive p={payload("table")} />);
    expect(screen.getByText("Barrio")).toBeInTheDocument();
    const firstCell = () => within(screen.getAllByRole("row")[1]).getAllByRole("cell")[0];
    expect(firstCell()).toHaveTextContent("Russafa");
    fireEvent.click(screen.getByRole("button", { name: /Precio mediano de venta/ }));
    expect(firstCell()).toHaveTextContent("Benimaclet");
    fireEvent.click(screen.getByRole("button", { name: /Precio mediano de venta/ }));
    expect(firstCell()).toHaveTextContent("Russafa");
  });

  it("table_formats_currency_and_derives_columns_when_missing", () => {
    render(<Primitive p={payload("table")} />);
    expect(screen.getByText((t) => t.replace(/\s/g, " ") === "285.000 €")).toBeInTheDocument();
    const derived = {
      kind: "table",
      rows_json: JSON.stringify([{ barrio: "Russafa", n: 3 }]),
      columns: [],
    } as PrimitivePayload;
    render(<Primitive p={derived} />);
    expect(screen.getByRole("button", { name: /barrio/ })).toBeInTheDocument();
  });

  it("table_sorts_nulls_last", () => {
    const p = {
      kind: "table",
      rows_json: JSON.stringify([
        { name: "A", v: null },
        { name: "B", v: 2 },
        { name: "C", v: 1 },
      ]),
      columns: [
        { field: "name", header: "Nombre", formatter: "text", alignment: "left", width_hint: "medium" },
        { field: "v", header: "Valor", formatter: "integer", alignment: "right", width_hint: "narrow" },
      ],
    } as PrimitivePayload;
    render(<Primitive p={p} />);
    const names = () =>
      screen
        .getAllByRole("row")
        .slice(1)
        .map((r) => within(r).getAllByRole("cell")[0].textContent);
    fireEvent.click(screen.getByRole("button", { name: /Valor/ }));
    expect(names()).toEqual(["C", "B", "A"]);
    fireEvent.click(screen.getByRole("button", { name: /Valor/ }));
    expect(names()).toEqual(["B", "C", "A"]);
  });

  it("table_sorts_numbers_before_strings_in_mixed_columns", () => {
    const p = {
      kind: "table",
      rows_json: JSON.stringify([{ v: "abc" }, { v: 10 }, { v: "(n/d)" }, { v: "9" }, { v: "Ábaco" }]),
      columns: [{ field: "v", header: "Valor", formatter: "text", alignment: "left", width_hint: "medium" }],
    } as PrimitivePayload;
    render(<Primitive p={p} />);
    const cells = () =>
      screen
        .getAllByRole("row")
        .slice(1)
        .map((r) => within(r).getAllByRole("cell")[0].textContent);
    fireEvent.click(screen.getByRole("button", { name: /Valor/ }));
    expect(cells()).toEqual(["9", "10", "(n/d)", "Ábaco", "abc"]);
    fireEvent.click(screen.getByRole("button", { name: /Valor/ }));
    expect(cells()).toEqual(["abc", "Ábaco", "(n/d)", "10", "9"]);
  });

  it("table_link_column_opens_in_new_tab", () => {
    const p = {
      kind: "table",
      rows_json: JSON.stringify([{ url: "https://www.idealista.com/inmueble/1/" }]),
      columns: [{ field: "url", header: "Anuncio", formatter: "link", alignment: "left", width_hint: "medium" }],
    } as PrimitivePayload;
    render(<Primitive p={p} />);
    const a = screen.getByRole("link", { name: "Ver en Idealista →" });
    expect(a).toHaveAttribute("href", "https://www.idealista.com/inmueble/1/");
    expect(a).toHaveAttribute("target", "_blank");
    expect(a).toHaveAttribute("rel", "noopener noreferrer");
  });

  it("table_link_column_renders_non_http_values_as_text", () => {
    const p = {
      kind: "table",
      rows_json: JSON.stringify([{ url: "javascript:alert(1)" }, { url: "no es una url" }]),
      columns: [{ field: "url", header: "Anuncio", formatter: "link", alignment: "left", width_hint: "medium" }],
    } as PrimitivePayload;
    render(<Primitive p={p} />);
    expect(screen.queryByRole("link")).toBeNull();
    expect(screen.getByText("javascript:alert(1)")).toBeInTheDocument();
    expect(screen.getByText("no es una url")).toBeInTheDocument();
  });

  it("table_without_rows_shows_message", () => {
    render(<Primitive p={{ kind: "table", rows_json: "[]", columns: [] } as PrimitivePayload} />);
    expect(screen.getByText("No hay filas que mostrar.")).toBeInTheDocument();
  });

  it("bar_respects_top_n_sorted_by_value", () => {
    const p = { ...payload("bar"), top_n: 2, data: [...payload("bar").data].reverse() };
    const { container } = render(<Primitive p={p} />);
    const ticks = [
      ...container.querySelectorAll(".recharts-yAxis-tick-labels .recharts-cartesian-axis-tick-value"),
    ].map((t) => t.textContent);
    expect(ticks).toEqual(["Benimaclet", "Russafa"]);
  });

  it("scatter_and_line_render_a_chart", () => {
    const s = render(<Primitive p={payload("scatter")} />);
    expect(s.container.querySelector(".recharts-surface")).not.toBeNull();
    s.unmount();
    const l = render(<Primitive p={payload("line")} />);
    expect(l.container.querySelector(".recharts-line")).not.toBeNull();
    expect(l.container.textContent).toContain("2026-08");
  });

  it("line_groups_color_field_into_series", () => {
    const p = {
      ...payload("line"),
      color_field: "op",
      data: [
        { month: "2026-07", median_alquiler_price: 1, op: "a" },
        { month: "2026-07", median_alquiler_price: 2, op: "b" },
        { month: "2026-08", median_alquiler_price: 3, op: "a" },
      ],
    };
    const { container } = render(<Primitive p={p} />);
    expect(container.querySelectorAll(".recharts-line")).toHaveLength(2);
  });

  it("line_x_axis_keeps_chronological_order_across_series_with_gaps", () => {
    const p = {
      ...payload("line"),
      color_field: "op",
      data: [
        { month: "2026-07", median_alquiler_price: 1, op: "a" },
        { month: "2026-08", median_alquiler_price: 2, op: "b" },
        { month: "2026-09", median_alquiler_price: 3, op: "a" },
      ],
    };
    const { container } = render(<Primitive p={p} />);
    const ticks = [
      ...container.querySelectorAll(".recharts-xAxis-tick-labels .recharts-cartesian-axis-tick-value"),
    ].map((t) => t.textContent);
    expect(ticks).toEqual(["2026-07", "2026-08", "2026-09"]);
  });

  const xTicks = (container: HTMLElement) =>
    [...container.querySelectorAll(".recharts-xAxis-tick-labels .recharts-cartesian-axis-tick-value")].map(
      (t) => t.textContent,
    );

  it.each([
    ["iso dates", ["2026-08", "2026-09", "2026-07", "2026-08"], ["2026-07", "2026-08", "2026-09"]],
    ["numbers", [2022, 2023, 2021, 2022], ["2021", "2022", "2023"]],
    ["other strings keep row order", ["T3", "T4", "T1", "T3"], ["T3", "T4", "T1"]],
  ] as const)("line_x_axis_sorted_when_rows_grouped_by_series (%s)", (_name, xs, expected) => {
    const p = {
      ...payload("line"),
      color_field: "op",
      data: xs.map((x, i) => ({ month: x, median_alquiler_price: i + 1, op: i < 2 ? "a" : "b" })),
    };
    const { container } = render(<Primitive p={p} />);
    expect(xTicks(container)).toEqual(expected);
  });

  it.each(["bar", "scatter", "line"] as const)("empty_data_shows_empty_state (%s)", (kind) => {
    const { unmount } = render(<Primitive p={{ ...payload(kind), data: [] }} />);
    expect(screen.getByText("Sin datos para este gráfico.")).toBeInTheDocument();
    unmount();
    render(<Primitive p={{ ...payload(kind), data: [{ unrelated: 1 }] }} />);
    expect(screen.getByText("Sin datos para este gráfico.")).toBeInTheDocument();
  });

  it("code_plotly_passes_parsed_figure", async () => {
    const p = payload("code", (x) => x.kind === "code" && x.output_type === "plotly_json");
    render(<Primitive p={p} />);
    await screen.findByTestId("plotly-mock");
    const last = plotlyProps.calls.at(-1) as { data: unknown };
    expect(last.data).toEqual(JSON.parse(p.output).data);
  });

  it("code_folium_renders_sandboxed_iframe", () => {
    const p = payload("code", (x) => x.kind === "code" && x.output_type === "folium_html");
    const { container } = render(<Primitive p={p} />);
    const iframe = container.querySelector("iframe");
    expect(iframe).not.toBeNull();
    expect(iframe).toHaveAttribute("sandbox", "allow-scripts");
    expect(iframe?.getAttribute("sandbox")).not.toContain("allow-same-origin");
    expect(iframe?.getAttribute("srcdoc")).toBe(p.output);
    expect(iframe).toHaveAttribute("title");
  });

  it("maps_are_loaded_dynamically", async () => {
    render(<Primitive p={payload("choropleth")} />);
    expect(await screen.findByTestId("choropleth-mock")).toBeInTheDocument();
    render(<Primitive p={payload("point_map")} />);
    expect(await screen.findByTestId("point-map-mock")).toBeInTheDocument();
  });

  it("composite_renders_heading_and_children", () => {
    const { container } = render(<Primitive p={payload("composite")} />);
    expect(screen.getByText("Informe: Benimaclet")).toBeInTheDocument();
    expect(screen.getByText("Rentabilidad media en Valencia")).toBeInTheDocument();
    expect(container.querySelector(".recharts-surface")).not.toBeNull();
  });

  it("unknown_kind_renders_nothing", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const { container } = render(<Primitive p={{ kind: "hologram" } as unknown as PrimitivePayload} />);
    expect(container).toBeEmptyDOMElement();
    expect(warn).toHaveBeenCalledTimes(1);
  });

  it("unknown_kind_warns_once_per_mount_across_rerenders", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const p = { kind: "hologram" } as unknown as PrimitivePayload;
    const { rerender } = render(<Primitive p={p} />);
    rerender(<Primitive p={{ ...p }} />);
    rerender(<Primitive p={{ ...p }} />);
    expect(warn).toHaveBeenCalledTimes(1);
  });
});
