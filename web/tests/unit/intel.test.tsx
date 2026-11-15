import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { COPY } from "@/lib/copy";

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

// A fake maplibre Map that keeps sources/layers by id, like the real one.
const fake = vi.hoisted(() => {
  const state = { instances: [] as FakeMap[] };
  class FakeSource {
    constructor(public spec: { data: unknown }) {}
    setData(data: unknown) {
      this.spec.data = data;
      return this;
    }
  }
  class FakeMap {
    sources: Record<string, FakeSource> = {};
    layers: { id: string; type: string; source: string; paint?: Record<string, unknown>; layout?: Record<string, unknown> }[] = [];
    handlers: { type: string; layer?: string; cb: (e: unknown) => void }[] = [];
    fitBounds = vi.fn();
    remove = vi.fn();
    constructor() {
      state.instances.push(this);
    }
    addControl() {
      return this;
    }
    addSource(id: string, spec: { data: unknown }) {
      if (this.sources[id]) throw new Error(`source ${id} exists`);
      this.sources[id] = new FakeSource(spec);
      return this;
    }
    getSource(id: string) {
      return this.sources[id];
    }
    addLayer(layer: FakeMap["layers"][number]) {
      if (this.getLayer(layer.id)) throw new Error(`layer ${layer.id} exists`);
      this.layers.push({ ...layer, layout: { ...layer.layout } });
      return this;
    }
    getLayer(id: string) {
      return this.layers.find((l) => l.id === id);
    }
    setLayoutProperty(id: string, name: string, value: unknown) {
      this.getLayer(id)!.layout![name] = value;
    }
    setPaintProperty(id: string, name: string, value: unknown) {
      const l = this.getLayer(id)!;
      l.paint = { ...l.paint, [name]: value };
    }
    visible(id: string) {
      const l = this.getLayer(id);
      return !!l && l.layout?.visibility !== "none";
    }
    queryRenderedFeatures() {
      return [];
    }
    getCanvas() {
      return { style: {} as Record<string, string> };
    }
    on(type: string, layerOrCb: string | ((e: unknown) => void), maybeCb?: (e: unknown) => void) {
      if (typeof layerOrCb === "function") {
        if (type === "load") layerOrCb({});
        else this.handlers.push({ type, cb: layerOrCb });
      } else if (maybeCb) this.handlers.push({ type, layer: layerOrCb, cb: maybeCb });
      return this;
    }
  }
  class FakePopup {
    setLngLat() {
      return this;
    }
    setDOMContent() {
      return this;
    }
    addTo() {
      return this;
    }
    remove() {
      return this;
    }
  }
  class FakeControl {}
  return { state, FakeMap, FakePopup, FakeControl };
});

// BaseMap waits for a resolved theme before creating the map.
vi.mock("next-themes", () => ({ useTheme: () => ({ resolvedTheme: "light" }) }));

vi.mock("maplibre-gl", () => ({
  Map: fake.FakeMap,
  Popup: fake.FakePopup,
  NavigationControl: fake.FakeControl,
}));

import IntelView from "@/components/intel/IntelView";

const POLY = { type: "Polygon", coordinates: [[[-0.38, 39.46], [-0.37, 39.46], [-0.37, 39.47], [-0.38, 39.46]]] };

function neighborhoods(value: number) {
  return {
    geojson: {
      type: "FeatureCollection",
      features: [{ type: "Feature", geometry: POLY, properties: { name: "Russafa", value, listings: 10, yield_pct: 6.1 } }],
    },
    vmin: 0,
    vmax: value,
  };
}

const RESPONSES: Record<string, unknown> = {
  "/api/rooster/intel/framing": { city_median_yield: 5.43, spread_pp: 3.21, n_strong: 7, n_with_yield: 40 },
  "/api/rooster/intel/ranking": [
    { neighborhood_name: "Russafa", yield_pct: 5.1, eur_per_sqm: 3000, score: 6.2 },
    { neighborhood_name: "Benimaclet", yield_pct: 7.4, eur_per_sqm: 2100, score: 8.1 },
    { neighborhood_name: "Ciutat Vella", yield_pct: 4.2, eur_per_sqm: 3900, score: 5.0 },
  ],
  "/api/rooster/intel/yield-liquidity": [
    { neighborhood_name: "Russafa", yield_pct: 5.1, listing_count: 40, eur_per_sqm: 3000 },
    { neighborhood_name: "Benimaclet", yield_pct: 7.4, listing_count: 12, eur_per_sqm: 2100 },
  ],
  "/api/rooster/intel/overlays/transit": [{ lat: 39.47, lng: -0.37, name: "Xàtiva", stop_type: "station" }],
  "/api/rooster/intel/overlays/tourism": [{ lat: 39.47, lng: -0.37, address: "C/ Sueca 1" }],
  "/api/rooster/intel/listings?operation=All": [
    { lat: 39.47, lng: -0.37, operation: "venta", price_int: 250000, eur_per_sqm: 3000, url: "https://x/1" },
  ],
  "/api/rooster/intel/parcels": { type: "FeatureCollection", features: [] },
  "/api/rooster/intel/neighborhoods?metric=investment_score": neighborhoods(8),
  "/api/rooster/intel/neighborhoods?metric=gross_rental_yield_pct": neighborhoods(7),
};

let overrides: Record<string, () => Response | Promise<Response>> = {};
const fetchMock = vi.fn(async (url: string) => {
  if (overrides[url]) return overrides[url]();
  if (url in RESPONSES) return Response.json(RESPONSES[url]);
  return new Response("not found", { status: 404 });
});

function calls(url: string) {
  return fetchMock.mock.calls.filter(([u]) => u === url).length;
}

function lastMap() {
  const m = fake.state.instances.at(-1);
  if (!m) throw new Error("no map created");
  return m;
}

function mapSection() {
  return screen.getByRole("region", { name: COPY.sectionMapa });
}

beforeEach(() => {
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
  fetchMock.mockClear();
  overrides = {};
  fake.state.instances.length = 0;
});

async function renderIntel() {
  render(<IntelView />);
  // Let the per-section fetches settle.
  await screen.findByText("Benimaclet");
}

describe("intel", () => {
  it("framing_cards_show_three_values", async () => {
    await renderIntel();
    const s = screen.getByRole("region", { name: COPY.sectionMercado });
    expect(within(s).getByText(COPY.kpiLabelMedian)).toBeInTheDocument();
    expect(within(s).getByText(/5,4\s%/)).toBeInTheDocument();
    expect(within(s).getByText(COPY.kpiLabelSpread)).toBeInTheDocument();
    expect(within(s).getByText("3,2 pp")).toBeInTheDocument();
    expect(within(s).getByText(COPY.kpiLabelStrong)).toBeInTheDocument();
    expect(within(s).getByText("7")).toBeInTheDocument();
    expect(within(s).getByText("de 40 analizados")).toBeInTheDocument();
    expect(within(s).getByText(COPY.kpiSubYield)).toBeInTheDocument();
    expect(within(s).getByText(COPY.kpiSubDispersion)).toBeInTheDocument();
  });

  it("framing_null_shows_empty_copy", async () => {
    overrides["/api/rooster/intel/framing"] = () => Response.json(null);
    await renderIntel();
    expect(await screen.findByText(COPY.intelNoFraming)).toBeInTheDocument();
  });

  it("layers_off_initially_no_overlay_fetch", async () => {
    await renderIntel();
    const s = mapSection();
    for (const label of [COPY.toggleBarrios, COPY.toggleTransport, COPY.toggleVut, COPY.toggleListings, COPY.toggleParcels]) {
      expect(within(s).getByRole("checkbox", { name: label })).not.toBeChecked();
    }
    const urls = fetchMock.mock.calls.map(([u]) => u);
    expect(urls.some((u) => /neighborhoods|overlays|listings|parcels/.test(u))).toBe(false);
    expect(lastMap().layers).toHaveLength(0);
  });

  it("toggle_fetches_once_and_adds_layer", async () => {
    await renderIntel();
    const box = within(mapSection()).getByRole("checkbox", { name: COPY.toggleTransport });
    fireEvent.click(box);
    await waitFor(() => expect(lastMap().visible("intel-transit")).toBe(true));
    const layer = lastMap().getLayer("intel-transit")!;
    expect(layer.type).toBe("circle");
    expect(JSON.stringify(layer.paint)).toContain("#534AB7");
    fireEvent.click(box);
    expect(lastMap().visible("intel-transit")).toBe(false);
    fireEvent.click(box);
    await waitFor(() => expect(lastMap().visible("intel-transit")).toBe(true));
    expect(calls("/api/rooster/intel/overlays/transit")).toBe(1);

    fireEvent.click(within(mapSection()).getByRole("checkbox", { name: COPY.toggleListings }));
    await waitFor(() => expect(lastMap().visible("intel-listings")).toBe(true));
    expect(calls("/api/rooster/intel/listings?operation=All")).toBe(1);
    expect(JSON.stringify(lastMap().getLayer("intel-listings")!.paint)).toContain("#E95D4F");
    expect(screen.getByText(COPY.intelListingsOpLegend)).toBeInTheDocument();

    fireEvent.click(within(mapSection()).getByRole("checkbox", { name: COPY.toggleParcels }));
    await waitFor(() => expect(lastMap().visible("intel-parcels")).toBe(true));
    expect(lastMap().getLayer("intel-parcels")!.type).toBe("line");
  });

  it("metric_change_refetches_barrios", async () => {
    await renderIntel();
    const s = mapSection();
    fireEvent.click(within(s).getByRole("checkbox", { name: COPY.toggleBarrios }));
    await waitFor(() => expect(lastMap().visible("intel-barrios-fill")).toBe(true));
    expect(calls("/api/rooster/intel/neighborhoods?metric=investment_score")).toBe(1);

    fireEvent.change(within(s).getByLabelText(COPY.intelMapColorBy), { target: { value: "gross_rental_yield_pct" } });
    await waitFor(() => expect(calls("/api/rooster/intel/neighborhoods?metric=gross_rental_yield_pct")).toBe(1));
    await waitFor(() =>
      expect(JSON.stringify(lastMap().getSource("intel-barrios").spec.data)).toContain('"value":7'),
    );
  });

  it("failed_layer_shows_error_and_unchecks", async () => {
    overrides["/api/rooster/intel/neighborhoods?metric=investment_score"] = () =>
      Response.json({ detail: "metric needs neighborhood_profile" }, { status: 422 });
    await renderIntel();
    const box = within(mapSection()).getByRole("checkbox", { name: COPY.toggleBarrios });
    fireEvent.click(box);
    const alert = await within(mapSection()).findByRole("alert");
    expect(alert).toHaveTextContent(COPY.mapLoadError);
    expect(alert).toHaveTextContent(COPY.toggleBarrios);
    expect(box).not.toBeChecked();
    expect(lastMap().getLayer("intel-barrios-fill")).toBeUndefined();
  });

  it("failed_layer_after_toggle_off_shows_no_error", async () => {
    let fail: () => void = () => {};
    const gate = new Promise<void>((r) => (fail = r));
    overrides["/api/rooster/intel/overlays/tourism"] = async () => {
      await gate;
      return new Response("boom", { status: 500 });
    };
    await renderIntel();
    const box = within(mapSection()).getByRole("checkbox", { name: COPY.toggleVut });
    fireEvent.click(box);
    fireEvent.click(box);
    await act(async () => fail());
    expect(within(mapSection()).queryByRole("alert")).toBeNull();
    expect(box).not.toBeChecked();
    // A later attempt fetches again rather than reusing the failed request.
    fireEvent.click(box);
    await waitFor(() => expect(calls("/api/rooster/intel/overlays/tourism")).toBe(2));
  });

  it("stale_metric_response_is_ignored", async () => {
    let releaseA: () => void = () => {};
    const gateA = new Promise<void>((r) => (releaseA = r));
    overrides["/api/rooster/intel/neighborhoods?metric=investment_score"] = async () => {
      await gateA;
      return Response.json(RESPONSES["/api/rooster/intel/neighborhoods?metric=investment_score"]);
    };
    await renderIntel();
    const s = mapSection();
    fireEvent.click(within(s).getByRole("checkbox", { name: COPY.toggleBarrios }));
    fireEvent.change(within(s).getByLabelText(COPY.intelMapColorBy), { target: { value: "gross_rental_yield_pct" } });
    await waitFor(() => expect(lastMap().visible("intel-barrios-fill")).toBe(true));
    await act(async () => releaseA());
    await act(async () => {
      await new Promise((r) => setTimeout(r, 0));
    });
    const source = JSON.stringify(lastMap().getSource("intel-barrios").spec.data);
    expect(source).toContain('"value":7');
    expect(source).not.toContain('"value":8');
  });

  it("transit_stop_position_is_green", async () => {
    await renderIntel();
    fireEvent.click(within(mapSection()).getByRole("checkbox", { name: COPY.toggleTransport }));
    await waitFor(() => expect(lastMap().visible("intel-transit")).toBe(true));
    const color = lastMap().getLayer("intel-transit")!.paint!["circle-color"] as unknown[];
    expect(color[0]).toBe("match");
    const arms = color.slice(2, -1);
    const colorFor = (kind: string) => {
      for (let i = 0; i < arms.length; i += 2) {
        const label = arms[i];
        if (label === kind || (Array.isArray(label) && label.includes(kind))) return arms[i + 1];
      }
      return color.at(-1);
    };
    expect(colorFor("stop_position")).toBe("#1D9E75");
    expect(colorFor("station")).toBe("#534AB7");
    expect(colorFor("tram_stop")).toBe("#534AB7");
    expect(colorFor("bus_stop")).toBe("#888780");
  });

  it("color_by_select_only_while_barrios_on", async () => {
    await renderIntel();
    const s = mapSection();
    expect(within(s).queryByLabelText(COPY.intelMapColorBy)).toBeNull();
    const box = within(s).getByRole("checkbox", { name: COPY.toggleBarrios });
    fireEvent.click(box);
    expect(within(s).getByLabelText(COPY.intelMapColorBy)).toBeInTheDocument();
    fireEvent.click(box);
    expect(within(s).queryByLabelText(COPY.intelMapColorBy)).toBeNull();
  });

  it("rank_caption_does_not_mention_bars", () => {
    expect(COPY.intelRankCaption).toMatch(/puntuación de inversión/);
    expect(COPY.intelRankCaption).not.toMatch(/barra/i);
  });

  it("ranking_sorts", async () => {
    await renderIntel();
    const s = screen.getByRole("region", { name: COPY.intelS3Title });
    const names = () => within(s).getAllByRole("row").slice(1).map((r) => r.querySelector("td")?.textContent);
    expect(names()).toEqual(["Russafa", "Benimaclet", "Ciutat Vella"]);
    fireEvent.click(within(s).getByRole("button", { name: /Rent\. %/ }));
    expect(names()).toEqual(["Ciutat Vella", "Russafa", "Benimaclet"]);
    expect(within(s).getByText(COPY.intelRankCaption)).toBeInTheDocument();
  });

  it("ranking_empty_shows_copy", async () => {
    overrides["/api/rooster/intel/ranking"] = () => Response.json([]);
    render(<IntelView />);
    expect(await screen.findByText(COPY.intelNoRanking)).toBeInTheDocument();
  });

  it("scatter_renders_points", async () => {
    await renderIntel();
    const s = screen.getByRole("region", { name: COPY.intelS4Title });
    await waitFor(() => expect(s.querySelector(".recharts-surface")).not.toBeNull());
    expect(within(s).getByText(COPY.intelScatterCaption)).toBeInTheDocument();
  });

  it("scatter_empty_copy", async () => {
    overrides["/api/rooster/intel/yield-liquidity"] = () => Response.json([]);
    await renderIntel();
    expect(await screen.findByText(COPY.intelNoScatter)).toBeInTheDocument();
  });

  it("section_failure_isolated", async () => {
    overrides["/api/rooster/intel/framing"] = () => new Response("boom", { status: 500 });
    await renderIntel();
    const s = screen.getByRole("region", { name: COPY.sectionMercado });
    expect(await within(s).findByText(COPY.intelLoadError)).toBeInTheDocument();
    expect(screen.getByRole("region", { name: COPY.intelS3Title })).toHaveTextContent("Benimaclet");
    expect(screen.getByRole("region", { name: COPY.intelS4Title })).toBeInTheDocument();
    expect(mapSection()).toBeInTheDocument();
  });

  it("sections_show_loading_first", async () => {
    let release: () => void = () => {};
    const gate = new Promise<void>((r) => (release = r));
    fetchMock.mockImplementationOnce(async () => {
      await gate;
      return Response.json(RESPONSES["/api/rooster/intel/framing"]);
    });
    render(<IntelView />);
    const s = screen.getByRole("region", { name: COPY.sectionMercado });
    expect(within(s).getByText(COPY.intelLoading)).toBeInTheDocument();
    await act(async () => release());
    await within(s).findByText("3,2 pp");
  });
});
