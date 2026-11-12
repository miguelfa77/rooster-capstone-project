import { render, screen } from "@testing-library/react";
import { readFileSync } from "node:fs";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { COPY } from "@/lib/copy";
import type { ChoroplethPayload, PointMapPayload, PrimitivePayload } from "@/lib/types";

// A fake maplibre Map that records what the components ask of it.
const fake = vi.hoisted(() => {
  const state = {
    instances: [] as FakeMap[],
    popups: [] as FakePopup[],
    throwOnCreate: false,
  };
  class FakeMap {
    options: Record<string, unknown>;
    sources: Record<string, unknown> = {};
    layers: { id: string; type: string; paint?: Record<string, unknown> }[] = [];
    handlers: { type: string; layer?: string; cb: (e: unknown) => void }[] = [];
    controls: unknown[] = [];
    fitBounds = vi.fn();
    remove = vi.fn();
    constructor(options: Record<string, unknown>) {
      if (state.throwOnCreate) throw new Error("Failed to initialize WebGL");
      this.options = options;
      state.instances.push(this);
    }
    addControl(c: unknown) {
      this.controls.push(c);
      return this;
    }
    addSource(id: string, spec: unknown) {
      this.sources[id] = spec;
      return this;
    }
    addLayer(layer: { id: string; type: string }) {
      this.layers.push(layer);
      return this;
    }
    getCanvas() {
      return { style: {} as Record<string, string> };
    }
    on(type: string, layerOrCb: string | ((e: unknown) => void), maybeCb?: (e: unknown) => void) {
      if (typeof layerOrCb === "function") {
        if (type === "load") layerOrCb({});
        else this.handlers.push({ type, cb: layerOrCb });
      } else if (maybeCb) {
        this.handlers.push({ type, layer: layerOrCb, cb: maybeCb });
      }
      return this;
    }
    fire(type: string, layer: string, e: unknown) {
      for (const h of this.handlers) if (h.type === type && h.layer === layer) h.cb(e);
    }
  }
  class FakePopup {
    content: Node | null = null;
    constructor() {
      state.popups.push(this);
    }
    setLngLat() {
      return this;
    }
    setDOMContent(node: Node) {
      this.content = node;
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

vi.mock("maplibre-gl", () => ({
  Map: fake.FakeMap,
  Popup: fake.FakePopup,
  NavigationControl: fake.FakeControl,
}));

import BaseMap from "@/components/map/BaseMap";
import ChoroplethMap from "@/components/map/ChoroplethMap";
import PointMap from "@/components/map/PointMap";

const fixture = JSON.parse(
  readFileSync(path.resolve(__dirname, "../../../tests/fixtures/primitive_payloads.json"), "utf8"),
) as { payloads: PrimitivePayload[] };

function payload<K extends "choropleth" | "point_map">(kind: K) {
  const found = fixture.payloads.find((p) => p.kind === kind);
  if (!found) throw new Error(`fixture has no ${kind}`);
  return structuredClone(found) as Extract<PrimitivePayload, { kind: K }>;
}

function lastMap() {
  const m = fake.state.instances.at(-1);
  if (!m) throw new Error("no map created");
  return m;
}

afterEach(() => {
  fake.state.instances.length = 0;
  fake.state.popups.length = 0;
  fake.state.throwOnCreate = false;
});

describe("maps", () => {
  it("base_map_uses_carto_dark_basemap_centered_on_valencia", () => {
    const onLoad = vi.fn();
    render(<BaseMap onLoad={onLoad} height={300} />);
    const m = lastMap();
    expect(m.options.center).toEqual([-0.3763, 39.4699]);
    expect(m.options.zoom).toBe(12);
    const style = JSON.stringify(m.options.style);
    expect(style).toContain("https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png");
    expect(style).toContain("https://d.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png");
    expect(style).toContain("© OpenStreetMap contributors © CARTO");
    expect(m.controls).toHaveLength(1);
    expect(onLoad).toHaveBeenCalledWith(m);
  });

  it("choropleth_adds_fill_layer_with_metric", () => {
    const p = payload("choropleth");
    render(<ChoroplethMap p={p} />);
    const m = lastMap();
    const fill = m.layers.find((l) => l.type === "fill");
    expect(fill).toBeDefined();
    expect(JSON.stringify(fill?.paint?.["fill-color"])).toContain(p.metric);
    const line = m.layers.find((l) => l.type === "line");
    expect(line?.paint?.["line-color"]).toBe("#333");
    expect(m.fitBounds).toHaveBeenCalledTimes(1);
    expect(screen.getByText(p.metric_label)).toBeInTheDocument();
  });

  it("choropleth_hover_popup_lists_tooltip_fields", () => {
    const p: ChoroplethPayload = {
      ...payload("choropleth"),
      tooltip_fields: [
        { field: "neighborhood_name", label: "Barrio" },
        { field: "gross_rental_yield_pct", label: "Rentabilidad" },
      ],
    };
    render(<ChoroplethMap p={p} />);
    lastMap().fire("mousemove", "choropleth-fill", {
      lngLat: { lng: -0.37, lat: 39.46 },
      features: [{ properties: { neighborhood_name: "<b>Russafa</b>", gross_rental_yield_pct: 6.1 } }],
    });
    const node = fake.state.popups.at(-1)?.content as HTMLElement;
    expect(node.textContent).toContain("Barrio: <b>Russafa</b>");
    expect(node.textContent).toContain("Rentabilidad: 6,10");
    expect(node.querySelector("b")).toBeNull();
  });

  it("point_map_adds_circle_layer", () => {
    const p = payload("point_map");
    render(<PointMap p={p} />);
    const m = lastMap();
    const circle = m.layers.find((l) => l.type === "circle");
    expect(circle).toBeDefined();
    expect(JSON.stringify(circle?.paint?.["circle-color"])).toContain("price");
    expect(m.fitBounds).toHaveBeenCalledTimes(1);
  });

  it("point_map_without_color_field_uses_accent", () => {
    const p: PointMapPayload = { ...payload("point_map"), color_field: null };
    render(<PointMap p={p} />);
    const circle = lastMap().layers.find((l) => l.type === "circle");
    expect(circle?.paint?.["circle-color"]).toBe("#e03030");
  });

  it("point_popup_escapes_values_and_links_safely", () => {
    const p = payload("point_map");
    render(<PointMap p={p} />);
    lastMap().fire("click", "points", {
      lngLat: { lng: -0.37, lat: 39.46 },
      features: [{ properties: { price: "<img src=x onerror=alert(1)>", url: "https://www.idealista.com/inmueble/1/" } }],
    });
    const node = fake.state.popups.at(-1)?.content as HTMLElement;
    expect(node.querySelector("img")).toBeNull();
    expect(node.textContent).toContain("Precio: <img src=x onerror=alert(1)>");
    const link = node.querySelector("a");
    expect(link?.getAttribute("href")).toBe("https://www.idealista.com/inmueble/1/");
    expect(link?.getAttribute("target")).toBe("_blank");
    expect(link?.getAttribute("rel")).toBe("noopener noreferrer");
    expect(link?.textContent).toBe(COPY.popupIdealista);

    lastMap().fire("click", "points", {
      lngLat: { lng: -0.37, lat: 39.46 },
      features: [{ properties: { price: 1, url: "javascript:alert(1)" } }],
    });
    expect((fake.state.popups.at(-1)?.content as HTMLElement).querySelector("a")).toBeNull();
  });

  it("empty_data_shows_empty_state", () => {
    const { unmount } = render(<ChoroplethMap p={{ ...payload("choropleth"), data: [{ geom: null }] }} />);
    expect(screen.getByText(COPY.emptyChart)).toBeInTheDocument();
    unmount();
    render(<PointMap p={{ ...payload("point_map"), data: [] }} />);
    expect(screen.getByText(COPY.emptyChart)).toBeInTheDocument();
    expect(fake.state.instances).toHaveLength(0);
  });

  it("map_removed_on_unmount", () => {
    const { unmount } = render(<ChoroplethMap p={payload("choropleth")} />);
    const m = lastMap();
    expect(m.remove).not.toHaveBeenCalled();
    unmount();
    expect(m.remove).toHaveBeenCalledTimes(1);
  });

  it("webgl_failure_shows_error", () => {
    fake.state.throwOnCreate = true;
    render(<PointMap p={payload("point_map")} />);
    expect(screen.getByText(COPY.mapLoadError)).toBeInTheDocument();
  });
});
