import { describe, expect, it } from "vitest";
import { boundsOf, colorRamp, extent, rowsToPoints, rowsToPolygons, stepExpression } from "@/components/map/geo";

const square = (x: number, y: number) => ({
  type: "Polygon",
  coordinates: [
    [
      [x, y],
      [x + 0.01, y],
      [x + 0.01, y + 0.01],
      [x, y + 0.01],
      [x, y],
    ],
  ],
});

describe("geo", () => {
  it("polygons_skip_null_geom", () => {
    const fc = rowsToPolygons(
      [
        { neighborhood_name: "Russafa", yield: 6.1, geom: square(-0.376, 39.459) },
        { neighborhood_name: "Sin geom", yield: 5, geom: null },
        { neighborhood_name: "Texto", yield: 5, geom: "not geojson" },
        { neighborhood_name: "Objeto", yield: 5, geom: { foo: 1 } },
        { neighborhood_name: "Cadena", yield: "5.5", geom: square(-0.36, 39.48) },
      ],
      "yield",
    );
    expect(fc.type).toBe("FeatureCollection");
    expect(fc.features).toHaveLength(2);
    expect(fc.features[0].geometry.type).toBe("Polygon");
    expect(fc.features[0].properties).toEqual({ neighborhood_name: "Russafa", yield: 6.1 });
    // numeric strings in the metric column are coerced so the color expression can use them
    expect(fc.features[1].properties?.yield).toBe(5.5);
  });

  it("points_skip_null_coords", () => {
    const fc = rowsToPoints([
      { lat: 39.46, lng: -0.374, price: 1 },
      { lat: null, lng: -0.37 },
      { lat: Number.NaN, lng: -0.37 },
      { lat: 39.47 },
      { lat: "39.48", lng: "-0.36", price: 2 },
    ]);
    expect(fc.features).toHaveLength(2);
    expect(fc.features[0].geometry).toEqual({ type: "Point", coordinates: [-0.374, 39.46] });
    expect(fc.features[0].properties).toEqual({ price: 1 });
    expect(fc.features[1].geometry).toEqual({ type: "Point", coordinates: [-0.36, 39.48] });
  });

  it("points_use_custom_keys", () => {
    const fc = rowsToPoints([{ y: 39.4, x: -0.3 }], "y", "x");
    expect(fc.features[0].geometry).toEqual({ type: "Point", coordinates: [-0.3, 39.4] });
  });

  it("extent_ignores_nulls", () => {
    expect(extent([{ v: 3 }, { v: null }, { v: "x" }, { v: 1 }, { v: "7" }, {}], "v")).toEqual([1, 7]);
    expect(extent([{ v: null }, {}], "v")).toBeNull();
  });

  it("bounds_cover_all_points", () => {
    const pts = rowsToPoints([
      { lat: 39.46, lng: -0.374 },
      { lat: 39.487, lng: -0.36 },
      { lat: 39.47, lng: -0.33 },
    ]);
    expect(boundsOf(pts)).toEqual([
      [-0.374, 39.46],
      [-0.33, 39.487],
    ]);
    const polys = rowsToPolygons([{ geom: square(-0.38, 39.46) }, { geom: square(-0.36, 39.48) }], "v");
    const b = boundsOf(polys);
    expect(b?.[0][0]).toBeCloseTo(-0.38);
    expect(b?.[0][1]).toBeCloseTo(39.46);
    expect(b?.[1][0]).toBeCloseTo(-0.35);
    expect(b?.[1][1]).toBeCloseTo(39.49);
    expect(boundsOf({ type: "FeatureCollection", features: [] })).toBeNull();
  });

  it("each_colormap_has_5_stops", () => {
    for (const name of ["YlOrRd", "YlGn", "Blues", "RdYlGn", "Purples"] as const) {
      const ramp = colorRamp(name);
      expect(ramp).toHaveLength(5);
      for (const c of ramp) expect(c).toMatch(/^#[0-9a-f]{6}$/i);
    }
  });

  it("step_expression_interpolates_between_min_and_max", () => {
    const ramp = colorRamp("Blues");
    expect(stepExpression("v", 0, 4, ramp)).toEqual([
      "interpolate",
      ["linear"],
      ["get", "v"],
      0,
      ramp[0],
      1,
      ramp[1],
      2,
      ramp[2],
      3,
      ramp[3],
      4,
      ramp[4],
    ]);
    // equal min/max still yields strictly ascending stops
    const flat = stepExpression("v", 5, 5, ramp) as unknown[];
    const stops = flat.slice(3).filter((_, i) => i % 2 === 0) as number[];
    for (let i = 1; i < stops.length; i++) expect(stops[i]).toBeGreaterThan(stops[i - 1]);
  });
});
