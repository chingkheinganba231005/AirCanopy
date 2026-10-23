import { describe, expect, it } from "vitest";
import {
  centreText,
  count,
  heightSourceText,
  metres,
  osmUrl,
  polygonArea,
  recordsById,
  shortText,
  skippedByReason,
} from "../src/describe";
import type { Assumptions, BuildingRecord, SiteJson } from "../src/types";

const ASSUMPTIONS: Assumptions = {
  "A-001": { storey_height_m: 3 },
  "A-002": { radius_m: 100, min_neighbours: 3 },
};

function square(x: number, y: number, size: number): number[][] {
  return [
    [x, y],
    [x + size, y],
    [x + size, y + size],
    [x, y + size],
    [x, y],
  ];
}

function building(id: string, height: number | null, source: string): BuildingRecord {
  return {
    id,
    footprint: { type: "Polygon", coordinates: [square(0, 0, 10)] },
    height,
    base: 0,
    height_source: source,
    base_source: "ground",
    kind: "yes",
    part: false,
  };
}

describe("words for the data", () => {
  it("says where a height came from, with the assumption's value", () => {
    expect(heightSourceText("osm:levels", ASSUMPTIONS)).toContain("3 m each (A-001)");
    expect(heightSourceText("estimated:neighbour-median", ASSUMPTIONS)).toContain("100 m");
    expect(heightSourceText("osm:height", ASSUMPTIONS)).toMatch(/^mapped/);
    expect(heightSourceText("missing", ASSUMPTIONS)).toBe("unknown");
    expect(heightSourceText("something:new", ASSUMPTIONS)).toBe("something:new");
  });

  it("links OpenStreetMap ids, including parts of a relation", () => {
    expect(osmUrl("osm:way/123")).toBe("https://www.openstreetmap.org/way/123");
    expect(osmUrl("osm:relation/5#2")).toBe("https://www.openstreetmap.org/relation/5");
    expect(osmUrl("ground")).toBeNull();
  });

  it("measures footprints with their courtyards taken out", () => {
    const withHole = {
      type: "Polygon" as const,
      coordinates: [square(0, 0, 10), square(4, 4, 2).reverse()],
    };
    expect(polygonArea(withHole)).toBeCloseTo(96);
    expect(polygonArea({ type: "Polygon", coordinates: [] })).toBe(0);
  });

  it("formats numbers and places", () => {
    expect(count(264199)).toBe("264,199");
    expect(metres(null)).toBe("unknown");
    expect(metres(72)).toBe("72.0 m");
    expect(centreText([114.13, 22.28, 114.14, 22.29])).toBe("22.2850° N, 114.1350° E");
    expect(centreText([-0.2, -33.9, -0.1, -33.8])).toBe("33.8500° S, 0.1500° W");
  });
});

describe("records", () => {
  const site: SiteJson = {
    schema: 1,
    name: "t",
    bbox: [0, 0, 1, 1],
    frame: { lon0: 0, lat0: 0, proj4: "" },
    created: "",
    sources: [],
    skipped: [
      { id: "a", reason: "underground" },
      { id: "b", reason: "open roof" },
      { id: "c", reason: "open roof" },
    ],
    buildings: [building("osm:way/1", 21, "osm:levels"), building("osm:way/2", null, "missing")],
    trees: [],
    roads: [],
    footways: [],
  };

  it("finds a record by its id and describes it in one line", () => {
    const records = recordsById(site);
    const known = records.get("osm:way/1");
    const unknown = records.get("osm:way/2");
    expect(known && shortText(known)).toBe("Building, 21 m");
    expect(unknown && shortText(unknown)).toBe("Building, height unknown");
  });

  it("counts the features left out by reason, most common first", () => {
    expect(skippedByReason(site)).toEqual([
      ["open roof", 2],
      ["underground", 1],
    ]);
  });
});
