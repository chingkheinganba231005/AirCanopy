// Plain-words descriptions of site data, for the panel and the info card. No three.js here, so
// these are easy to unit-test.

import type { Assumptions, FeatureRecord, GeoJsonPolygon, SiteJson, SummaryJson } from "./types";

const NUMBER = new Intl.NumberFormat("en", { maximumFractionDigits: 0 });

export function count(n: number): string {
  return NUMBER.format(n);
}

export function metres(value: number | null, digits = 1): string {
  return value === null ? "unknown" : `${value.toFixed(digits)} m`;
}

/** Where a building's height came from, in words, with the values of the assumptions used. */
export function heightSourceText(
  source: string,
  a: Assumptions,
  dataset: string | null = null,
): string {
  const from = dataset ?? "dataset not stated";
  switch (source) {
    case "overture:height":
      return `given by Overture Maps, from ${from}`;
    case "overture:num_floors":
      return (
        `estimated from the number of storeys in Overture Maps (${from}), ` +
        `${a["A-001"].storey_height_m} m each (A-001)`
      );
    case "osm:height":
      return "mapped (OpenStreetMap height tag)";
    case "osm:building:height":
      return "mapped (OpenStreetMap building:height tag)";
    case "osm:levels":
      return `estimated from the number of storeys, ${a["A-001"].storey_height_m} m each (A-001)`;
    case "estimated:neighbour-median":
      return (
        `estimated: median height of the buildings within ${a["A-002"].radius_m} m ` +
        `that have one (A-002)`
      );
    case "missing":
      return "unknown";
    default:
      return source;
  }
}

/** The OpenStreetMap page of a feature id such as "osm:way/123" or "osm:relation/5#1". */
export function osmUrl(id: string): string | null {
  const m = /^osm:(node|way|relation)\/(\d+)/.exec(id);
  return m ? `https://www.openstreetmap.org/${m[1]}/${m[2]}` : null;
}

function ringArea(ring: number[][]): number {
  let twice = 0;
  for (let i = 0; i + 1 < ring.length; i++) {
    const [x1 = 0, y1 = 0] = ring[i] ?? [];
    const [x2 = 0, y2 = 0] = ring[i + 1] ?? [];
    twice += x1 * y2 - x2 * y1;
  }
  return Math.abs(twice) / 2;
}

/** Area in square metres of a polygon in local metres: the outer ring minus the holes. */
export function polygonArea(p: GeoJsonPolygon): number {
  const [outer, ...holes] = p.coordinates;
  if (!outer) return 0;
  return holes.reduce((area, hole) => area - ringArea(hole), ringArea(outer));
}

/** "22.2840° N, 114.1370° E" for the centre of a box (west, south, east, north). */
export function centreText(bbox: [number, number, number, number]): string {
  const [w, s, e, n] = bbox;
  const lat = (s + n) / 2;
  const lon = (w + e) / 2;
  const ns = lat >= 0 ? "N" : "S";
  const ew = lon >= 0 ? "E" : "W";
  return `${Math.abs(lat).toFixed(4)}° ${ns}, ${Math.abs(lon).toFixed(4)}° ${ew}`;
}

/** Records by feature id, so a clicked triangle's id leads to its data. */
export function recordsById(site: SiteJson): Map<string, FeatureRecord> {
  const out = new Map<string, FeatureRecord>();
  for (const record of site.buildings) out.set(record.id, { kind: "building", record });
  for (const record of site.trees) out.set(record.id, { kind: "tree", record });
  for (const record of site.roads) out.set(record.id, { kind: "road", record });
  for (const record of site.footways) out.set(record.id, { kind: "footway", record });
  return out;
}

/** One line for the hover tooltip. */
export function shortText(f: FeatureRecord): string {
  switch (f.kind) {
    case "building": {
      const h = f.record.height;
      return `Building, ${h === null ? "height unknown" : `${h.toFixed(0)} m`}`;
    }
    case "tree":
      return `Tree${f.record.genus ? ` (${f.record.genus})` : ""}, ${metres(f.record.height)}`;
    case "road":
      return `Road: ${f.record.highway}`;
    case "footway":
      return `Path: ${f.record.highway}`;
  }
}

/** Reasons features were left out, with how many each. */
export function skippedByReason(site: SiteJson): [string, number][] {
  const counts = new Map<string, number>();
  for (const s of site.skipped) counts.set(s.reason, (counts.get(s.reason) ?? 0) + 1);
  return [...counts.entries()].sort((a, b) => b[1] - a[1]);
}

/** Where a building's outline came from. */
export function footprintSourceText(source: string | undefined): string {
  if (!source || source === "osm") return "OpenStreetMap";
  return source.startsWith("overture:") ? `Overture Maps: ${source.slice(9)}` : source;
}

/** One sentence about the ground: which model, the range of heights, its uncertainty. */
export function groundText(t: SummaryJson["terrain"]): string {
  if (!t || t.model === "flat") return "Level ground: no elevation model was used.";
  const name = t.model === "gedtm30" ? "the GEDTM30 terrain model" : `the ${t.model} model`;
  const range =
    t.ground_min_m !== null && t.ground_max_m !== null
      ? `${t.ground_min_m.toFixed(0)} to ${t.ground_max_m.toFixed(0)} m above sea level`
      : "heights unknown";
  const std = t.std_mean_m !== null ? `, typically within ±${t.std_mean_m.toFixed(1)} m` : "";
  return `From ${name}: ${range}${std}.`;
}
