// The files of a site folder, as written by `aircanopy site build` (src/aircanopy/site/).

export interface GeoJsonPolygon {
  type: "Polygon";
  coordinates: number[][][];
}

export interface GeoJsonLine {
  type: "LineString";
  coordinates: number[][];
}

export interface BuildingRecord {
  id: string;
  footprint: GeoJsonPolygon;
  height: number | null;
  base: number;
  height_source: string;
  base_source: string;
  kind: string;
  part: boolean;
  footprint_source?: string; // "osm", or "overture:<dataset>"
  height_dataset?: string | null; // for a height from Overture: the dataset that gave it
}

export interface TreeRecord {
  id: string;
  x: number;
  y: number;
  height: number | null;
  crown_diameter: number | null;
  trunk_circumference: number | null;
  species: string | null;
  genus: string | null;
  leaf_type: string | null;
  leaf_cycle: string | null;
  height_source: string;
  crown_source: string;
}

export interface RoadRecord {
  id: string;
  line: GeoJsonLine;
  highway: string;
  lanes: number | null;
  width: number | null;
  oneway: boolean;
  tunnel: boolean;
  bridge: boolean;
  lanes_source: string;
  width_source: string;
}

export interface SourceRecord {
  url?: string;
  path?: string;
  accessed?: string;
  bytes: number;
  sha256: string;
  licence: string;
  attribution: string;
  note?: string;
}

export interface SiteJson {
  schema: number;
  name: string;
  bbox: [number, number, number, number];
  frame: { lon0: number; lat0: number; proj4: string };
  created: string;
  sources: SourceRecord[];
  skipped: { id: string; reason: string }[];
  buildings: BuildingRecord[];
  trees: TreeRecord[];
  roads: RoadRecord[];
  footways: RoadRecord[];
}

export interface Assumptions {
  "A-001": { storey_height_m: number };
  "A-002": { radius_m: number; min_neighbours: number };
}

export interface SummaryJson {
  name: string;
  bbox: [number, number, number, number];
  size_m: [number, number];
  buildings: number;
  building_heights: Record<string, number>;
  buildings_without_height: number;
  estimated_from_neighbours: number;
  tallest_building_m: number;
  trees: number;
  trees_with_height: number;
  roads: number;
  footways: number;
  skipped: number;
  voxels: { dx_m: number; shape_zyx: [number, number, number]; building_cells: number };
  assumptions: Assumptions;
  height_kinds?: Record<string, number>;
  footprint_sources?: Record<string, number>;
  overture?: { release?: string; error?: string; added?: number } | null;
  credits?: { dataset: string; credit: string; licence: string; buildings: number }[];
  sources: SourceRecord[];
}

export type FeatureRecord =
  | { kind: "building"; record: BuildingRecord }
  | { kind: "tree"; record: TreeRecord }
  | { kind: "road"; record: RoadRecord }
  | { kind: "footway"; record: RoadRecord };
