// Opens the site folder served at ./site/ (by `aircanopy view`, or by `npm run dev`).

import "./style.css";
import { recordsById, shortText } from "./describe";
import { type Layer, renderInfo, renderPanel } from "./panel";
import type { SiteJson, SummaryJson } from "./types";
import { type Hit, Viewer } from "./viewer";

declare global {
  interface Window {
    /** For the browser tests and for poking around in the developer console. */
    aircanopy?: {
      viewer: Viewer;
      screenPositionOf(id: string): { x: number; y: number } | null;
      selected(): string | null;
    };
  }
}

// Links for the licences that site sources name. The OpenStreetMap API states both URLs in
// every response (docs/sources.md).
const LICENCE_LINKS: Record<string, string> = {
  "ODbL-1.0": "https://www.openstreetmap.org/copyright",
};

function byId<T extends HTMLElement = HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) throw new Error(`the page has no #${id}`);
  return el as T;
}

async function getJson<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return (await r.json()) as T;
}

function layersOf(viewer: Viewer, summary: SummaryJson): Layer[] {
  const storey = summary.assumptions["A-001"].storey_height_m;
  const defs = [
    { key: "buildings_measured", label: "Measured by lidar", groups: ["buildings_measured"] },
    { key: "buildings_mapped", label: "Height in the map data", groups: ["buildings_mapped"] },
    {
      key: "buildings_levels",
      label: `From storeys, ${storey} m each`,
      groups: ["buildings_levels"],
    },
    { key: "buildings_ml", label: "Machine-learning estimate", groups: ["buildings_ml"] },
    {
      key: "buildings_estimated",
      label: "Estimated from neighbours",
      groups: ["buildings_estimated"],
    },
    { key: "roads", label: "Roads", groups: ["roads"] },
    { key: "footways", label: "Footways and paths", groups: ["footways"] },
    { key: "trees", label: "Trees", groups: ["tree_crowns", "tree_trunks"] },
  ];
  return defs.map((d) => {
    const g = viewer.groups.get(d.groups[0] ?? "");
    const colour = g ? (g.mesh.material as { color?: { getStyle(): string } }).color : undefined;
    return { ...d, drawn: g?.ids.length ?? 0, colour: colour?.getStyle() ?? null };
  });
}

function attributionOf(summary: SummaryJson): HTMLElement[] {
  const seen = new Set<string>();
  const parts: HTMLElement[] = [];
  for (const s of summary.sources) {
    const text = `${s.attribution} (${s.licence})`;
    if (seen.has(text)) continue;
    seen.add(text);
    const link = LICENCE_LINKS[s.licence];
    const el = document.createElement(link ? "a" : "span");
    el.textContent = text;
    if (link) Object.assign(el, { href: link, target: "_blank", rel: "noopener" });
    parts.push(el);
  }
  return parts;
}

async function start(): Promise<void> {
  const canvas = byId<HTMLCanvasElement>("scene");
  const [summary, site] = await Promise.all([
    getJson<SummaryJson>("site/summary.json"),
    getJson<SiteJson>("site/site.json"),
  ]);
  document.title = `${summary.name} · AirCanopy`;

  const viewer = new Viewer(canvas);
  await viewer.load("site/scene.glb");
  const records = recordsById(site);

  const panel = byId("panel");
  renderPanel(panel, summary, site, layersOf(viewer, summary), (layer, visible) => {
    for (const g of layer.groups) viewer.setVisible(g, visible);
  });
  // On a wide screen the panel covers the left of the view; centre the site on the rest.
  const inset = () =>
    viewer.setLeftInset(window.innerWidth > 760 ? panel.getBoundingClientRect().right : 0);
  window.addEventListener("resize", inset);
  inset();
  viewer.frame();

  let selected: Hit | null = null;
  const info = byId("info");
  const select = (hit: Hit | null) => {
    selected = hit;
    viewer.select(hit);
    renderInfo(info, hit ? (records.get(hit.id) ?? null) : null, summary, () => select(null));
  };

  // A click selects; a drag turns the view and selects nothing.
  let down: { x: number; y: number } | null = null;
  canvas.addEventListener("pointerdown", (e) => {
    down = { x: e.clientX, y: e.clientY };
  });
  canvas.addEventListener("pointerup", (e) => {
    if (down && Math.hypot(e.clientX - down.x, e.clientY - down.y) < 5) {
      select(viewer.pick(e.clientX, e.clientY));
    }
    down = null;
  });
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape") select(null);
  });

  // Tooltip under the pointer, at most once per frame.
  const tooltip = byId("tooltip");
  let last: PointerEvent | null = null;
  canvas.addEventListener("pointermove", (e) => {
    if (e.buttons !== 0) {
      tooltip.hidden = true;
      return;
    }
    if (!last) {
      requestAnimationFrame(() => {
        const ev = last;
        last = null;
        if (!ev) return;
        const hit = viewer.pick(ev.clientX, ev.clientY);
        const record = hit ? records.get(hit.id) : undefined;
        tooltip.hidden = !record;
        canvas.style.cursor = record ? "pointer" : "";
        if (record) {
          tooltip.textContent = shortText(record);
          tooltip.style.left = `${ev.clientX}px`;
          tooltip.style.top = `${ev.clientY}px`;
        }
      });
    }
    last = e;
  });
  canvas.addEventListener("pointerleave", () => {
    tooltip.hidden = true;
  });

  const compass = byId("compass");
  viewer.onRender = () => {
    compass.style.transform = `rotate(${viewer.northAngle().toFixed(1)}deg)`;
  };
  viewer.requestRender();

  const attribution = byId("attribution");
  const credits = attributionOf(summary).flatMap((el, i) => (i ? ["; ", el] : [el]));
  attribution.replaceChildren("Map data: ", ...credits);

  window.aircanopy = {
    viewer,
    screenPositionOf: (id) => viewer.screenPositionOf(id),
    selected: () => selected?.id ?? null,
  };
  byId("status").hidden = true;
  document.body.dataset.ready = "true";
}

start().catch((e: unknown) => {
  const status = document.getElementById("status");
  if (status) {
    status.textContent = `Could not open the site: ${e instanceof Error ? e.message : String(e)}`;
    status.classList.add("error");
  }
  document.body.dataset.ready = "error";
  console.error(e);
});
