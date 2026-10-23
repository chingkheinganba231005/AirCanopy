// The side panel (what the site contains, layer switches, data sources) and the info card.

import {
  centreText,
  count,
  footprintSourceText,
  heightSourceText,
  metres,
  osmUrl,
  polygonArea,
  skippedByReason,
} from "./describe";
import type { FeatureRecord, SiteJson, SummaryJson } from "./types";

type Child = Node | string | null | undefined | false;

/** The children that are present, for append() and replaceChildren(). */
function present(children: Child[]): (Node | string)[] {
  return children.filter((c): c is Node | string => Boolean(c));
}

/** A small helper to build DOM elements: h("p", { class: "x" }, "text", child). */
export function h(
  tag: string,
  attrs: Record<string, string | boolean> = {},
  ...children: Child[]
): HTMLElement {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === false) continue;
    el.setAttribute(k, v === true ? "" : v);
  }
  el.append(...present(children));
  return el;
}

export interface Layer {
  key: string;
  label: string;
  groups: string[]; // mesh groups in scene.glb
  drawn: number; // features drawn
  colour: string | null; // CSS colour of the swatch
}

export function renderPanel(
  panel: HTMLElement,
  summary: SummaryJson,
  site: SiteJson,
  layers: Layer[],
  onToggle: (layer: Layer, visible: boolean) => void,
): void {
  const [w, hgt] = summary.size_m;
  const layerRow = (layer: Layer) => {
    const box = h("input", { type: "checkbox", checked: layer.drawn > 0 }) as HTMLInputElement;
    box.disabled = layer.drawn === 0;
    box.addEventListener("change", () => onToggle(layer, box.checked));
    const swatch = h("span", { class: "swatch", "aria-hidden": "true" });
    if (layer.colour) swatch.style.background = layer.colour;
    return h(
      "label",
      { class: "layer", "data-layer": layer.key },
      box,
      swatch,
      h("span", { class: "layer-label" }, layer.label),
      h("span", { class: "num" }, count(layer.drawn)),
    );
  };
  const byKey = (k: string) => layers.filter((l) => l.key.startsWith(k));
  const credits = (summary.credits ?? []).map((c) =>
    h(
      "li",
      {},
      c.credit,
      h(
        "span",
        { class: "muted" },
        ` (${c.licence}${c.buildings > 0 ? `; ${count(c.buildings)} buildings` : ""})`,
      ),
    ),
  );
  const overtureError = summary.overture?.error;
  const sources = summary.sources.map((s) =>
    h(
      "li",
      {},
      `${s.attribution}, ${s.licence}`,
      h("br"),
      h("span", { class: "muted" }, s.accessed ? `downloaded ${s.accessed}` : (s.note ?? "")),
    ),
  );
  const skipped = skippedByReason(site);

  panel.replaceChildren(
    ...present([
      h(
        "header",
        {},
        h("p", { class: "eyebrow" }, "AirCanopy · site"),
        h("h1", {}, summary.name),
        h("p", { class: "muted" }, `${centreText(summary.bbox)} · ${count(w)} m × ${count(hgt)} m`),
      ),
      h(
        "section",
        {},
        h("h2", {}, "Buildings"),
        h(
          "p",
          { class: "lead" },
          `${count(summary.buildings)} buildings, the tallest ${metres(summary.tallest_building_m, 0)}.`,
        ),
        h("p", { class: "muted" }, "Colour shows where each height comes from:"),
        ...byKey("buildings").map(layerRow),
        summary.buildings_without_height > 0 &&
          h(
            "p",
            { class: "note" },
            `${count(summary.buildings_without_height)} buildings have no height in the data and too ` +
              "few neighbours to estimate one; they are not drawn and not in the grid.",
          ),
      ),
      h(
        "section",
        {},
        h("h2", {}, "Streets and trees"),
        ...byKey("roads").map(layerRow),
        ...byKey("footways").map(layerRow),
        ...byKey("trees").map(layerRow),
        summary.trees > summary.trees_with_height &&
          h(
            "p",
            { class: "note" },
            `${count(summary.trees)} trees are mapped, ${count(summary.trees_with_height)} with a ` +
              "height. Trees without a height are not drawn yet; tree heights from satellite " +
              "data come in a later step.",
          ),
      ),
      h(
        "section",
        {},
        h("h2", {}, "Grid for the physics"),
        h(
          "p",
          {},
          `Cells of ${metres(summary.voxels.dx_m, 0)}: ` +
            `${summary.voxels.shape_zyx.slice().reverse().map(count).join(" × ")}, ` +
            `${count(summary.voxels.building_cells)} inside buildings.`,
        ),
      ),
      skipped.length > 0 &&
        h(
          "details",
          {},
          h("summary", {}, `${count(site.skipped.length)} features left out`),
          h("ul", {}, ...skipped.map(([reason, n]) => h("li", {}, `${count(n)}: ${reason}`))),
        ),
      h(
        "section",
        { class: "sources" },
        h("h2", {}, "Data"),
        h("ul", {}, ...sources),
        credits.length > 0 && h("p", { class: "muted" }, "Datasets used, with their credits:"),
        credits.length > 0 && h("ul", { class: "credits" }, ...credits),
        overtureError && h("p", { class: "note" }, overtureError),
        h("p", { class: "muted" }, `Site built ${site.created}.`),
        h(
          "p",
          { class: "muted" },
          "The light is for display only. Sun, shade, wind and heat are simulated in later phases.",
        ),
      ),
    ]),
  );
}

function row(label: string, value: Child): HTMLElement {
  return h("div", { class: "row" }, h("dt", {}, label), h("dd", {}, value));
}

export function renderInfo(
  card: HTMLElement,
  feature: FeatureRecord | null,
  summary: SummaryJson,
  onClose: () => void,
): void {
  if (!feature) {
    card.hidden = true;
    card.replaceChildren();
    return;
  }
  const r = feature.record;
  const url = osmUrl(r.id);
  const close = h("button", { class: "close", type: "button", "aria-label": "Close" }, "×");
  close.addEventListener("click", onClose);
  const idLink = url
    ? h("a", { href: url, target: "_blank", rel: "noopener" }, r.id)
    : document.createTextNode(r.id);
  const rows: HTMLElement[] = [];
  let title = "";
  switch (feature.kind) {
    case "building": {
      const b = feature.record;
      title = b.part ? "Building part" : "Building";
      rows.push(
        row("Height", metres(b.height)),
        row(
          "Height from",
          heightSourceText(b.height_source, summary.assumptions, b.height_dataset ?? null),
        ),
        row("Outline from", footprintSourceText(b.footprint_source)),
        row("Base", metres(b.base)),
        row("Footprint", `${count(polygonArea(b.footprint))} m²`),
        row("Type", b.kind),
      );
      break;
    }
    case "tree": {
      const t = feature.record;
      title = "Tree";
      rows.push(
        row("Species", t.species ?? t.genus ?? "unknown"),
        row("Height", metres(t.height)),
        row("Crown width", metres(t.crown_diameter)),
        row("Trunk girth", metres(t.trunk_circumference, 2)),
        row("Leaves", [t.leaf_type, t.leaf_cycle].filter(Boolean).join(", ") || "unknown"),
      );
      break;
    }
    case "road":
    case "footway": {
      const s = feature.record;
      title = feature.kind === "road" ? "Road" : "Path";
      rows.push(
        row("Class", s.highway),
        row("Lanes", s.lanes === null ? "unknown" : String(s.lanes)),
        row("Width", metres(s.width)),
        row("One way", s.oneway ? "yes" : "no"),
      );
      if (s.tunnel || s.bridge) rows.push(row("On", s.tunnel ? "tunnel" : "bridge"));
      break;
    }
  }
  card.replaceChildren(
    h("header", {}, h("h2", {}, title), close),
    h("p", { class: "muted id" }, idLink),
    h("dl", {}, ...rows),
  );
  card.hidden = false;
}
