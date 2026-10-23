// Screenshots of a running viewer, for the journal and the README.
//
//   uv run aircanopy view sites/hku --port 4180 --no-browser &
//   node ui/scripts/screenshot.mjs http://127.0.0.1:4180/ docs/images/hku
//
// writes <prefix>_view.jpg (the whole site), <prefix>_view_building.jpg (the tallest building
// selected) and <prefix>_view_tree.jpg (the tallest tree selected, buildings hidden).
//
//   node ui/scripts/screenshot.mjs http://127.0.0.1:4180/ docs/images/hku --focus ID NAME
//
// instead zooms towards one feature (its id in site.json), selects it and writes only
// <prefix>_view_<NAME>.jpg.

import { chromium } from "@playwright/test";

const [url, prefix, option, focusId, focusName] = process.argv.slice(2);
if (!url || !prefix || (option && (option !== "--focus" || !focusId || !focusName))) {
  console.error("usage: node ui/scripts/screenshot.mjs URL OUTPUT_PREFIX [--focus ID NAME]");
  process.exit(2);
}

const browser = await chromium.launch({
  args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"],
});
const page = await browser.newPage({ viewport: { width: 1400, height: 880 } });
await page.goto(url);
await page.waitForFunction(() => document.body.dataset.ready === "true", null, {
  timeout: 120000,
});
const shot = async (name) => {
  await page.waitForTimeout(800); // let the camera settle and the frame draw
  await page.screenshot({ path: `${prefix}_${name}.jpg`, type: "jpeg", quality: 82 });
  console.log(`wrote ${prefix}_${name}.jpg`);
};
const site = await (await page.request.get(new URL("site/site.json", url).href)).json();
const tallest = (items) =>
  items.filter((f) => f.height !== null).sort((a, b) => b.height - a.height)[0];
const positionOf = async (id) => {
  const pos = await page.evaluate((i) => window.aircanopy.screenPositionOf(i), id);
  if (!pos) throw new Error(`${id} is not on screen`);
  return pos;
};
const click = async (id) => {
  const pos = await positionOf(id);
  await page.mouse.move(pos.x, pos.y);
  await page.mouse.click(pos.x, pos.y);
};

if (focusId) {
  // Zoom in a few steps, aiming at the feature again after each (the view zooms towards
  // the pointer, and the feature drifts a little on screen as it does).
  for (let step = 0; step < 6; step++) {
    const pos = await positionOf(focusId);
    await page.mouse.move(pos.x, pos.y);
    await page.mouse.wheel(0, -300);
    await page.waitForTimeout(300);
  }
  await click(focusId);
  await shot(`view_${focusName}`);
} else {
  await shot("view");
  await click(tallest(site.buildings).id);
  await shot("view_building");
  await page.keyboard.press("Escape");
  const tree = tallest(site.trees);
  if (tree) {
    const keys = await page
      .locator('[data-layer^="buildings_"]:has(input:checked)')
      .evaluateAll((rows) => rows.map((r) => r.getAttribute("data-layer")));
    for (const key of keys) await page.locator(`[data-layer="${key}"] input`).uncheck();
    await click(tree.id);
    await shot("view_tree");
  }
}
await browser.close();
