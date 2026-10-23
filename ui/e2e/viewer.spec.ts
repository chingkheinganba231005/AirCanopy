import { expect, type Page, test } from "@playwright/test";
import type { SiteJson, SummaryJson } from "../src/types";

const SCREENS = "test-results/screens";

async function open(page: Page): Promise<string[]> {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });
  await page.goto("./");
  await expect(page.locator("body")).toHaveAttribute("data-ready", "true");
  return errors;
}

test("opens the site and shows what it contains", async ({ page, request }) => {
  const summary = (await (await request.get("site/summary.json")).json()) as SummaryJson;
  const errors = await open(page);
  const panel = page.locator("#panel");
  await expect(panel.locator("h1")).toHaveText(summary.name);
  await expect(panel).toContainText(`${summary.buildings} buildings`);
  await expect(page.locator("#attribution")).toContainText("OpenStreetMap contributors");
  await expect(page.locator("#attribution")).toContainText("GEDTM30 v1.2");
  await expect(panel).toContainText("From the GEDTM30 terrain model");
  // Each layer's count is the number of features drawn in that group.
  const levels = summary.building_heights["osm:levels"] ?? 0;
  await expect(panel.locator('[data-layer="buildings_levels"] .num')).toHaveText(String(levels));
  await page.waitForTimeout(500);
  await page.screenshot({ path: `${SCREENS}/overview.png` });
  expect(errors).toEqual([]);
});

test("clicking a building says where its height came from", async ({ page, request }) => {
  const site = (await (await request.get("site/site.json")).json()) as SiteJson;
  const tallest = site.buildings
    .filter((b) => b.height !== null)
    .sort((a, b) => (b.height ?? 0) - (a.height ?? 0))[0];
  expect(tallest).toBeDefined();
  if (!tallest) return;
  await open(page);
  const pos = await page.evaluate((id) => window.aircanopy?.screenPositionOf(id), tallest.id);
  expect(pos).toBeTruthy();
  if (!pos) return;
  // The whole site is framed, so the tallest roof is on screen and clear of the panel.
  const panel = await page.locator("#panel").boundingBox();
  expect(pos.x).toBeGreaterThan((panel?.x ?? 0) + (panel?.width ?? 0));
  expect(pos.y).toBeGreaterThan(0);
  await page.mouse.click(pos.x, pos.y);
  const info = page.locator("#info");
  await expect(info).toBeVisible();
  await expect(info).toContainText(tallest.id);
  await expect(info).toContainText(`${tallest.height?.toFixed(1)} m`);
  expect(await page.evaluate(() => window.aircanopy?.selected())).toBe(tallest.id);
  await page.waitForTimeout(300);
  await page.screenshot({ path: `${SCREENS}/selected.png` });
  await page.keyboard.press("Escape");
  await expect(info).toBeHidden();
});

test("a layer can be switched off", async ({ page }) => {
  await open(page);
  const box = page.locator('[data-layer="buildings_levels"] input');
  await box.uncheck();
  expect(
    await page.evaluate(
      () => window.aircanopy?.viewer.groups.get("buildings_levels")?.mesh.visible,
    ),
  ).toBe(false);
  await page.waitForTimeout(300);
  await page.screenshot({ path: `${SCREENS}/layer-off.png` });
});
