import { defineConfig } from "@playwright/test";

// The browser tests run the real thing: Python builds a site from the stored OpenStreetMap,
// elevation and canopy samples (no network), `aircanopy view` serves it with the built viewer, and the tests drive a
// browser. Screenshots go to test-results/screens/; look at them after a change.
const PORT = 4174;

export default defineConfig({
  testDir: "e2e",
  outputDir: "test-results/output",
  timeout: 120_000,
  expect: { timeout: 60_000 },
  reporter: [["list"]],
  use: {
    baseURL: `http://127.0.0.1:${PORT}/`,
    viewport: { width: 1400, height: 880 },
    // No graphics card in CI or in the cloud container: draw WebGL in software.
    launchOptions: {
      args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"],
    },
  },
  webServer: {
    command:
      "uv run aircanopy site build --osm-file ../tests/data/osm/hku_small.osm.gz " +
      "--terrain-file ../tests/data/terrain/hku_small_gedtm30.tif " +
      "--canopy-file ../tests/data/canopy/hku_small_meta_wri_v1.tif " +
      "--name sample --out e2e/.site && " +
      `uv run aircanopy view e2e/.site --port ${PORT} --no-browser`,
    url: `http://127.0.0.1:${PORT}/`,
    env: { AIRCANOPY_UI: "dist" },
    timeout: 180_000,
    reuseExistingServer: false,
  },
});
