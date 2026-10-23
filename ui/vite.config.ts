import { createReadStream, statSync } from "node:fs";
import { extname, resolve, sep } from "node:path";
import type { Plugin } from "vite";
import { defineConfig } from "vitest/config";

// `npm run dev` serves a site folder at /site/, as `aircanopy view` does. The folder is
// AIRCANOPY_SITE, or ../sites/hku (made by `aircanopy site build --name hku`).
function siteFolder(): Plugin {
  const root = resolve(process.env.AIRCANOPY_SITE ?? "../sites/hku");
  const types: Record<string, string> = {
    ".json": "application/json",
    ".glb": "model/gltf-binary",
  };
  return {
    name: "aircanopy-site-folder",
    configureServer(server) {
      server.middlewares.use("/site", (req, res) => {
        const path = decodeURIComponent((req.url ?? "/").split("?")[0] ?? "/");
        const file = resolve(root, "." + path);
        let isFile = false;
        try {
          isFile = file.startsWith(root + sep) && statSync(file).isFile();
        } catch {
          isFile = false;
        }
        if (!isFile) {
          res.statusCode = 404;
          res.end();
          return;
        }
        res.setHeader("Content-Type", types[extname(file)] ?? "application/octet-stream");
        res.setHeader("Cache-Control", "no-store");
        createReadStream(file).pipe(res);
      });
    },
  };
}

export default defineConfig({
  // Relative asset paths, so the built viewer works from any folder or URL path.
  base: "./",
  plugins: [siteFolder()],
  // three.js alone is about 600 kB; the viewer is opened from this computer, so one file is fine.
  build: { target: "es2022", sourcemap: true, chunkSizeWarningLimit: 1000 },
  test: { include: ["tests/**/*.test.ts"] },
});
