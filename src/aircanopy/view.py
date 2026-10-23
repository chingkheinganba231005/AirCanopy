"""Serve the 3D viewer and a site folder on this computer.

`aircanopy view sites/hku` starts a small web server and opens the viewer in the browser. The
viewer is a web page (the `ui/` folder, built with `npm run build`) that reads the site folder's
files over HTTP, because browsers do not let a page read files from the disk by itself.

    /            the viewer (the built files of ui/)
    /site/...    the files of the site folder

The server listens on 127.0.0.1 by default, so only this computer can reach it.
"""

from __future__ import annotations

import http.server
import os
import posixpath
import urllib.parse
from pathlib import Path

#: Content types the viewer needs; Python's own table does not know .glb on every system.
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".glb": "model/gltf-binary",
    ".npz": "application/octet-stream",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".wasm": "application/wasm",
}


def find_ui() -> Path | None:
    """The built viewer: $AIRCANOPY_UI, then a copy inside the package, then ui/dist in the repo."""
    candidates = []
    if env := os.environ.get("AIRCANOPY_UI"):
        candidates.append(Path(env))
    here = Path(__file__).resolve().parent
    candidates += [here / "ui", here.parents[1] / "ui" / "dist"]
    return next((c for c in candidates if (c / "index.html").is_file()), None)


def check_site(site_dir: Path) -> None:
    missing = [
        n for n in ("summary.json", "site.json", "scene.glb") if not (site_dir / n).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{site_dir} is not a site folder (missing {', '.join(missing)}); "
            "make one with `aircanopy site build`"
        )


class _Handler(http.server.SimpleHTTPRequestHandler):
    ui_dir: Path
    site_dir: Path
    quiet = True

    def translate_path(self, path: str) -> str:
        # normpath resolves ".." against the root first, so no request can leave the two folders.
        path = posixpath.normpath(urllib.parse.unquote(urllib.parse.urlsplit(path).path))
        if path == "/site" or path.startswith("/site/"):
            root, rel = self.site_dir, path[len("/site") :]
        else:
            root, rel = self.ui_dir, path
        parts = [p for p in rel.split("/") if p not in ("", ".", "..")]
        return str(root.joinpath(*parts))

    def list_directory(self, path):  # no folder listings
        self.send_error(404)
        return None

    def guess_type(self, path) -> str:
        return CONTENT_TYPES.get(Path(path).suffix.lower()) or super().guess_type(path)

    def end_headers(self) -> None:
        # Site files change whenever the site is rebuilt; never let the browser reuse old ones.
        if self.path.startswith("/site/"):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format, *args) -> None:
        if not self.quiet:
            super().log_message(format, *args)


def make_server(
    site_dir: Path, ui_dir: Path, host: str = "127.0.0.1", port: int = 8765, quiet: bool = True
) -> http.server.ThreadingHTTPServer:
    """A server for the viewer and one site folder; port 0 picks a free port."""
    check_site(site_dir)
    handler = type(
        "Handler",
        (_Handler,),
        {"ui_dir": ui_dir.resolve(), "site_dir": site_dir.resolve(), "quiet": quiet},
    )
    return http.server.ThreadingHTTPServer((host, port), handler)
