"""The local server behind `aircanopy view`."""

import json
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager

import pytest

from aircanopy import cli
from aircanopy.view import make_server


@pytest.fixture
def folders(tmp_path):
    ui = tmp_path / "ui"
    (ui / "assets").mkdir(parents=True)
    (ui / "index.html").write_text("<!doctype html><title>viewer</title>")
    (ui / "assets" / "app.js").write_text("console.log(1)")
    site = tmp_path / "sites" / "s"
    site.mkdir(parents=True)
    (site / "summary.json").write_text(json.dumps({"name": "s"}))
    (site / "site.json").write_text("{}")
    (site / "scene.glb").write_bytes(b"glTF\x02\x00\x00\x00")
    (tmp_path / "secret.txt").write_text("not for the browser")
    return ui, site


@contextmanager
def running(ui, site):
    server = make_server(site, ui, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.headers, r.read()


def status(url):
    try:
        return get(url)[0]
    except urllib.error.HTTPError as e:
        return e.code


def test_serves_the_viewer_and_the_site(folders):
    ui, site = folders
    with running(ui, site) as base:
        code, headers, body = get(base + "/")
        assert code == 200 and b"viewer" in body
        assert headers["Content-Type"].startswith("text/html")
        code, headers, body = get(base + "/assets/app.js")
        assert headers["Content-Type"].startswith("text/javascript")
        code, headers, body = get(base + "/site/summary.json")
        assert json.loads(body) == {"name": "s"}
        assert headers["Content-Type"] == "application/json"
        assert headers["Cache-Control"] == "no-store"
        code, headers, body = get(base + "/site/scene.glb")
        assert headers["Content-Type"] == "model/gltf-binary" and body.startswith(b"glTF")


def test_nothing_outside_the_two_folders_and_no_listings(folders):
    ui, site = folders
    with running(ui, site) as base:
        assert status(base + "/site/../../secret.txt") == 404
        assert status(base + "/site/%2e%2e/%2e%2e/secret.txt") == 404
        assert status(base + "/../secret.txt") == 404
        assert status(base + "/site/") == 404
        assert status(base + "/assets/") == 404


def test_refuses_a_folder_that_is_not_a_site(folders, tmp_path):
    ui, _ = folders
    with pytest.raises(FileNotFoundError, match="not a site folder"):
        make_server(tmp_path, ui, port=0)


def test_view_command_explains_a_missing_build(folders, monkeypatch, capsys):
    _, site = folders
    monkeypatch.setattr("aircanopy.view.find_ui", lambda: None)
    assert cli.main(["view", str(site), "--no-browser"]) == 1
    assert "npm run build" in capsys.readouterr().err
