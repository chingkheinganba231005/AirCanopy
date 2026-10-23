# Sources

Every library, dataset and reference the project relies on, with its licence and when it was
checked. Anything that could not be checked is marked `UNVERIFIED`.

## Data

| Data | Source | Licence | Checked |
| --- | --- | --- | --- |
| OpenStreetMap (buildings, trees, roads) | OpenStreetMap API v0.6, `map` call | Open Database Licence 1.0; attribution "© OpenStreetMap contributors". The licence URL is stated in every API response (`license="http://opendatacommons.org/licenses/odbl/1-0/"`) | 2026-10-23 |

## References

| What we use | Reference | Checked |
| --- | --- | --- |
| The `map` call accepts at most 0.25 square degrees | `https://api.openstreetmap.org/api/0.6/capabilities`, element `area maximum="0.25"` | 2026-10-23 |
| Meaning of `height`, `min_height`, `building:levels`, `building:min_level`, `roof:levels`, `roof:height`; parts replace the outline in 3D | OpenStreetMap wiki, "Simple 3D Buildings", `https://wiki.openstreetmap.org/wiki/Simple_3D_Buildings` (revision 3106809) | 2026-10-23 |
| Highway classes for vehicles and for walking | OpenStreetMap wiki, "Key:highway", `https://wiki.openstreetmap.org/wiki/Key:highway` | UNVERIFIED (not yet read for this project) |

## Software

Licences from the PyPI metadata of each package.

| Package | Licence | Checked |
| --- | --- | --- |
| numpy | BSD-3-Clause (with bundled 0BSD, MIT, Zlib, CC0-1.0 parts) | 2026-10-23 |
| shapely | BSD-3-Clause | 2026-10-23 |
| pyproj | MIT | 2026-10-23 |
| trimesh | MIT | 2026-10-23 |
| mapbox-earcut | ISC | 2026-10-23 |
| pytest, ruff (development only) | MIT | 2026-10-23 |
| hatchling (build only) | MIT | 2026-10-23 |
