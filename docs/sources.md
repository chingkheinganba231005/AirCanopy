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
| `location=underground` marks a feature that is "mainly under ground"; such buildings are left out | OpenStreetMap wiki, "Key:location", `https://wiki.openstreetmap.org/wiki/Key:location` (revision 3100870) | 2026-10-23 |
| `building:part=roof` is "A permanent roof structure without vertical walls that forms part of a building" | OpenStreetMap wiki, "Tag:building:part=roof", `https://wiki.openstreetmap.org/wiki/Tag:building:part%3Droof` (revision 3058073) | 2026-10-23 |
| glTF axes and units: "glTF uses a right-handed coordinate system", "glTF defines +Y as up", "The units for all linear distances are meters" | Khronos glTF 2.0 Specification, source `https://github.com/KhronosGroup/glTF/blob/main/specification/2.0/Specification.adoc` (section "Coordinate System and Units"). The rendered page on registry.khronos.org was behind a bot check, so the source file was read instead | 2026-10-23 |
| glTF base colour factors are linear ("The factor value is a linear multiplier") | Khronos glTF 2.0 Specification, as above, section "Metallic-Roughness Material" | 2026-10-23 |
| sRGB transfer function: C' = 12.92 C for C ≤ 0.0031308, else 1.055 C^(1/2.4) − 0.055 | International Color Consortium, "sRGB", `https://www.color.org/chardata/rgb/srgb.xalter` (the glTF specification's reference for sRGB is IEC 61966-2-1) | 2026-10-23 |
| three.js is the most used WebGL library: npm downloads in the week 2026-10-12 to 2026-10-18 were three 14 115 096, @babylonjs/core 234 083, cesium 195 653, playcanvas 106 867 | npm download counts API, `https://api.npmjs.org/downloads/point/last-week/<package>` | 2026-10-23 |
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

### Viewer (`ui/`)

Licences from the npm registry metadata of each package; exact versions are pinned in
`ui/package.json`.

| Package | Licence | Checked |
| --- | --- | --- |
| three 0.186.1 (3D drawing) | MIT | 2026-10-23 |
| vite 8.3.4 (build and development server, development only) | MIT | 2026-10-23 |
| typescript 7.0.2 (type checking, development only) | Apache-2.0 | 2026-10-23 |
| vitest 5.0.5 (unit tests, development only) | MIT | 2026-10-23 |
| @playwright/test 1.56.1 (browser tests, development only) | Apache-2.0 | 2026-10-23 |
| prettier 3.9.9 (formatting, development only) | MIT | 2026-10-23 |
| @types/three 0.186.0, @types/node 22.20.5 (type definitions, development only) | MIT | 2026-10-23 |
