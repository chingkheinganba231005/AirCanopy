# Sources

Every library, dataset and reference the project relies on, with its licence and when it was
checked. Anything that could not be checked is marked `UNVERIFIED`.

## Data

| Data | Source | Licence | Checked |
| --- | --- | --- | --- |
| Overture Maps buildings (footprints, heights, storey counts, with the dataset behind each) | Release 2026-10-21.0, GeoParquet on Amazon S3, found through the STAC catalogue `https://stac.overturemaps.org/catalog.json` | Theme licence ODbL-1.0 (stated in the STAC collection and on `https://docs.overturemaps.org/attribution/`). Credit "© OpenStreetMap contributors, Overture Maps Foundation", plus each dataset's own credit, listed in `sources/overture.py` from the attribution page | 2026-10-23 |
| East Asian building footprints inside Overture | Qian Shi, Jiajun Zhu, Zhengyu Liu, Haonan Guo, Mengxi Liu, Zihong Liu, "A first high-quality vector data of buildings in East Asian countries based on a comprehensive large-scale mapping framework", Zenodo, 2023-07-22, doi:10.5281/zenodo.8174931 | CC BY 4.0 (Zenodo record metadata) | 2026-10-23 |
| Ground elevation: GEDTM30 v1.2, a bare-earth terrain model at 1 arc-second, with a per-pixel standard deviation | `https://s3.opengeohub.org/global/dtm/v1.2/` (file list in `https://codeberg.org/openlandmap/GEDTM30`, `metadata/cog_list.csv`); Zenodo record doi:10.5281/zenodo.18887460 (v1.2.0, 2026-03-06); method: Ho et al. 2025, PeerJ, doi:10.7717/peerj.19673. Heights above EGM2008 (the file's own `VERTICAL_DATUM` tag says `EGM2008/EPSG:3855`). The record calls this first release one "for testing purposes" | CC BY 4.0 (Zenodo record and the repository's README) | 2026-10-23 |
| Ground elevation, fallback: Copernicus DEM GLO-30 | `https://copernicus-dem-30m.s3.amazonaws.com/` (AWS Open Data Registry, `copernicus-dem`). It "is a Digital Surface Model (DSM) which represents the surface of the Earth including buildings, infrastructure and vegetation"; "ocean areas do not have tiles, there one can assume height values equal to zero" (registry entry). Vertical datum EGM2008, EPSG 3855 (Copernicus DEM Product Handbook, issue 6.1, section 1.2.1) | Free licence; required notice for modified data: "produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved" (`https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM`) | 2026-10-23 |
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
| Overture's file layout: 512 building files, 277 GB, 128 row groups per file, bounding-box statistics per row group; the collection lists one box per file in the order of its items | Our own reading of release 2026-10-21.0: the S3 listing, the STAC collection and items, and the Parquet footers (sources/overture.py) | 2026-10-23 |
| Dataset names in Overture building sources ("OpenStreetMap", "Microsoft ML Buildings", "Google Open Buildings", "USGS Lidar", "doi:10.5281/zenodo.8174931"), and how many buildings have heights by region | Our own sample of seven files of release 2026-10-21.0, `benchmarks/overture_probe.py` → `results/overture_probe.json` | 2026-10-23 |
| Over the 500 m HKU square, Copernicus GLO-30 lies 3.45 m above GEDTM30 on average (from −9.32 to +22.76 m); GEDTM30's uncertainty averages 3.12 m; reading the site's window cost 2.47 MB of GEDTM30's 432 GB file | Our own computation, `benchmarks/terrain_probe.py` → `results/terrain_probe.json` | 2026-10-23 |
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
| pyarrow | Apache-2.0 | 2026-10-23 |
| rasterio (with GDAL inside its wheels) | BSD-3-Clause (rasterio); MIT (GDAL) | 2026-10-23 |
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
