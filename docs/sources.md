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
| Tree canopy height, about 1 m, worldwide: High Resolution Canopy Height Maps v1 | `https://dataforgood-fb-data.s3.amazonaws.com/forests/v1/alsgedi_global_v6_float/` (AWS Open Data Registry, `dataforgood-fb-forests`); tiles at web-map zoom 9 named by quadkey (our quadkey formula reproduces all 73 tile names we checked in the map's `tiles.geojson`); per-tile image dates in `metadata/<quadkey>.geojson`. Method and accuracy: Tolan et al. 2024, "Very high resolution canopy height maps from RGB imagery using self-supervised vision transformer and convolutional decoder trained on aerial lidar", Remote Sensing of Environment, doi:10.1016/j.rse.2023.113888 (arXiv 2304.07213): mean absolute error 2.8 m, mean error 0.6 m; "thresholding height values of at least five meters (5m) as tree canopy extent"; annotators labelled trees "above one meter (1m) tall and with a canopy diameter of more than three meters (3m)". Version 2 (`forests/v2/`) is not used: its licence is not documented | CC BY 4.0 (registry). Citation: "Meta and World Resources Institude (WRI) - 2024. High Resolution Canopy Height Maps (CHM). Source imagery for CHM © 2016 Maxar" | 2026-10-23 |
| Place names to coordinates | Nominatim, OpenStreetMap's geocoder, `https://nominatim.openstreetmap.org/search` (another server can be set with `AIRCANOPY_GEOCODER`); answers cached in the cache folder | The answers are OpenStreetMap data: each states "Data © OpenStreetMap contributors, ODbL 1.0. http://osm.org/copyright". Use follows the Nominatim Usage Policy (see References) | 2026-10-23 |
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
| Highway classes for vehicles (motorway to service roads, living streets and the link roads) and for walking (footway, pedestrian, path, steps, cycleway); `highway=corridor` is "For a hallway inside of a building", so corridors are left out | OpenStreetMap wiki, "Key:highway", `https://wiki.openstreetmap.org/wiki/Key:highway` | 2026-10-23 |
| `building:levels` is "The number of above ground levels in the building facades (excluding roof levels and underground levels)", and for a part that does not start at the ground it "should include the number of skipped levels below the building part" | OpenStreetMap wiki, "Key:building:levels", `https://wiki.openstreetmap.org/wiki/Key:building:levels` (oldid 3105209) | 2026-10-23 |
| `podium` is not a documented value of `location` (the page lists underground, underwater, overground, overhead, roof, rooftop, surface, outdoor, indoor, platform, bridge, cave) | OpenStreetMap wiki, "Key:location", `https://wiki.openstreetmap.org/wiki/Key:location` | 2026-10-23 |
| Nominatim usage policy: "an absolute maximum of 1 request per second"; "Provide a valid HTTP Referer or User-Agent identifying the application" (library defaults do not count); "Clearly display attribution as suitable for your medium"; cache results; "Apps must make sure that they can switch the service at our request at any time" | OSM Foundation, "Nominatim Usage Policy", `https://operations.osmfoundation.org/policies/nominatim/` (no date on the page) | 2026-10-23 |
| What the geocoder answers for "University of Hong Kong", "The University of Hong Kong" and "Mong Kok" | Our own queries, `benchmarks/place_names.py` → `results/place_names.json` | 2026-10-23 |
| Build times with and without downloads, and the bytes downloaded, for the HKU and Mong Kok sites | Our own measurement, `benchmarks/build_time.py` → `results/build_time_hku.json`, `results/build_time_mongkok.json` | 2026-10-23 |
| Shares of buildings, footprint area and volume by kind of height; the largest buildings with estimated heights; `location=podium` towers; tunnels, bridges and features left out | Our own computation, `benchmarks/site_review.py` → `results/site_review_hku.json`, `results/site_review_mongkok.json` | 2026-10-23 |
| The textbook account of the lattice Boltzmann method: velocity sets and their isotropy conditions, lattice units, the Mach-number limit (errors grow with Ma²), boundaries, forcing | Krüger, Kusumaatmaja, Kuzmin, Shardt, Silva and Viggen, *The Lattice Boltzmann Method: Principles and Practice*, Springer 2017, doi:10.1007/978-3-319-44649-3 (bibliographic record checked on Crossref; the book itself not read in this container) | 2026-10-23 |
| Regularised collision: rebuild the non-equilibrium packets from the stress alone | Latt and Chopard, "Lattice Boltzmann method with regularized pre-collision distribution functions", Mathematics and Computers in Simulation 72:165–168 (2006), doi:10.1016/j.matcom.2006.05.017 | 2026-10-23 |
| Forcing term, and the half-force correction of the velocity | Guo, Zheng and Shi, "Discrete lattice effects on the forcing term in the lattice Boltzmann method", Physical Review E 65:046308 (2002), doi:10.1103/PhysRevE.65.046308 | 2026-10-23 |
| Smagorinsky model with the strain rate from the non-equilibrium packets | Hou, Sterling, Chen and Doolen, "A lattice Boltzmann subgrid model for high Reynolds number flows", arXiv comp-gas/9401004 (1994) | 2026-10-23 |
| Bulk and shear viscosities; removing the trace of the non-equilibrium stress damps sound | Dellar, "Bulk and shear viscosities in lattice Boltzmann equations", Physical Review E 64:031203 (2001), doi:10.1103/PhysRevE.64.031203 | 2026-10-23 |
| Starting a flow with the non-equilibrium part from the velocity gradients | Skordos, "Initial and boundary conditions for the lattice Boltzmann method", Physical Review E 48:4823–4842 (1993), doi:10.1103/PhysRevE.48.4823 | 2026-10-23 |
| Consistent initial conditions: settle the density with the velocity held | Mei, Luo, Lallemand and d'Humières, "Consistent initial conditions for lattice Boltzmann simulations", Computers & Fluids 35:855–862 (2006), doi:10.1016/j.compfluid.2005.08.008 | 2026-10-23 |
| The choice of velocity set affects rotational invariance in 3D | White and Chong, "Rotational invariance in the three-dimensional lattice Boltzmann method is dependent on the choice of lattice", Journal of Computational Physics 230:6367–6378 (2011), doi:10.1016/j.jcp.2011.04.031 (title and record checked on Crossref) | 2026-10-23 |
| Errors of the wind solver against the shear wave, Poiseuille flow and the Taylor–Green vortex, and their orders of convergence | Our own computation, `benchmarks/wind_exact.py` → `results/wind_exact.json` | 2026-10-23 |
| Speed of the wind solver on the container's processor | Our own measurement, `benchmarks/wind_speed.py` → `results/wind_speed_cpu.json` | 2026-10-23 |

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
| scipy | BSD-3-Clause | 2026-10-23 |
| scikit-image | BSD-3-Clause | 2026-10-23 |
| pytest, ruff (development only) | MIT | 2026-10-23 |
| hatchling (build only) | MIT | 2026-10-23 |
| jax 0.11.2, jaxlib 0.11.2 (the wind solver) | Apache-2.0 | 2026-10-23 |
| ml-dtypes 0.6.0, opt-einsum 3.4.0 (needed by JAX) | Apache-2.0; MIT | 2026-10-23 |
| matplotlib 3.11.2 (pictures for the journal; the optional `docs` group only) | Matplotlib License (a permissive licence based on the PSF licence) | 2026-10-23 |
| contourpy 1.4.0, cycler 0.12.1, kiwisolver 1.5.1 (needed by matplotlib) | BSD-3-Clause; BSD; BSD | 2026-10-23 |
| fonttools 4.66.1, pyparsing 3.3.3, python-dateutil 2.9.0.post0, six 1.17.0 (needed by matplotlib) | MIT; MIT; Apache-2.0 or BSD; MIT | 2026-10-23 |

Releases published in the last two weeks are held back (D-009): `pyproject.toml` sets uv's
`exclude-newer` date (D-017). In the viewer, vitest 5.0.5 was eight days old when taken, to fix
an install failure of 5.0.3; that is the exception D-009 allows.

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
