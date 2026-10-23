# Test data

Small samples of real data, so the tests run without the network. Each file's source, access date,
licence and SHA-256 are listed here.

| File | Source | Accessed | Licence | SHA-256 |
| --- | --- | --- | --- | --- |
| `osm/hku_small.osm.gz` | OpenStreetMap API, `https://api.openstreetmap.org/api/0.6/map?bbox=114.1360000,22.2832000,114.1380000,22.2848000` (a 200 m × 180 m box near the University of Hong Kong), compressed with `gzip -9 -n` | 2026-10-23T08:54:39Z | ODbL 1.0, © OpenStreetMap contributors | stored file `d9c4593ce8ef5c257b514cf35b37070db7acd2756e411f5afa62eda9a6dbb5a0`; uncompressed `40e979488b60d2e0f7419862403b87f17e9924aeee5c88a444f9d402dec0b556` |
| `overture/hku_small.parquet` | Overture Maps release 2026-10-21.0, buildings theme, type `building`: the buildings whose bounding box overlaps 114.1360,22.2832,114.1380,22.2848 (the same box as `osm/hku_small.osm.gz`), read with `aircanopy.sources.overture.fetch_buildings` from `https://overturemaps-us-west-2.s3.us-west-2.amazonaws.com/release/2026-10-21.0/theme=buildings/type=building/` | 2026-10-23T09:42:11Z | ODbL 1.0 for the theme (© OpenStreetMap contributors, Overture Maps Foundation); the footprints from Shi et al. 2023 (doi:10.5281/zenodo.8174931) are CC BY 4.0 | `6ab18bc4c9c7dce81b9121c60fa7346f25b662266baab219050d609ff17e502b` |
