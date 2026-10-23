# Assumptions

Values the project uses that are not measured or taken from a verified source. Each says where it
is used, why it is acceptable for now, and what will replace it. Results that depend on an
assumption are labelled with its ID.

| ID | Assumption | Used in | Why acceptable for now | Replaced by |
| --- | --- | --- | --- | --- |
| A-001 | One storey is 3.0 m tall, when a building's height has to be estimated from its number of storeys | `site/heights.py`; heights labelled `osm:levels` | Storey heights vary (homes are lower than offices); every estimated height is labelled so it can be told apart from a mapped one | Measured heights: city 3D models, Overture Maps heights, or a building-height dataset for the region |
| A-002 | A building with no height has about the median height of the buildings around it: those with a known height whose centres are within 100 m, if there are at least 3 | `site/fill.py`; heights labelled `estimated:neighbour-median` | Neighbouring buildings are often of a similar age and type; the median ignores a single tower among low houses; every filled height is labelled and counted in the site summary, and buildings with too few neighbours stay without a height and out of the grid | Measured heights: Overture Maps, city 3D models, or a building-height dataset |

## Display-only values

These values change only how a feature is drawn when the data do not give its size. The physics
never reads them.

| Value | Used for | In |
| --- | --- | --- |
| 3.0 m per lane (2 lanes if not mapped) | width of a drawn road without a `width` tag | `site/mesh.py`, `DISPLAY_LANE_WIDTH_M` |
| 2.0 m | width of a drawn footway without a `width` tag | `DISPLAY_FOOTWAY_WIDTH_M` |
| 0.15 m | radius of a drawn trunk without a mapped circumference | `DISPLAY_TRUNK_RADIUS_M` |
| 0.6 of the tree height | width of a drawn crown without a mapped `diameter_crown` | `DISPLAY_CROWN_SHARE` |
| 0.6 of the tree height, at most the crown width | depth of a drawn crown | `DISPLAY_CROWN_DEPTH_SHARE` |
| 0.25 of the crown depth | how far a drawn trunk reaches into the crown | `DISPLAY_TRUNK_INTO_CROWN` |
| A sun 40° above the horizon, from the south-east (135° from north), and a clear sky | lighting of the 3D view, chosen so that shadows fall where the starting camera can see them | `ui/src/viewer.ts`, `DISPLAY_SUN_*` and the sky settings |
