# Assumptions

Values the project uses that are not measured or taken from a verified source. Each says where it
is used, why it is acceptable for now, and what will replace it. Results that depend on an
assumption are labelled with its ID.

| ID | Assumption | Used in | Why acceptable for now | Replaced by |
| --- | --- | --- | --- | --- |
| A-001 | One storey is 3.0 m tall, when a building's height has to be estimated from its number of storeys | `site/heights.py`; heights labelled `osm:levels` | Storey heights vary (homes are lower than offices); every estimated height is labelled so it can be told apart from a mapped one | Measured heights: city 3D models, Overture Maps heights, or a building-height dataset for the region |
| A-002 | A building with no height has about the median height of the buildings around it: those with a known height whose centres are within 100 m, if there are at least 3 | `site/fill.py`; heights labelled `estimated:neighbour-median` | Neighbouring buildings are often of a similar age and type; the median ignores a single tower among low houses; every filled height is labelled and counted in the site summary, and buildings with too few neighbours stay without a height and out of the grid | Measured heights: Overture Maps, city 3D models, or a building-height dataset |
| A-003 | Kinds of building height are trusted in this order: measured by lidar, mapped (a height in map data), from storeys (A-001), estimated by machine learning, median of neighbours (A-002). Each building keeps the most trusted height available | `site/heights.py` (`HEIGHT_KINDS`), `site/from_overture.py` | Lidar measures heights directly; mapped heights are usually surveyed or taken from plans; storey counts are exact counts with an assumed storey height; machine-learning estimates are known to carry errors of metres; neighbours say nothing about the building itself. The order is a judgement, and every height keeps its kind and dataset, so results can be checked by kind | A comparison against measured heights for sample cities, which can reorder the kinds per region |
| A-004 | An Overture footprint that overlaps a building already in the site by more than half of the smaller of the two is the same building | `site/from_overture.py` (`OVERLAP_SHARE`) | Overture already removes most such doubles; the rule only catches what remains. Real neighbouring buildings share walls, not half their area | A match on both overlap and outline shape |
| A-005 | Between the pixels of the elevation model (about 30 m apart), the ground is a surface of flat triangles on a 20 m grid | `site/terrain.py`, `site/build.py` (`TERRAIN_DX_M`) | The models hold no detail finer than their pixels; any smooth surface through them is a guess, and flat triangles are the simplest one, the same one the view draws | Finer elevation data where it exists (city lidar surveys) |
| A-006 | A building stands on the lowest ground under its footprint | `site/terrain.py` (`lowest_under`) | OpenStreetMap measures a building's height from the lowest point where it meets the ground (Simple 3D Buildings); on a slope, the uphill side of the building is partly below ground, as many hillside buildings are | Building base heights from city 3D models |

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
| 1 km of ground around the site | the surroundings drawn in the 3D view; the physics uses only the site's square | `site/build.py`, `CONTEXT_M` |
| 0.15 m (roads), 0.2 m (footways) | how far above the ground roads and paths are drawn, so the ground does not show through them | `site/mesh.py`, `DISPLAY_ROAD_LIFT_M`, `DISPLAY_FOOTWAY_LIFT_M` |
| 3 m | width of the band that marks the edge of the simulated area | `site/mesh.py`, `DISPLAY_OUTLINE_WIDTH_M` |
