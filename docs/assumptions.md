# Assumptions

Values the project uses that are not measured or taken from a verified source. Each says where it
is used, why it is acceptable for now, and what will replace it. Results that depend on an
assumption are labelled with its ID.

| ID | Assumption | Used in | Why acceptable for now | Replaced by |
| --- | --- | --- | --- | --- |
| A-001 | One storey is 3.0 m tall, when a building's height has to be estimated from its number of storeys | `site/heights.py`; heights labelled `osm:levels` | Storey heights vary (homes are lower than offices); every estimated height is labelled so it can be told apart from a measured one | Measured heights: city 3D models, Overture Maps heights, or a building-height dataset for the region |
