# Where do seniors walk in the sun?

AR5958A AI for Urban Planning (NUS), Group 5. We rank 134 footpath segments on ten senior walking routes in Ang Mo Kio by sun and rain exposure, so agencies know where to check first for covered linkways and shade trees.

**Live page:** https://serenelaww.github.io/ar5958a-final-project/

## Method in one line
KartaView 360° panoramas → SegFormer (ADE20K) pixel labels → sky openness = sky ÷ (sky + tree), 0 under an LTA linkway → NEA walking-hour rain rate × uncovered share → priority = 0.7 × openness + 0.3 × rain (both rescaled 0–1).

## Repository layout
| Path | Contents |
|---|---|
| `index.html` | The project page (GitHub Pages) |
| `assets/` | Figures and `segments.js` (segment geometry + scores for the interactive map) |
| `data/` | Result tables: `segment_priority_v2.csv`, `route_priority_v2.csv`, `sensitivity_weights_v2.csv`, `segment_shelter_lta.csv`, `photo_vs_panorama_v2.csv`, `AMK_Rain_Exposure_by_segment.csv`, `segment_geometry_wkt.csv` |
| `scripts/shade/` | `shade_pipeline.py`: panorama segmentation, view factors and shade-by-hour model |
| `scripts/ranking/` | Integration and mapping scripts |

## Run order (ranking)
1. `shelter_lta_v2.py`: LTA Covered Linkway overlay (3 m tolerance) → `segment_shelter_lta.csv`
2. `rain_table_v2.py`: walking-hour rain exposure per segment
3. `integrate_v2.py`: joins openness, shelter and rain; scores, ranks and tests weights → `segment_priority_v2.csv`, `route_priority_v2.csv`, `sensitivity_weights_v2.csv`
4. `photo_check_v2.py`, `photo_check_plot.py`: Mapillary street-photo cross-check
5. Maps: `town_routes_map.py`, `sitecheck_map.py`, `route_priority_map.py`, `rain_overview_map.py`, `map_v2.py`

`rank_segments.py` is an earlier pilot version kept for reference; `integrate_v2.py` is the final ranking.

Raw inputs (panoramas, Mapillary photos, NEA downloads, LTA and URA shapefiles) are not included because of size and licensing; see the Data sources table on the page. Python 3.10+, pandas, geopandas, shapely, matplotlib; see `scripts/shade/requirements.txt` for the segmentation model.

## Data and licences
Imagery © KartaView and Mapillary contributors (CC BY-SA). Map data © OpenStreetMap contributors (ODbL). Rainfall and WBGT: NEA via data.gov.sg. Covered linkways: LTA DataMall. Land use: URA Master Plan.
