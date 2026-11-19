"""Covered length per segment from LTA's Covered Linkway layer (DataMall, Sep 2026; polygons in SVY21).
Rebuilds Flora's ArcGIS step (Intersect segments x CoveredLinkWay -> SUM cover_m -> uncover_m = length - cover).
Segments (WGS84) are projected to SVY21 (EPSG:3414). Covered = segment length inside a linkway roof polygon.
Main result uses a 3 m tolerance (path centrelines are drawn beside, not exactly under, the roof outline);
0 m is kept as a strict check. Output: outputs_v2/segment_shelter_lta.csv"""
import sqlite3, pathlib, zipfile
import pandas as pd, shapefile
from shapely import wkb, ops
from shapely.geometry import shape
from shapely.strtree import STRtree
from pyproj import Transformer
HERE = pathlib.Path(__file__).resolve().parent; V2 = HERE.parent; OUT = HERE / "outputs_v2"
SRC = V2 / "4. Rain Exposure-1109"
if not (SRC / "CoveredLinkWay_Sep2026/CoveredLinkWay.shp").exists():
    zipfile.ZipFile(SRC / "CoveredLinkWay_Sep2026.zip").extractall(SRC)
polys = [shape(s.__geo_interface__) for s in shapefile.Reader(str(SRC / "CoveredLinkWay_Sep2026/CoveredLinkWay.shp")).iterShapes()]
polys = [p if p.is_valid else p.buffer(0) for p in polys]
tree = STRtree(polys)
to_svy = Transformer.from_crs(4326, 3414, always_xy=True).transform

def gpkg_geom(blob):
    env = (blob[3] >> 1) & 0x07
    return wkb.loads(bytes(blob[8 + {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[env]:]))

con = sqlite3.connect(V2 / "3. Shade_by_Hour_Complete/data/routes/senior_routes.gpkg")
seg = pd.read_sql("select seg_id, route_id, seq, length_m, geom from route_segments", con)
rows = []
for _, s in seg.iterrows():
    line = ops.transform(to_svy, gpkg_geom(s.geom))
    r = {"seg_id": s.seg_id, "route_id": s.route_id, "length_svy_m": round(line.length, 1)}
    for tol in (0, 3):
        cand = [polys[i] for i in tree.query(line.buffer(tol + 1))]
        roof = ops.unary_union([p.buffer(tol) if tol else p for p in cand]) if cand else None
        r[f"covered_m_tol{tol}"] = round(line.intersection(roof).length, 1) if roof is not None else 0.0
    rows.append(r)
d = pd.DataFrame(rows)
d["covered_share"] = (d.covered_m_tol3 / d.length_svy_m).clip(0, 1).round(3)
d.to_csv(OUT / "segment_shelter_lta.csv", index=False)
print("linkway polygons", len(polys))
print("segments with any cover (3 m):", int((d.covered_m_tol3 > 0).sum()), "| (0 m):", int((d.covered_m_tol0 > 0).sum()), "of", len(d))
print("total covered m (3 m):", d.covered_m_tol3.sum().round(0), "of", d.length_svy_m.sum().round(0), "| share", round(d.covered_m_tol3.sum() / d.length_svy_m.sum(), 3))
print("fully covered (>=90%):", int((d.covered_share >= .9).sum()))
print(d.groupby("route_id").apply(lambda g: round(g.covered_m_tol3.sum() / g.length_svy_m.sum(), 2), include_groups=False).to_dict())
