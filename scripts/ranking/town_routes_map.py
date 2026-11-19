"""Context map: Ang Mo Kio planning area (URA land use, SVY21) with the 10 senior walking routes."""
import sys, sqlite3, pathlib
import shapefile, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from shapely import wkb, ops
from shapely.geometry import shape
from pyproj import Transformer
AMK, V2, OUT = map(pathlib.Path, sys.argv[1:4])
to_svy = Transformer.from_crs(4326, 3414, always_xy=True).transform
NAVY, BG = "#2B4C6D", "#FAF8F3"
fig, ax = plt.subplots(figsize=(9, 9), dpi=220); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
lu = shapefile.Reader(str(AMK / "Landuse.shp")); polys = []
for sr, rec in zip(lu.iterShapes(), lu.iterRecords()):
    g = shape(sr.__geo_interface__).buffer(0); polys.append(g); d = rec["LU_DESC"]
    col = {"PARK": "#DCE6D3", "OPEN SPACE": "#E6EBDD", "WATERBODY": "#C3D7EA", "MASS RAPID TRANSIT": "#4173B0"}.get(d)
    if col:
        for p in getattr(g, "geoms", [g]): ax.fill(*p.exterior.xy, color=col, lw=0, zorder=1)
town = ops.unary_union(polys).buffer(15).buffer(-15)
for p in getattr(town, "geoms", [town]): ax.plot(*p.exterior.xy, color="#56708A", lw=1.2, zorder=4)
bd = shapefile.Reader(str(V2 / "6. Ranking_Integration/inputs/building.shp"))
for sh in bd.iterShapes():
    g = ops.transform(to_svy, shape(sh.__geo_interface__))
    for p in getattr(g, "geoms", [g]):
        if p.geom_type == "Polygon": ax.fill(*p.exterior.xy, color="#E4E0D6", lw=0, zorder=2)
rd = shapefile.Reader(str(AMK / "Road Network.shp"))
for sh, rec in zip(rd.iterShapes(), rd.iterRecords()):
    w = {"motorway": 1.6, "trunk": 1.4, "primary": 1.1, "secondary": 0.8}.get(rec["fclass"])
    if w:
        g = ops.transform(to_svy, shape(sh.__geo_interface__))
        for l in getattr(g, "geoms", [g]): ax.plot(*l.xy, color="#FFFFFF", lw=w * 2.2, zorder=3, solid_capstyle="round")
con = sqlite3.connect(V2 / "3. Shade_by_Hour_Complete/data/routes/senior_routes.gpkg")
gp = lambda b: wkb.loads(bytes(b[8 + {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[(b[3] >> 1) & 7]:]))
for name, sql in [("seg", "select route_id, seq, geom from route_segments order by route_id, seq")]:
    import pandas as pd
    seg = pd.read_sql(sql, con)
for rid, grp in seg.groupby("route_id"):
    lines = [ops.transform(to_svy, gp(b)) for b in grp.geom]
    for l in lines: ax.plot(*l.xy, color=NAVY, lw=4.2, zorder=6, solid_capstyle="round")
    o = lines[0].coords[0]; dpt = lines[-1].coords[-1]
    ax.scatter(*o, s=55, color="#EB8861", edgecolor=NAVY, lw=1, zorder=7)
    ax.scatter(*dpt, s=70, marker="s", color="#EDCE7A", edgecolor=NAVY, lw=1, zorder=7)
    mid = ops.linemerge(ops.unary_union(lines)); mid = mid if mid.geom_type == "LineString" else max(mid.geoms, key=lambda x: x.length)
    m = mid.interpolate(0.5, normalized=True)
    ax.annotate(str(int(rid[1:])), (m.x, m.y), xytext=(14, 10), textcoords="offset points", ha="center", va="center", fontsize=11,
                fontweight="bold", color=BG, zorder=9, bbox=dict(boxstyle="circle,pad=0.3", fc=NAVY, ec="none"))
x0, y0, x1, y1 = town.bounds; ax.set_xlim(x0 - 120, x1 + 120); ax.set_ylim(y0 - 120, y1 + 120); ax.set_aspect(1); ax.axis("off")
sx = x0 + 150; sy = y0 + 40; ax.plot([sx, sx + 1000], [sy, sy], color=NAVY, lw=3); ax.text(sx + 500, sy + 70, "1 km", ha="center", color=NAVY, fontsize=11)
ax.annotate("N", (x1 - 100, y1 - 300), xytext=(x1 - 100, y1 - 500), ha="center", fontsize=13, fontweight="bold", color=NAVY,
            arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=2))
plt.tight_layout(pad=0.2); plt.savefig(OUT, facecolor=BG); print("ok", town.bounds)
