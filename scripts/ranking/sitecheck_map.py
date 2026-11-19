"""Site-check map: high-priority segments (score >= 0.7) and linkway candidates on the AMK base map,
with destination amenities labelled. Args: <base-map folder, e.g. V2/6. Ranking_Integration/inputs> <V2 folder> <out.png>"""
import sys, sqlite3, pathlib
import pandas as pd, shapefile, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from shapely import wkb, ops
from shapely.geometry import shape, box
from pyproj import Transformer
AMK, V2, OUT = map(pathlib.Path, sys.argv[1:4])
to_svy = Transformer.from_crs(4326, 3414, always_xy=True).transform
NAVY, BG, HI, LK = "#2B4C6D", "#FAF8F3", "#EB8861", "#4173B0"
s = pd.read_csv(V2 / "6. Ranking_Integration/outputs_v2/segment_priority_v2.csv")
con = sqlite3.connect(V2 / "3. Shade_by_Hour_Complete/data/routes/senior_routes.gpkg")
gp = lambda b: wkb.loads(bytes(b[8 + {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[(b[3] >> 1) & 7]:]))
seg = pd.read_sql("select seg_id, route_id, seq, geom from route_segments order by route_id, seq", con).merge(
    s[["seg_id", "priority", "linkway_candidate", "qc_status"]], on="seg_id")
seg["g"] = seg.geom.apply(lambda b: ops.transform(to_svy, gp(b)))
u = ops.unary_union(list(seg.g)); x0, y0, x1, y1 = u.bounds; pad = 260
x0 -= pad; x1 += pad + 250; y0 -= pad; y1 += pad
view = box(x0, y0, x1, y1)
fig, ax = plt.subplots(figsize=(11.5, 9), dpi=200); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
lu = shapefile.Reader(str(AMK / "Landuse.shp"))
for sr, rec in zip(lu.iterShapes(), lu.iterRecords()):
    col = {"PARK": "#DCE6D3", "OPEN SPACE": "#E6EBDD", "WATERBODY": "#C3D7EA", "MASS RAPID TRANSIT": "#9DB5D3"}.get(rec["LU_DESC"])
    if col:
        g = shape(sr.__geo_interface__).buffer(0)
        if g.intersects(view):
            for p in getattr(g, "geoms", [g]): ax.fill(*p.exterior.xy, color=col, lw=0, zorder=1)
for sh in shapefile.Reader(str(V2 / "6. Ranking_Integration/inputs/building.shp")).iterShapes():
    g = ops.transform(to_svy, shape(sh.__geo_interface__))
    if not g.intersects(view): continue
    for p in getattr(g, "geoms", [g]):
        if p.geom_type == "Polygon": ax.fill(*p.exterior.xy, color="#E4E0D6", lw=0, zorder=2)
rd = shapefile.Reader(str(AMK / "Road Network.shp"))
for sh, rec in zip(rd.iterShapes(), rd.iterRecords()):
    w = {"motorway": 1.6, "trunk": 1.4, "primary": 1.1, "secondary": 0.8}.get(rec["fclass"])
    if w:
        g = ops.transform(to_svy, shape(sh.__geo_interface__))
        if g.intersects(view):
            for l in getattr(g, "geoms", [g]): ax.plot(*l.xy, color="#FFFFFF", lw=w * 2.4, zorder=3, solid_capstyle="round")
for _, r in seg.iterrows(): ax.plot(*r.g.xy, color="#B7C3CF", lw=3.2, zorder=4, solid_capstyle="round")
for _, r in seg[seg.linkway_candidate].iterrows(): ax.plot(*r.g.xy, color=LK, lw=5.5, zorder=5, solid_capstyle="butt")
for _, r in seg[seg.priority >= 0.7].iterrows():
    ax.plot(*r.g.xy, color=NAVY, lw=10, zorder=6, solid_capstyle="round")
    ax.plot(*r.g.xy, color=HI, lw=7, zorder=7, solid_capstyle="round")
    if r.qc_status == "REVIEW_DISTANCE": ax.plot(*r.g.xy, color=BG, lw=1.8, ls=(0, (2, 2)), zorder=8)
DEST = {"R01": "AMK Community Hospital", "R02": "628 Market", "R03": "Mayflower MRT", "R04": "Blk 226 shops", "R05": "AMK Town Centre",
        "R06": "Blk 338 shops", "R07": "AMK MRT & bus interchange", "R08": "Blk 548 shops", "R09": "Blk 525 shops", "R10": "Blk 574 shops"}
OFF = {"R01": (12, 6), "R02": (12, -2), "R03": (-12, -14), "R04": (10, 8), "R05": (-12, 10), "R06": (8, -16), "R07": (12, -22),
       "R08": (10, 10), "R09": (12, -2), "R10": (12, -10)}
BADGE = {"R01": (-18, 0), "R09": (16, 14), "R10": (-16, -12)}
for rid, g in seg.groupby("route_id"):
    o = g.iloc[0].g.coords[0]; d = g.iloc[-1].g.coords[-1]
    ax.scatter(*o, s=40, color="#FAF8F3", edgecolor=NAVY, lw=1.4, zorder=9)
    ax.scatter(*d, s=95, marker="s", color="#EDCE7A", edgecolor=NAVY, lw=1.3, zorder=9)
    dx, dy = OFF[rid]
    ax.annotate(f"{DEST[rid]}", d, xytext=(dx, dy), textcoords="offset points", fontsize=10.5, fontweight="bold", color=NAVY, zorder=10,
                ha="left" if dx > 0 else "right", va="center", bbox=dict(boxstyle="round,pad=0.25", fc=BG, ec="none", alpha=.9))
    m = ops.linemerge(ops.unary_union(list(g.g))); m = m if m.geom_type == "LineString" else max(m.geoms, key=lambda x: x.length)
    p = m.interpolate(0.5, normalized=True)
    ax.annotate(str(int(rid[1:])), (p.x, p.y), xytext=BADGE.get(rid, (-14, -12)), textcoords="offset points", ha="center", va="center", fontsize=10,
                fontweight="bold", color=BG, zorder=10, bbox=dict(boxstyle="circle,pad=0.28", fc=NAVY, ec="none"))
ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect(1); ax.axis("off")
sx, sy = x0 + 80, y0 + 60; ax.plot([sx, sx + 500], [sy, sy], color=NAVY, lw=3); ax.text(sx + 250, sy + 40, "500 m", ha="center", fontsize=10, color=NAVY)
ax.annotate("N", (x1 - 90, y1 - 140), xytext=(x1 - 90, y1 - 300), ha="center", fontsize=13, fontweight="bold", color=NAVY,
            arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=2))
plt.subplots_adjust(0, 0, 1, 1); plt.savefig(OUT, facecolor=BG); print("ok", round(x1 - x0), round(y1 - y0))
