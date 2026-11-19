"""Slide 11 map: Flora's rain exposure score (rain frequency x uncovered metres, natural breaks, 5 classes) on the
same Ang Mo Kio base map as the route map. Legend is left out of the image (drawn on the slide).
Args: <base-map folder, e.g. V2/6. Ranking_Integration/inputs> <V2 folder> <out.png> <classes.json>"""
import sys, sqlite3, pathlib, json
import numpy as np, pandas as pd, shapefile, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from shapely import wkb, ops
from shapely.geometry import shape
from pyproj import Transformer
AMK, V2, OUT, CLS = map(pathlib.Path, sys.argv[1:5])
to_svy = Transformer.from_crs(4326, 3414, always_xy=True).transform
NAVY, BG = "#2B4C6D", "#FAF8F3"
COLS = ["#C3D7EA", "#8FB0D6", "#EDCE7A", "#EB8861", "#B9473A"]; NAMES = ["Very low", "Low", "Moderate", "High", "Very high"]

def jenks(v, k):
    v = np.sort(v); n = len(v); mat1 = np.zeros((n + 1, k + 1)); mat2 = np.full((n + 1, k + 1), np.inf)
    mat1[1, 1:] = 1; mat2[1, 1:] = 0
    for l in range(2, n + 1):
        s1 = s2 = w = 0.0
        for m in range(1, l + 1):
            i3 = l - m + 1; val = v[i3 - 1]; s2 += val * val; s1 += val; w += 1; var = s2 - s1 * s1 / w
            if i3 != 1:
                for j in range(2, k + 1):
                    if mat2[l, j] >= var + mat2[i3 - 1, j - 1]: mat1[l, j] = i3; mat2[l, j] = var + mat2[i3 - 1, j - 1]
        mat1[l, 1] = 1; mat2[l, 1] = var
    br = [v[-1]]; kk = n
    for j in range(k, 1, -1): idx = int(mat1[kk, j]) - 2; br.append(v[idx]); kk = int(mat1[kk, j]) - 1
    return [v[0]] + sorted(br)

fig, ax = plt.subplots(figsize=(9, 9), dpi=220); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
lu = shapefile.Reader(str(AMK / "Landuse.shp")); polys = []
for sr, rec in zip(lu.iterShapes(), lu.iterRecords()):
    g = shape(sr.__geo_interface__).buffer(0); polys.append(g)
    col = {"PARK": "#DCE6D3", "OPEN SPACE": "#E6EBDD", "WATERBODY": "#C3D7EA", "MASS RAPID TRANSIT": "#4173B0"}.get(rec["LU_DESC"])
    if col:
        for p in getattr(g, "geoms", [g]): ax.fill(*p.exterior.xy, color=col, lw=0, zorder=1)
town = ops.unary_union(polys).buffer(15).buffer(-15)
for p in getattr(town, "geoms", [town]): ax.plot(*p.exterior.xy, color="#56708A", lw=1.2, zorder=4)
for sh in shapefile.Reader(str(V2 / "6. Ranking_Integration/inputs/building.shp")).iterShapes():
    g = ops.transform(to_svy, shape(sh.__geo_interface__))
    for p in getattr(g, "geoms", [g]):
        if p.geom_type == "Polygon": ax.fill(*p.exterior.xy, color="#E4E0D6", lw=0, zorder=2)
rd = shapefile.Reader(str(AMK / "Road Network.shp"))
for sh, rec in zip(rd.iterShapes(), rd.iterRecords()):
    w = {"motorway": 1.6, "trunk": 1.4, "primary": 1.1, "secondary": 0.8}.get(rec["fclass"])
    if w:
        g = ops.transform(to_svy, shape(sh.__geo_interface__))
        for l in getattr(g, "geoms", [g]): ax.plot(*l.xy, color="#FFFFFF", lw=w * 2.2, zorder=3, solid_capstyle="round")
t = pd.read_csv(V2 / "4. Rain Exposure-1109/AMK_Rain_Exposure_by_segment.csv")
br = jenks(t.rain_score_flora.values, 5)
t["cls"] = np.clip(np.searchsorted(br[1:-1], t.rain_score_flora.values, side="left"), 0, 4)
con = sqlite3.connect(V2 / "3. Shade_by_Hour_Complete/data/routes/senior_routes.gpkg")
gp = lambda b: wkb.loads(bytes(b[8 + {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[(b[3] >> 1) & 7]:]))
seg = pd.read_sql("select seg_id, route_id, seq, geom from route_segments order by route_id, seq", con).merge(t[["seg_id", "cls"]], on="seg_id")
for rid, grp in seg.groupby("route_id"):
    lines = [ops.transform(to_svy, gp(b)) for b in grp.geom]
    for l in lines: ax.plot(*l.xy, color=NAVY, lw=7, zorder=5, solid_capstyle="round")
    for l, c in zip(lines, grp.cls): ax.plot(*l.xy, color=COLS[c], lw=4.4, zorder=6, solid_capstyle="butt")
    mid = ops.linemerge(ops.unary_union(lines)); mid = mid if mid.geom_type == "LineString" else max(mid.geoms, key=lambda x: x.length)
    m = mid.interpolate(0.5, normalized=True)
    ax.annotate(str(int(rid[1:])), (m.x, m.y), xytext=(14, 10), textcoords="offset points", ha="center", va="center", fontsize=11,
                fontweight="bold", color=BG, zorder=9, bbox=dict(boxstyle="circle,pad=0.3", fc=NAVY, ec="none"))
st = pd.read_csv(V2 / "4. Rain Exposure-1109/NEA_Stations_ArcGIS.csv")
st = st.rename(columns={"location_latitude": "lat", "location_longitude": "lon"}).dropna(subset=["lat"])  # V2 coordinates only
x0, y0, x1, y1 = town.bounds; shown = []
for _, s in st.iterrows():
    x, y = to_svy(s.lon, s.lat)
    if x0 - 300 <= x <= x1 + 300 and y0 - 300 <= y <= y1 + 300:
        ax.scatter(x, y, s=150, marker="^", color="#EDCE7A", edgecolor=NAVY, lw=1.4, zorder=8)
        ax.annotate(f"{s.station_id}  {s.rain_freq*100:.1f}%", (x, y), xytext=(10, -4), textcoords="offset points", fontsize=10,
                    fontweight="bold", color=NAVY, zorder=9, bbox=dict(boxstyle="round,pad=0.2", fc=BG, ec="none", alpha=.85))
        shown.append({"id": s.station_id, "name": s.station_name if isinstance(s.station_name, str) else s.station_id, "freq_pct": round(s.rain_freq * 100, 2)})
        x0, y0, x1, y1 = min(x0, x - 200), min(y0, y - 200), max(x1, x + 900), max(y1, y + 200)
ax.set_xlim(x0 - 120, x1 + 120); ax.set_ylim(y0 - 120, y1 + 120); ax.set_aspect(1); ax.axis("off")
bx, by = town.bounds[0] + 150, town.bounds[1] + 40
ax.plot([bx, bx + 1000], [by, by], color=NAVY, lw=3); ax.text(bx + 500, by + 70, "1 km", ha="center", color=NAVY, fontsize=11)
ax.annotate("N", (x1 - 100, y1 - 300), xytext=(x1 - 100, y1 - 500), ha="center", fontsize=13, fontweight="bold", color=NAVY,
            arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=2))
plt.tight_layout(pad=0.2); plt.savefig(OUT, facecolor=BG)
counts = t.cls.value_counts().reindex(range(5), fill_value=0).tolist()
json.dump({"breaks": [round(float(b), 3) for b in br], "names": NAMES, "colours": COLS, "counts": counts, "stations": shown}, open(CLS, "w"), indent=1)
print(json.load(open(CLS)))
