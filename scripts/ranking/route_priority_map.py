"""Route deep-dive map in the same style as the R01 map (r01_map_v2.py): segments coloured by priority score
(0-1) with a horizontal score bar, segment labels, origin and destination names; flagged panoramas (>25 m) dashed.
Args: <V2 folder> <route_id> <origin label> <destination label> <out.png>"""
import sys, pathlib, pandas as pd, shapefile, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from shapely import wkt
V2, RID, ORIG, DEST, OUT = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
O = V2 / "6. Ranking_Integration/outputs_v2"; BG = "#FAF8F3"; NAVY = "#2B4C6D"
s = pd.read_csv(O / "segment_priority_v2.csv").merge(pd.read_csv(O / "segment_geometry_wkt.csv"), on="seg_id")
d = s[s.route_id == RID].sort_values("seq").copy(); d["geom"] = d.wkt.apply(wkt.loads)
cmap = LinearSegmentedColormap.from_list("o", ["#4173B0", "#C3D7EA", "#EDCE7A", "#EB8861"])
fig, ax = plt.subplots(figsize=(6, 8), dpi=200); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
b = d.geom.apply(lambda x: x.bounds); pad = 0.0012
box = [min(v[0] for v in b) - pad, min(v[1] for v in b) - pad, max(v[2] for v in b) + pad, max(v[3] for v in b) + pad]
w, h = box[2] - box[0], box[3] - box[1]
if w / h > 0.75: dh = w / 0.75 - h; box[1] -= dh / 2; box[3] += dh / 2
for sh in shapefile.Reader(str(V2 / "6. Ranking_Integration/inputs/building.shp")).iterShapes():
    bb = sh.bbox
    if bb[2] < box[0] or bb[0] > box[2] or bb[3] < box[1] or bb[1] > box[3]: continue
    pts = sh.points; parts = list(sh.parts) + [len(pts)]
    for i in range(len(parts) - 1):
        xs, ys = zip(*pts[parts[i]:parts[i + 1]]); ax.fill(xs, ys, color="#E4E0D6", lw=0)
for _, r in d.iterrows():
    x, y = r.geom.xy; ax.plot(x, y, color=cmap(r.priority), lw=7, solid_capstyle="round")
    if r.qc_status == "REVIEW_DISTANCE": ax.plot(x, y, color=BG, lw=1.6, ls=(0, (2, 2)))
    if r.length_m >= 15:
        m = r.geom.interpolate(0.5, normalized=True)
        ax.annotate(r.seg_id.split("-")[1], (m.x, m.y), xytext=(8, 0), textcoords="offset points", fontsize=8, color=NAVY, va="center")
s0 = d.iloc[0].geom.coords[0]; s1 = d.iloc[-1].geom.coords[-1]
import math
def end_label(txt, pt, nxt):
    dx, dy = pt[0] - nxt[0], pt[1] - nxt[1]; n = math.hypot(dx, dy) or 1; ux, uy = dx / n, dy / n   # point away from the route
    ax.annotate(txt, pt, xytext=(ux * 16, uy * 16), textcoords="offset points", fontsize=10, fontweight="bold", color=NAVY,
                ha="left" if ux > 0.3 else ("right" if ux < -0.3 else "center"), va="bottom" if uy > 0.3 else ("top" if uy < -0.3 else "center"))
c0 = list(d.iloc[0].geom.coords); c1 = list(d.iloc[-1].geom.coords)
end_label(ORIG, c0[0], c0[min(3, len(c0) - 1)]); end_label(DEST, c1[-1], c1[max(-4, -len(c1))])
ax.set_xlim(box[0], box[2]); ax.set_ylim(box[1], box[3]); ax.set_aspect(1); ax.axis("off")
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1)); cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.035, pad=0.01, shrink=0.8)
cb.set_label("Priority score (0 = low, 1 = high)", color=NAVY, fontsize=9); cb.outline.set_visible(False); cb.ax.tick_params(colors=NAVY, labelsize=8)
plt.tight_layout(); plt.savefig(OUT, facecolor=BG, bbox_inches="tight"); print("ok", RID)
