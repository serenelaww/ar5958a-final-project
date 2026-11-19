"""Town-wide priority map for integrate_v2.py outputs."""
import pathlib, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from shapely import wkt
HERE = pathlib.Path(__file__).resolve().parent; OUT = HERE / "outputs_v2"
seg = pd.read_csv(OUT / "segment_priority_v2.csv").merge(pd.read_csv(OUT / "segment_geometry_wkt.csv"), on="seg_id")
seg["g"] = seg.wkt.apply(wkt.loads)
cmap = LinearSegmentedColormap.from_list("p", ["#4173B0", "#C3D7EA", "#EDCE7A", "#EB8861"])
fig, ax = plt.subplots(figsize=(9, 12), dpi=200); fig.patch.set_facecolor("#FAF8F3"); ax.set_facecolor("#FAF8F3")
minx, miny, maxx, maxy = [f(seg.g.apply(lambda g: g.bounds[i])) for i, f in zip(range(4), [min, min, max, max])]
pad = 0.003; box = (minx - pad, miny - pad, maxx + pad, maxy + pad)
try:
    import shapefile
    r = shapefile.Reader(str(HERE / "inputs/building.shp"))
    for sh in r.iterShapes():
        b = sh.bbox
        if b[2] < box[0] or b[0] > box[2] or b[3] < box[1] or b[1] > box[3]: continue
        pts = sh.points; parts = list(sh.parts) + [len(pts)]
        for i in range(len(parts) - 1):
            xs, ys = zip(*pts[parts[i]:parts[i + 1]]); ax.fill(xs, ys, color="#E4E0D6", lw=0)
except Exception as e:
    print("no buildings:", e)
top = seg[seg["rank"] <= 20]
for _, s in top.iterrows():
    x, y = s.g.xy; ax.plot(x, y, color="#2B4C6D", lw=7.5, solid_capstyle="round", zorder=2)
for _, s in seg.iterrows():
    x, y = s.g.xy; ax.plot(x, y, color=cmap(s.priority), lw=4.2, solid_capstyle="round", zorder=3)
for rid, g in seg.groupby("route_id"):
    c = g.loc[g.seq.idxmin()].g.coords[0]
    ax.annotate(rid, c, fontsize=10, fontweight="bold", color="#2B4C6D", xytext=(6, 6), textcoords="offset points", zorder=4,
                bbox=dict(boxstyle="round,pad=0.2", fc="#FAF8F3", ec="none", alpha=0.85))
ax.set_xlim(box[0], box[2]); ax.set_ylim(box[1], box[3]); ax.set_aspect(1 / 0.99998); ax.axis("off")
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.03, pad=0.01, shrink=0.6)
cb.set_label("Priority score (0.7 openness + 0.3 rain)  ·  dark outline = top 20 segments", color="#2B4C6D", fontsize=10)
cb.ax.tick_params(colors="#2B4C6D", labelsize=9); cb.outline.set_visible(False)
plt.tight_layout(); plt.savefig(OUT / "priority_map_v2.png", facecolor="#FAF8F3", bbox_inches="tight")
print("ok", box)
