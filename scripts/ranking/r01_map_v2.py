"""R01 street-photo openness map from the full photo run (photo_check_v2.py outputs)."""
import pathlib, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from shapely import wkt
import shapefile
HERE = pathlib.Path(__file__).resolve().parent; OUT = HERE / "outputs_v2"
g = pd.read_csv(OUT / "segment_geometry_wkt.csv"); c = pd.read_csv(OUT / "photo_vs_panorama_v2.csv")
p = pd.read_csv(OUT / "photo_points_v2.csv"); p = p[(p.route == "R01")]
d = g.merge(c, on="seg_id"); d = d[d.route_id == "R01"].copy(); d["geom"] = d.wkt.apply(wkt.loads)
print("R01 photos", len(p), "usable", int(p.usable.sum()), "segments with photo", d.openness_photo.notna().sum())
print(d[["seg_id", "openness_photo", "n_photos"]].round(2).to_string(index=False))
cmap = LinearSegmentedColormap.from_list("o", ["#4173B0", "#C3D7EA", "#EDCE7A", "#EB8861"])
fig, ax = plt.subplots(figsize=(6, 8), dpi=200); fig.patch.set_facecolor("#FAF8F3"); ax.set_facecolor("#FAF8F3")
b = d.geom.apply(lambda x: x.bounds); pad = 0.0012
box = (min(v[0] for v in b) - pad, min(v[1] for v in b) - pad, max(v[2] for v in b) + pad, max(v[3] for v in b) + pad)
for sh in shapefile.Reader(str(HERE / "inputs/building.shp")).iterShapes():
    bb = sh.bbox
    if bb[2] < box[0] or bb[0] > box[2] or bb[3] < box[1] or bb[1] > box[3]: continue
    pts = sh.points; parts = list(sh.parts) + [len(pts)]
    for i in range(len(parts) - 1):
        xs, ys = zip(*pts[parts[i]:parts[i + 1]]); ax.fill(xs, ys, color="#E4E0D6", lw=0)
for _, s in d.iterrows():
    x, y = s.geom.xy; ax.plot(x, y, color=cmap(s.openness_photo), lw=7, solid_capstyle="round")
    m = s.geom.interpolate(0.5, normalized=True)
    ax.annotate(s.seg_id.split("-")[1], (m.x, m.y), xytext=(8, 0), textcoords="offset points", fontsize=8, color="#2B4C6D", va="center")
s0 = d.sort_values("seg_id").iloc[0].geom.coords[0]; s1 = d.sort_values("seg_id").iloc[-1].geom.coords[-1]
ax.annotate("Blk 612", s0, xytext=(8, -12), textcoords="offset points", fontsize=10, fontweight="bold", color="#2B4C6D")
ax.annotate("AMK Community\nHospital", s1, xytext=(8, 4), textcoords="offset points", fontsize=10, fontweight="bold", color="#2B4C6D")
ax.set_xlim(box[0], box[2]); ax.set_ylim(box[1], box[3]); ax.set_aspect(1); ax.axis("off")
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1)); cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.035, pad=0.01, shrink=0.8)
cb.set_label("Street-photo openness (0 = under trees, 1 = open sky)", color="#2B4C6D", fontsize=9); cb.outline.set_visible(False); cb.ax.tick_params(colors="#2B4C6D", labelsize=8)
plt.tight_layout(); plt.savefig(OUT / "R01_photo_openness_v2.png", facecolor="#FAF8F3", bbox_inches="tight"); print("ok")
