"""Scatter of panorama vs photo openness per segment (outputs of photo_check_v2.py)."""
import pathlib, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = pathlib.Path(__file__).resolve().parent / "outputs_v2"
c = pd.read_csv(OUT / "photo_vs_panorama_v2.csv").dropna(subset=["openness_photo"])
fig, ax = plt.subplots(figsize=(7.5, 7), dpi=200); fig.patch.set_facecolor("#FAF8F3"); ax.set_facecolor("#FAF8F3")
ax.plot([0, 1], [0, 1], color="#C3D7EA", lw=2, zorder=1)
o = c[c.route_id != "R05"]; r5 = c[c.route_id == "R05"]
ax.scatter(o.openness, o.openness_photo, s=70, color="#4173B0", alpha=.85, label="Other routes", zorder=2, edgecolor="none")
ax.scatter(r5.openness, r5.openness_photo, s=110, color="#EB8861", label="R05 (town centre)", zorder=3, edgecolor="#2B4C6D", lw=1)
ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
ax.set_xlabel("360° panorama openness (2022)", color="#2B4C6D", fontsize=13)
ax.set_ylabel("Street-photo openness (2017)", color="#2B4C6D", fontsize=13)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
for s in ["left", "bottom"]: ax.spines[s].set_color("#56708A")
ax.tick_params(colors="#3E5670", labelsize=11)
ax.legend(frameon=False, fontsize=12, labelcolor="#2B4C6D", loc="upper left")
ax.text(0.98, 0.04, "line = perfect agreement", ha="right", color="#56708A", fontsize=11, transform=ax.transAxes)
plt.tight_layout(); plt.savefig(OUT / "photo_vs_panorama_v2.png", facecolor="#FAF8F3"); print("ok")
