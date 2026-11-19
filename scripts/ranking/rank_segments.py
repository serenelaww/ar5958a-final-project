#!/usr/bin/env python3
"""Role 6 - Ranking + integration, v0 (pilot route R1: Blk 354 AMK St 32 -> 338 Family Clinic).

Joins every role's output onto one footpath-segment table, scores each component separately,
combines them with explicit weights, tests how sensitive the ranking is to those weights, and maps it.

Inputs (relative to the Final Project folder):
  data/route.json                                   route polyline (Google Routes walking path)
  streetview_pilot/streetview_pilot_metadata.csv    panorama locations + capture dates (Role 1)
  streetview_pilot/shade_outputs/*/shade_by_hour_all_four_days.csv   hourly shade state (Role 3)
  4. AMK/building.shp                               building footprints (map background)
  [pending] heat_wbgt.csv  (Role 5: segment_id, analysis_date, local_hour, wbgt_c)
  [pending] rain.csv       (Role 4: segment_id, rain_share, uncovered_m)

Status of components in v0:
  sun exposure  REAL   share of daylight walking-hour slots in direct sun (Street View + SegFormer + shade-by-hour)
  heat (WBGT)   PENDING Role 5 -> column left empty, weight re-normalised away
  rain          PENDING Role 4 -> column left empty, weight re-normalised away
  senior demand PENDING Role 4 destinations map; every segment on one route serves the same trip,
                so demand cannot vary along a single route anyway (needs the wider network).

QC rule added here (disclose on slides): a sun ray that lands on ADE20K water classes
(water/sea/river/lake/pool) above the horizon must be misread sky, so it is counted as SUNLIT and flagged.

Run: python3 "6. Ranking_Integration/rank_segments.py"
"""
import csv, glob, itertools, json, math, os, pathlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = pathlib.Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

SEG_LEN_M = 25          # ASSUMPTION: segment length along the route
MATCH_M = 15            # Role 1 rule: a panorama counts for a segment only within 15 m
WATER_IDS = {21, 26, 60, 109, 128}
WEIGHTS = {"sun": 0.4, "heat": 0.3, "rain": 0.3}   # ASSUMPTION: agree with the group / lead agency


def hav(a, b):
    R = 6371000; p1, p2 = math.radians(a[0]), math.radians(b[0])
    x = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(b[1] - a[1]) / 2) ** 2
    return 2 * R * math.asin(math.sqrt(x))


def split_route(line, step):
    """Cut the polyline into consecutive pieces of ~step metres; return (start, end, mid, length)."""
    pts = [tuple(line[0])]
    for a, b in zip(line, line[1:]):
        d = hav(a, b); n = max(1, int(d // 2))
        for i in range(1, n + 1):
            pts.append((a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n))
    segs, cur, acc = [], [pts[0]], 0.0
    for p, q in zip(pts, pts[1:]):
        acc += hav(p, q); cur.append(q)
        if acc >= step:
            segs.append((cur, acc)); cur, acc = [q], 0.0
    if acc > 0:
        segs.append((cur, acc))
    return segs


def point_to_seg_m(p, coords):
    return min(hav(p, c) for c in coords)


# ---------- 1. segments ----------
route = json.loads((ROOT / "data/route.json").read_text())["routes"][0]
pieces = split_route(route["polyline"], SEG_LEN_M)
segs = pd.DataFrame([{"segment_id": f"R1_S{i+1:02d}", "length_m": round(L, 1),
                      "coords": c, "mid_lat": c[len(c)//2][0], "mid_lon": c[len(c)//2][1],
                      "dist_from_start_m": None} for i, (c, L) in enumerate(pieces)])
segs["dist_from_start_m"] = segs.length_m.cumsum().shift(fill_value=0).round(0)

# ---------- 2. panoramas -> nearest segment ----------
meta = pd.read_csv(ROOT / "streetview_pilot/streetview_pilot_metadata.csv")
rows = []
for _, m in meta.iterrows():
    d = segs.coords.apply(lambda c: point_to_seg_m((m.latitude, m.longitude), c))
    rows.append({"pano": m.segment_id, "nearest_segment": segs.segment_id[d.idxmin()],
                 "dist_to_route_m": round(d.min(), 1), "capture_date": m.capture_date})
pano = pd.DataFrame(rows)
pano["used"] = pano.dist_to_route_m <= MATCH_M

# ---------- 3. shade per panorama (Role 3 output) with water QC ----------
def shade_summary(pid):
    f = glob.glob(str(ROOT / f"streetview_pilot/shade_outputs/{pid}__*/shade_by_hour_all_four_days.csv"))[0]
    df = pd.read_csv(f, encoding="utf-8-sig")
    df["qc_water_as_sky"] = df.ray_class_id.isin(WATER_IDS) & (df.sun_elevation_deg > 0)
    df.loc[df.qc_water_as_sky, "shade_state"] = "SUNLIT"
    day = df[~df.shade_state.isin(["NIGHT", "UNKNOWN"])]
    am, pm = day[day.local_hour <= 12], day[day.local_hour > 12]
    return {"pano": pid, "n_slots": len(day),
            "sun_share": (day.shade_state == "SUNLIT").mean(),
            "sun_share_am": (am.shade_state == "SUNLIT").mean(),
            "sun_share_pm": (pm.shade_state == "SUNLIT").mean(),
            "tree_share": (day.shade_state == "SHADED_TREE").mean(),
            "building_share": (day.shade_state == "SHADED_SOLID").mean(),
            "qc_water_fixed": int(df.qc_water_as_sky.sum())}

shade = pd.DataFrame([shade_summary(p) for p in pano.pano])
pano = pano.merge(shade, on="pano")
pano.to_csv(OUT / "panorama_qc.csv", index=False)

# ---------- 4. segment table ----------
used = pano[pano.used]
agg = used.groupby("nearest_segment").agg(
    n_panos=("pano", "count"), sun_share=("sun_share", "mean"), sun_share_am=("sun_share_am", "mean"),
    sun_share_pm=("sun_share_pm", "mean"), tree_share=("tree_share", "mean"),
    building_share=("building_share", "mean"), oldest_image=("capture_date", "min"),
    qc_water_fixed=("qc_water_fixed", "sum")).reset_index().rename(columns={"nearest_segment": "segment_id"})
seg = segs.drop(columns="coords").merge(agg, on="segment_id", how="left")
seg["n_panos"] = seg.n_panos.fillna(0).astype(int)
seg["wbgt_mean_c"] = np.nan          # PENDING Role 5
seg["rain_exposure"] = np.nan        # PENDING Role 4
seg["data_flag"] = np.select(
    [seg.n_panos == 0, seg.qc_water_fixed > 0, seg.oldest_image < "2023"],
    ["NO IMAGERY - site check", "sky misread as water, corrected", "imagery before 2023 - check trees"], "")


def norm(s):
    r = s.max() - s.min()
    return (s - s.min()) / r if r > 0 else s * 0


def rank(df, w):
    have = {k: v for k, v in w.items() if k == "sun" or df[{"heat": "wbgt_mean_c", "rain": "rain_exposure"}[k]].notna().any()}
    tot = sum(have.values())
    score = 0
    if "sun" in have: score = score + have["sun"] / tot * norm(df.sun_share)
    if "heat" in have: score = score + have["heat"] / tot * norm(df.wbgt_mean_c)
    if "rain" in have: score = score + have["rain"] / tot * norm(df.rain_exposure)
    return score, have

scored = seg[seg.n_panos > 0].copy()
scored["priority_score"], used_w = rank(scored, WEIGHTS)
scored["rank"] = scored.priority_score.rank(ascending=False, method="min").astype(int)
seg = seg.merge(scored[["segment_id", "priority_score", "rank"]], on="segment_id", how="left")
seg.to_csv(OUT / "segment_ranking_v0.csv", index=False)

# ---------- 5. sensitivity: AM vs PM weighting of sun (the only real component so far) ----------
sens = []
for w_am in np.linspace(0, 1, 11):
    s = w_am * norm(scored.sun_share_am) + (1 - w_am) * norm(scored.sun_share_pm)
    top = scored.segment_id[s.idxmax()]
    sens.append({"weight_on_morning": round(w_am, 1), "top_segment": top,
                 "top3": ", ".join(scored.assign(s=s).nlargest(3, "s").segment_id)})
pd.DataFrame(sens).to_csv(OUT / "sensitivity_am_pm.csv", index=False)

# ---------- 6. map ----------
import shapefile
lats = [p[0] for p in route["polyline"]]; lons = [p[1] for p in route["polyline"]]
pad = 0.0012
bb = (min(lons) - pad, min(lats) - pad, max(lons) + pad, max(lats) + pad)
fig, ax = plt.subplots(figsize=(10, 7.5))
for shp in shapefile.Reader(str(ROOT / "4. AMK/building")).shapes():
    x0, y0, x1, y1 = shp.bbox
    if x1 < bb[0] or x0 > bb[2] or y1 < bb[1] or y0 > bb[3]: continue
    parts = list(shp.parts) + [len(shp.points)]
    for a, b in zip(parts, parts[1:]):
        xs, ys = zip(*shp.points[a:b]); ax.fill(xs, ys, color="#d9d9d9", lw=0.3, ec="#bdbdbd")
cmap = plt.get_cmap("YlOrRd")
for (c, _), (_, r) in zip(pieces, seg.iterrows()):
    ys, xs = zip(*c)
    if r.n_panos == 0:
        ax.plot(xs, ys, color="#555", lw=3, ls=(0, (2, 2)))
    else:
        ax.plot(xs, ys, color=cmap(0.15 + 0.85 * r.sun_share / max(0.01, seg.sun_share.max())), lw=6, solid_capstyle="round")
        ax.annotate(f"#{int(r['rank'])}  {r.sun_share:.0%} sun", (r.mid_lon, r.mid_lat), xytext=(6, 8),
                    textcoords="offset points", fontsize=8.5, weight="bold" if r["rank"] <= 3 else "normal")
for _, p in pano.iterrows():
    m = meta[meta.segment_id == p.pano].iloc[0]
    ax.plot(m.longitude, m.latitude, marker="o" if p.used else "x", ms=5, color="#1f4e79" if p.used else "#c00000")
ax.plot(route["polyline"][0][1], route["polyline"][0][0], "s", color="k"); ax.annotate("Blk 354 (start)", (route["polyline"][0][1], route["polyline"][0][0]), xytext=(6, -12), textcoords="offset points", fontsize=8)
ax.plot(route["polyline"][-1][1], route["polyline"][-1][0], "*", color="k", ms=11); ax.annotate("338 Family Clinic", (route["polyline"][-1][1], route["polyline"][-1][0]), xytext=(6, 6), textcoords="offset points", fontsize=8)
ax.set_xlim(bb[0], bb[2]); ax.set_ylim(bb[1], bb[3]); ax.set_aspect(1 / math.cos(math.radians(1.364)))
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("Pilot route R1: share of senior walking hours in direct sun, by 25 m segment (v0)\n"
             "Sun exposure only; heat (WBGT) and rain layers pending. Dashed = no Street View panorama within 15 m.", fontsize=10)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], color=cmap(0.95), lw=6, label="more direct sun"), Line2D([], [], color=cmap(0.2), lw=6, label="less direct sun"),
                   Line2D([], [], color="#555", lw=3, ls=(0, (2, 2)), label="no imagery"),
                   Line2D([], [], marker="o", color="#1f4e79", ls="", label="panorama used"), Line2D([], [], marker="x", color="#c00000", ls="", label="panorama >15 m off route (dropped)")],
          loc="lower left", fontsize=8, frameon=True)
fig.tight_layout(); fig.savefig(OUT / "priority_map_v0.png", dpi=160)

print("weights used:", used_w)
print(pano[["pano", "nearest_segment", "dist_to_route_m", "used", "capture_date", "sun_share", "qc_water_fixed"]].round(2).to_string(index=False))
print(seg[["segment_id", "dist_from_start_m", "length_m", "n_panos", "sun_share", "sun_share_am", "sun_share_pm", "tree_share", "building_share", "rank", "data_flag"]].round(2).to_string(index=False))
print(pd.DataFrame(sens).to_string(index=False))
