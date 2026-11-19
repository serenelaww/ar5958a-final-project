#!/usr/bin/env python3
"""Role 6 - final integration (V2, 11 Nov 2026). Run: python3 "V2/6. Ranking_Integration/integrate_v2.py" (paths are relative to this file)

Joins on seg_id for all 134 segments of routes R01-R10:
  HEAT PROXY  sun_exposure = openness x (1 - covered share); openness = sky / (sky + tree) view factors from Sun Yu's KartaView 360 panoramas (V2/3. Shade_by_Hour_Complete).
              Higher = more open sky = more sun exposure. NOT a temperature. No hourly sun claim (shade-by-hour was set aside).
  RAIN        walking-hour rain frequency per NEA station (Flora's NEA_Stations_ArcGIS.csv), stations without coordinates in her file are left out
              (none are near AMK), inverse-distance weighted
              (3 nearest stations) to each segment midpoint. Shelter: LTA Covered Linkway polygons (DataMall, Sep 2026)
              intersected with each segment (shelter_lta_v2.py) plus OSM covered=yes. HDB/Town Council walkways are not in
              the LTA layer, so "no LTA linkway" is flagged, not proven open. rain_per_m = rain_freq x uncovered share.
  CROSS-CHECK Tejas's one-day (5 Nov 2026) provisional direct-sun % per segment (V2/5. HEAT), joined but not scored.
  PRIORITY    w_heat*norm(openness) + w_rain*norm(rain_per_m), default 0.7/0.3; sensitivity over w_heat 0.5-1.0.
"""
import math, sqlite3, pathlib
import numpy as np, pandas as pd
from shapely import wkb

HERE = pathlib.Path(__file__).resolve().parent              # V2/6. Ranking_Integration
V2 = HERE.parent; OUT = HERE / "outputs_v2"
OUT.mkdir(exist_ok=True)
W_HEAT, W_RAIN = 0.7, 0.3   # ASSUMPTION: heat exposure weighted above rain; tested below


def gpkg_geom(blob):
    flags = blob[3]; env = (flags >> 1) & 0x07
    return wkb.loads(bytes(blob[8 + {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[env]:]))


def hav(lat1, lon1, lat2, lon2):
    R = 6371000; p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# ---- segments and routes ----
con = sqlite3.connect(V2 / "3. Shade_by_Hour_Complete/data/routes/senior_routes.gpkg")
seg = pd.read_sql("select geom, seg_id, route_id, seq, kind, length_m, highway, covered_osm, mid_lon, mid_lat from route_segments", con)
seg["geometry"] = seg.geom.apply(gpkg_geom); seg = seg.drop(columns="geom")
routes = pd.read_sql("select route_id, origin, destination, dest_type, length_m as route_length_m from senior_routes", con)

# ---- heat proxy: openness from 360 view factors ----
vf = pd.read_csv(V2 / "3. Shade_by_Hour_Complete/outputs/point_view_factors.csv")
man = pd.read_csv(V2 / "3. Shade_by_Hour_Complete/data/panorama_manifest.csv")
vf = vf.merge(man[["seg_id", "distance_to_segment_midpoint_m", "capture_date"]], left_on="segment_id", right_on="seg_id")
vf["openness"] = vf.sky_view_factor / (vf.sky_view_factor + vf.tree_view_factor)
seg = seg.merge(vf[["seg_id", "openness", "sky_view_factor", "tree_view_factor", "building_view_factor",
                    "unknown_view_factor", "distance_to_segment_midpoint_m", "qc_status"]], on="seg_id", how="left")

# ---- rain ----
st = pd.read_csv(V2 / "4. Rain Exposure-1109/NEA_Stations_ArcGIS.csv")
st = st.rename(columns={"location_latitude": "lat", "location_longitude": "lon"})  # V2 coordinates only (Flora's file)
st = st[(st.quality_flag == "OK") & st.lat.notna()].reset_index(drop=True)
rows = []
for _, s in seg.iterrows():
    d = hav(s.mid_lat, s.mid_lon, st.lat.values, st.lon.values); idx = np.argsort(d)[:3]
    w = 1 / np.maximum(d[idx], 50) ** 2
    rows.append({"seg_id": s.seg_id, "rain_freq": float((w * st.rain_freq.values[idx]).sum() / w.sum()),
                 "nearest_station": st.station_id[idx[0]], "nearest_station_name": st.station_name[idx[0]],
                 "nearest_station_m": round(float(d[idx[0]]))})
seg = seg.merge(pd.DataFrame(rows), on="seg_id")
# Shelter: LTA Covered Linkway layer (DataMall Sep 2026), from shelter_lta_v2.py (3 m tolerance); OSM covered=yes as a fallback.
lta = pd.read_csv(OUT / "segment_shelter_lta.csv")[["seg_id", "covered_share"]].rename(columns={"covered_share": "lta_share"})
seg = seg.merge(lta, on="seg_id", how="left")
share = np.where(seg.covered_osm == "yes", 1.0, seg.lta_share.fillna(0))
seg["covered_m"] = (share * seg.length_m).round(1)
seg["uncovered_m"] = seg.length_m - seg.covered_m
seg["shelter_source"] = np.where(seg.covered_osm == "yes", "OSM covered=yes",
                        np.where(seg.lta_share > 0, "LTA linkway (partial)", "No LTA linkway; HDB/Town Council shelters not mapped"))
seg["rain_per_m"] = seg.rain_freq * seg.uncovered_m / seg.length_m

# ---- cross-check: Tejas one-day sun % ----
tj = pd.read_csv(V2 / "5. HEAT/AMK_Heat_Handover_for_Serene.csv")[["segment_id", "provisional_solar_exposure_pct"]]
seg = seg.merge(tj.rename(columns={"segment_id": "seg_id", "provisional_solar_exposure_pct": "sun_pct_5nov"}), on="seg_id", how="left")

# ---- priority ----
def norm(x):
    return (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x * 0

# Roof fix: an existing linkway roof shades the walker too, but road-level panoramas cannot see it.
seg["covered_share"] = (seg.covered_m / seg.length_m).clip(0, 1)
seg["sun_exposure"] = seg.openness * (1 - seg.covered_share)
seg["heat_score"] = norm(seg.sun_exposure); seg["rain_score"] = norm(seg.rain_per_m)
# Linkway candidate: shaded by trees (openness < 0.35) but no cover - trees handle the sun, nothing keeps the rain off.
seg["linkway_candidate"] = (seg.openness < 0.35) & (seg.covered_m == 0)
seg["priority"] = W_HEAT * seg.heat_score + W_RAIN * seg.rain_score
seg["rank"] = seg.priority.rank(ascending=False, method="min").astype(int)
flags = []
for _, s in seg.iterrows():
    f = []
    if s.qc_status == "REVIEW_DISTANCE": f.append(f"panorama {s.distance_to_segment_midpoint_m:.0f} m away - site check")
    if s.covered_osm != "yes" and not s.lta_share > 0: f.append("no LTA linkway - check estate shelters")
    if s.length_m < 15: f.append("very short segment")
    if s.linkway_candidate: f.append("linkway candidate (shaded, uncovered)")
    flags.append("; ".join(f))
seg["data_flags"] = flags

# sensitivity: overlap of top 20 with the default across weights
top_def = set(seg.nsmallest(20, "rank").seg_id); sens = []
for wh in np.round(np.arange(0.5, 1.01, 0.1), 1):
    p = wh * seg.heat_score + (1 - wh) * seg.rain_score
    top = set(seg.assign(p=p).nlargest(20, "p").seg_id)
    sens.append({"w_heat": wh, "w_rain": round(1 - wh, 1), "top20_overlap_with_default": len(top & top_def),
                 "top_segment": seg.assign(p=p).nlargest(1, "p").seg_id.iloc[0]})
sens = pd.DataFrame(sens)

# ---- route summary (length-weighted) ----
def wmean(g, c): return float((g[c] * g.length_m).sum() / g.length_m.sum())
rs = seg.groupby("route_id").apply(lambda g: pd.Series({
    "segments": len(g), "length_m": g.length_m.sum(), "openness": wmean(g, "openness"),
    "rain_per_m": wmean(g, "rain_per_m"), "priority": wmean(g, "priority"), "linkway_candidates": int(g.linkway_candidate.sum()), "linkway_candidate_m": float(g.length_m[g.linkway_candidate].sum()),
    "top20_segments": int(g.seg_id.isin(top_def).sum()),
    "flagged_panorama": int((g.qc_status == "REVIEW_DISTANCE").sum()),
    "sun_pct_5nov": wmean(g.dropna(subset=["sun_pct_5nov"]), "sun_pct_5nov")}), include_groups=False).reset_index()
rs = rs.merge(routes, on="route_id").sort_values("priority", ascending=False)
rs["route_rank"] = range(1, len(rs) + 1)

cols = ["seg_id", "route_id", "seq", "kind", "highway", "length_m", "openness", "sky_view_factor", "tree_view_factor",
        "building_view_factor", "unknown_view_factor", "rain_freq", "nearest_station", "nearest_station_name",
        "nearest_station_m", "covered_m", "covered_share", "uncovered_m", "shelter_source", "rain_per_m", "sun_exposure", "linkway_candidate", "sun_pct_5nov",
        "heat_score", "rain_score", "priority", "rank", "qc_status", "distance_to_segment_midpoint_m", "data_flags", "mid_lat", "mid_lon"]
seg.sort_values("rank")[cols].round(4).to_csv(OUT / "segment_priority_v2.csv", index=False)
rs.round(4).to_csv(OUT / "route_priority_v2.csv", index=False)
sens.to_csv(OUT / "sensitivity_weights_v2.csv", index=False)
seg[["seg_id", "geometry"]].assign(wkt=seg.geometry.apply(lambda g: g.wkt)).drop(columns="geometry").to_csv(OUT / "segment_geometry_wkt.csv", index=False)

print(rs.round(3).to_string(index=False))
print(seg.sort_values("rank")[["seg_id", "kind", "openness", "rain_freq", "shelter_source", "sun_pct_5nov", "priority", "rank", "data_flags"]].head(12).round(3).to_string(index=False))
print(sens.to_string(index=False))
print("rain_freq range", seg.rain_freq.min().round(4), seg.rain_freq.max().round(4), "| nearest stations:", seg.nearest_station_name.value_counts().to_dict())
print("corr openness vs Tejas sun% :", round(seg.openness.corr(seg.sun_pct_5nov), 2))
print("corr openness vs rain_per_m :", round(seg.openness.corr(seg.rain_per_m), 2))
