"""Per-segment rain exposure table (Flora's ArcGIS layout, rebuilt): writes V2/4. Rain Exposure-1109/AMK_Rain_Exposure_by_segment.csv.
rain_score = rain_freq x uncovered_m (Flora's formula, grows with length); rain_per_m = rain_freq x uncovered share (used in ranking)."""
import pathlib, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent
s = pd.read_csv(HERE / "outputs_v2/segment_priority_v2.csv")
t = s[["seg_id", "route_id", "seq", "length_m", "covered_m", "uncovered_m", "shelter_source", "nearest_station",
       "nearest_station_name", "nearest_station_m", "rain_freq", "rain_per_m"]].copy()
t["rain_score_flora"] = (t.rain_freq * t.uncovered_m).round(3)
t.sort_values(["route_id", "seq"]).to_csv(HERE.parent / "4. Rain Exposure-1109/AMK_Rain_Exposure_by_segment.csv", index=False)
print(len(t), "rows; corr(rain_score_flora, length) =", round(t.rain_score_flora.corr(t.length_m), 3))
