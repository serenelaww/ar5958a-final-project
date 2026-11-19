"""Check the 360 panorama openness against Vedika's SegFormer results on Shelly's Mapillary photos.
Inputs: V2/2./SegFormer_AMK_Results_R10/raw_class_masks (holds every route's masks), V2/1. photos/R*/R*_photos.csv.
Step 1 (resumable, run until all cached): per-photo class shares -> outputs_v2/photo_mask_stats.csv
Step 2 (`python3 photo_check_v2.py check`): openness = sky/(sky+tree) in the upper half of each photo; keep photos
<=20 m from the route with <10% vehicles; assign to segments by distance along the route; compare with panorama openness."""
import os, re, sys, time, glob, sqlite3, pathlib
import numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent; V2 = HERE.parent; OUT = HERE / "outputs_v2"
MASKS = V2 / "2./SegFormer_AMK_Results_R10/raw_class_masks"; CACHE = OUT / "photo_mask_stats.csv"
SKY = [2]; WATER = [21, 26, 60, 109, 128]   # QC rule: "water" overhead is overcast sky
TREE = [4, 17, 66, 72]; ROOF = [5, 61, 86, 106, 114]; BUILD = [0, 1, 25, 42, 48, 84]; VEH = [20, 80, 83, 102, 116, 127]

def step1(budget=150):
    t0 = time.time(); done = pd.read_csv(CACHE) if CACHE.exists() else pd.DataFrame()
    have = set(done.photo_no) if len(done) else set(); rows = []
    files = sorted(f for f in os.listdir(MASKS) if f.endswith(".npy"))
    for f in files:
        pno = f.split("_")[0]
        if pno in have: continue
        if time.time() - t0 > budget: break
        a = None
        for k in range(5):
            try: a = np.load(MASKS / f); break
            except OSError: time.sleep(2)
        if a is None: continue
        up = a[: a.shape[0] // 2]
        frac = lambda arr, ids: float(np.isin(arr, ids).mean())
        rows.append({"photo_no": pno, "route": re.search(r"_(R\d\d)_", f).group(1), "mask_file": f,
                     "sky_up": frac(up, SKY + WATER), "tree_up": frac(up, TREE), "build_up": frac(up, BUILD),
                     "roof_up": frac(up, ROOF), "veh_all": frac(a, VEH)})
        have.add(pno)
    if rows: pd.concat([done, pd.DataFrame(rows)]).to_csv(CACHE, index=False)
    print(f"cached {len(have)} of {len({f.split('_')[0] for f in files})}")

def step2():
    st = pd.read_csv(CACHE)
    meta = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(V2 / "1. photos/R*/R*_photos.csv")))])
    st = st.merge(meta[["photo_no", "dist_to_route_m", "along_route_m", "view_vs_route", "capture_date"]], on="photo_no", how="left")
    st["openness_photo"] = st.sky_up / (st.sky_up + st.tree_up).replace(0, np.nan)
    st["usable"] = (st.dist_to_route_m <= 20) & (st.veh_all < 0.10) & st.openness_photo.notna()
    con = sqlite3.connect(V2 / "3. Shade_by_Hour_Complete/data/routes/senior_routes.gpkg")
    seg = pd.read_sql("select seg_id, route_id, seq, length_m from route_segments order by route_id, seq", con)
    seg["end_m"] = seg.groupby("route_id").length_m.cumsum(); seg["start_m"] = seg.end_m - seg.length_m
    def find(r):
        s = seg[(seg.route_id == r.route) & (seg.start_m <= r.along_route_m) & (r.along_route_m < seg.end_m)]
        return s.seg_id.iloc[0] if len(s) else None
    st["seg_id"] = st.apply(find, axis=1); st.to_csv(OUT / "photo_points_v2.csv", index=False)
    u = st[st.usable & st.seg_id.notna()]
    ps = u.groupby("seg_id").agg(openness_photo=("openness_photo", "mean"), n_photos=("photo_no", "count")).reset_index()
    pri = pd.read_csv(OUT / "segment_priority_v2.csv")[["seg_id", "route_id", "openness", "rank"]]
    cmp = pri.merge(ps, on="seg_id", how="left"); cmp.to_csv(OUT / "photo_vs_panorama_v2.csv", index=False)
    c = cmp.dropna(subset=["openness_photo"])
    print("photos", len(st), "| usable", int(st.usable.sum()), "| segments with a usable photo", len(c), "of 134")
    print("overall r =", round(c.openness.corr(c.openness_photo), 2), "| spearman =", round(c.openness.rank().corr(c.openness_photo.rank()), 2))
    print(c.groupby("route_id").apply(lambda g: pd.Series({"n": len(g), "r": round(g.openness.corr(g.openness_photo), 2) if len(g) > 2 else np.nan,
          "pano": round(g.openness.mean(), 2), "photo": round(g.openness_photo.mean(), 2)}), include_groups=False).to_string())
    print("roof share max:", round(st.roof_up.max(), 3), "| usable views:", st[st.usable].view_vs_route.value_counts().to_dict(),
          "| dates:", st.capture_date.str[:4].value_counts().to_dict())
    top20 = set(pri.nsmallest(20, "rank").seg_id); t = c[c.seg_id.isin(top20)]
    print("top-20 with photo:", len(t), "photo openness", round(t.openness_photo.mean(), 2), "vs rest", round(c[~c.seg_id.isin(top20)].openness_photo.mean(), 2))

if __name__ == "__main__":
    step2() if "check" in sys.argv else step1()
