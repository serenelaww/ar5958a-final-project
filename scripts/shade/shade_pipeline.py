from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import sqlite3
import ssl
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SKY_IDS = {2}
TREE_IDS = {4, 17, 66, 72}
ROOF_IDS = {5, 61, 86, 106, 114}
BUILDING_IDS = {0, 1, 25, 42, 48, 84}
WALKING_HOURS = [7, 8, 9, 10, 16, 17, 18, 19]
REPRESENTATIVE_DATES = ["2026-03-20", "2026-06-21", "2026-09-22", "2026-12-21"]
SSL_CONTEXT = ssl._create_unverified_context()


def category_for_class(class_id: int) -> str:
    if class_id == 255:
        return "unknown"
    if class_id in SKY_IDS:
        return "sky"
    if class_id in TREE_IDS:
        return "tree"
    if class_id in ROOF_IDS:
        return "roof"
    if class_id in BUILDING_IDS:
        return "building"
    return "other_solid"


def haversine_m(lat1, lon1, lat2, lon2):
    radius = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def read_segments(gpkg: Path):
    connection = sqlite3.connect(gpkg)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """
        SELECT seg_id, route_id, seq, kind, length_m, highway, street,
               covered_osm, mid_lon, mid_lat
        FROM route_segments
        ORDER BY route_id, seq
        """
    ).fetchall()
    connection.close()
    return [dict(row) for row in rows]


def api_post(url: str, fields: dict, attempts: int = 4):
    data = urllib.parse.urlencode(fields).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST")
    request.add_header("User-Agent", "AR5958A-ShadeByHour/1.0")
    last_error = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=90, context=SSL_CONTEXT) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            last_error = exc
            time.sleep(2 + attempt * 2)
    raise RuntimeError(f"KartaView request failed after {attempts} attempts: {last_error}")


def get_kartaview_candidates(lat: float, lon: float, radius_m: int):
    payload = api_post(
        "https://api.kartaview.org/1.0/list/nearby-photos/",
        {"lat": f"{lat:.7f}", "lng": f"{lon:.7f}", "radius": str(radius_m)},
    )
    items = payload.get("currentPageItems") or payload.get("result", {}).get("data") or []
    candidates = []
    for item in items:
        projection = str(item.get("projection", "")).upper()
        fov = float(item.get("field_of_view") or item.get("fieldOfView") or 0)
        if projection != "SPHERE" and fov < 350:
            continue
        try:
            item_lat = float(item.get("lat"))
            item_lon = float(item.get("lng"))
        except (TypeError, ValueError):
            continue
        name = item.get("name") or item.get("filepath")
        if not name:
            continue
        heading = item.get("heading")
        try:
            heading = float(heading)
        except (TypeError, ValueError):
            heading = None
        item = dict(item)
        item["distance_to_segment_midpoint_m"] = haversine_m(lat, lon, item_lat, item_lon)
        item["heading_numeric"] = heading
        item["source_path"] = name
        candidates.append(item)
    candidates.sort(key=lambda row: (row["distance_to_segment_midpoint_m"], -(float(row.get("timestamp") or 0))))
    return candidates


def wrapped_url(source_path: str):
    parts = source_path.replace("\\", "/").split("/")
    storage = parts[0]
    rest = "/".join(parts[1:])
    rest = rest.replace("/proc/", "/wrapped_proc/")
    return f"https://{storage}.openstreetcam.org/{rest}"


def download_bytes(url: str, attempts: int = 4):
    request = urllib.request.Request(url, headers={"User-Agent": "AR5958A-ShadeByHour/1.0"})
    last_error = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=180, context=SSL_CONTEXT) as response:
                return response.read()
        except Exception as exc:
            last_error = exc
            time.sleep(3 + attempt * 3)
    raise RuntimeError(f"Image download failed: {url}: {last_error}")


def save_north_aligned_panorama(image_bytes: bytes, heading_deg: float, output_path: Path):
    with Image.open(io.BytesIO(image_bytes)) as source:
        source = source.convert("RGB")
        if source.width / source.height < 1.95 or source.width / source.height > 2.05:
            raise ValueError(f"Wrapped panorama is not 2:1: {source.size}")
        resized = source.resize((4096, 2048), Image.Resampling.LANCZOS)
    north_at_left = (heading_deg - 180.0) % 360.0
    north_x = int(round(((0.0 - north_at_left) % 360.0) / 360.0 * resized.width)) % resized.width
    if north_x:
        aligned = Image.new("RGB", resized.size)
        aligned.paste(resized.crop((north_x, 0, resized.width, resized.height)), (0, 0))
        aligned.paste(resized.crop((0, 0, north_x, resized.height)), (resized.width - north_x, 0))
    else:
        aligned = resized
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aligned.save(output_path, quality=88, optimize=True, progressive=True)
    return north_at_left, north_x


def acquire_panoramas(segments, root: Path, radius_m: int = 160):
    panoramas_root = root / "data" / "panoramas"
    cache_root = root / "data" / "acquisition_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    used_by_route = defaultdict(set)
    manifest = []
    for index, segment in enumerate(segments, 1):
        seg_id = segment["seg_id"]
        route_id = segment["route_id"]
        cache_path = cache_root / f"{seg_id}.json"
        if cache_path.exists():
            candidates = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            try:
                candidates = get_kartaview_candidates(float(segment["mid_lat"]), float(segment["mid_lon"]), radius_m)
            except Exception as exc:
                candidates = []
                cache_path.write_text(json.dumps({"error": str(exc)}, indent=2), encoding="utf-8")
            else:
                cache_path.write_text(json.dumps(candidates, indent=2), encoding="utf-8")
        if isinstance(candidates, dict):
            candidates = []
        distinct = [c for c in candidates if str(c.get("id")) not in used_by_route[route_id]]
        ordered = distinct + [c for c in candidates if c not in distinct]
        selected = None
        error_messages = []
        for candidate in ordered[:12]:
            heading = candidate.get("heading_numeric")
            if heading is None:
                continue
            image_id = str(candidate.get("id"))
            output_path = panoramas_root / route_id / f"{seg_id}__KV{image_id}__north_aligned.jpg"
            if output_path.exists():
                selected = candidate
                selected["output_path"] = str(output_path.relative_to(root)).replace("\\", "/")
                selected["north_at_left_raw_deg"] = (heading - 180.0) % 360.0
                selected["horizontal_shift_px"] = "existing"
                break
            try:
                source_url = wrapped_url(candidate["source_path"])
                image_bytes = download_bytes(source_url)
                raw_north, shift_px = save_north_aligned_panorama(image_bytes, heading, output_path)
                selected = candidate
                selected["source_url"] = source_url
                selected["output_path"] = str(output_path.relative_to(root)).replace("\\", "/")
                selected["north_at_left_raw_deg"] = raw_north
                selected["horizontal_shift_px"] = shift_px
                break
            except Exception as exc:
                error_messages.append(f"{image_id}: {exc}")
        if selected:
            used_by_route[route_id].add(str(selected.get("id")))
            distance = float(selected["distance_to_segment_midpoint_m"])
            status = "PASS" if distance <= 25 else "REVIEW_DISTANCE"
            manifest.append(
                {
                    **segment,
                    "image_id": str(selected.get("id")),
                    "sequence_id": str(selected.get("sequence_id") or selected.get("sequenceId") or ""),
                    "image_lat": float(selected.get("lat")),
                    "image_lon": float(selected.get("lng")),
                    "distance_to_segment_midpoint_m": round(distance, 1),
                    "capture_date": selected.get("shot_date") or selected.get("shotDate") or "",
                    "heading_deg_center": selected.get("heading_numeric"),
                    "north_alignment_method": "KartaView SPHERE heading assumed at panorama centre; circular shift places north at left seam",
                    "heading_uncertainty_deg": 10.0,
                    "source": "KartaView",
                    "source_url": selected.get("source_url") or wrapped_url(selected["source_path"]),
                    "license_note": "KartaView public imagery; retain image ID, contributor and source URL for attribution",
                    "panorama_file": selected["output_path"],
                    "qc_status": status,
                    "qc_notes": "Heading has not been field-validated. " + ("" if status == "PASS" else "Panorama is more than 25 m from the segment midpoint."),
                }
            )
        else:
            manifest.append(
                {
                    **segment,
                    "image_id": "",
                    "sequence_id": "",
                    "image_lat": "",
                    "image_lon": "",
                    "distance_to_segment_midpoint_m": "",
                    "capture_date": "",
                    "heading_deg_center": "",
                    "north_alignment_method": "",
                    "heading_uncertainty_deg": "",
                    "source": "KartaView",
                    "source_url": "",
                    "license_note": "",
                    "panorama_file": "",
                    "qc_status": "MISSING_PANORAMA",
                    "qc_notes": " | ".join(error_messages[:3]) or "No SPHERE panorama with heading found within search radius.",
                }
            )
        print(f"Acquire {index:03d}/{len(segments)} {seg_id}: {manifest[-1]['qc_status']}", flush=True)
    write_csv(root / "data" / "panorama_manifest.csv", manifest)
    return manifest


def write_csv(path: Path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def create_onnx_session(model_path: Path, runtime_path: Path | None):
    if runtime_path:
        import sys

        sys.path.insert(0, str(runtime_path))
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, min(8, __import__("os").cpu_count() or 4))
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])


def segment_upper_hemisphere(session, panorama_path: Path):
    with Image.open(panorama_path) as image:
        image = image.convert("RGB").resize((2048, 1024), Image.Resampling.LANCZOS)
        upper = image.crop((0, 0, 2048, 512))
    upper_array = np.asarray(upper, dtype=np.uint8)
    observed = np.max(upper_array, axis=2) > 12
    tiles = []
    for index in range(4):
        tile = np.asarray(upper.crop((index * 512, 0, (index + 1) * 512, 512)), dtype=np.float32) / 255.0
        tile = (tile - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        tiles.append(np.transpose(tile, (2, 0, 1)))
    batch = np.stack(tiles).astype(np.float32)
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: batch})[0]
    labels = np.argmax(logits, axis=1).astype(np.uint8)
    full = Image.new("L", (2048, 512))
    for index in range(4):
        tile = Image.fromarray(labels[index]).resize((512, 512), Image.Resampling.NEAREST)
        full.paste(tile, (index * 512, 0))
    result = np.asarray(full, dtype=np.uint8).copy()
    result[~observed] = 255
    return result


def view_factors(mask: np.ndarray):
    height, width = mask.shape
    zenith = (np.arange(height) + 0.5) / height * 90.0
    elevation = 90.0 - zenith
    weights_by_row = np.cos(np.radians(elevation))
    weights = np.repeat(weights_by_row[:, None], width, axis=1)
    totals = Counter()
    for class_id in np.unique(mask):
        totals[category_for_class(int(class_id))] += float(weights[mask == class_id].sum())
    denominator = float(weights.sum())
    return {key: totals.get(key, 0.0) / denominator for key in ["sky", "tree", "building", "roof", "other_solid", "unknown"]}


def category_image(mask: np.ndarray):
    palette = {
        "sky": (80, 164, 224),
        "tree": (57, 139, 80),
        "building": (125, 132, 143),
        "roof": (116, 82, 66),
        "other_solid": (205, 205, 205),
        "unknown": (82, 65, 105),
    }
    rgb = np.empty((*mask.shape, 3), dtype=np.uint8)
    for class_id in np.unique(mask):
        rgb[mask == class_id] = palette[category_for_class(int(class_id))]
    return rgb


def save_fisheye(category_rgb: np.ndarray, output_path: Path):
    size = 900
    yy, xx = np.indices((size, size))
    center = (size - 1) / 2
    dx = xx - center
    dy = yy - center
    radius = np.sqrt(dx * dx + dy * dy)
    inside = radius <= center
    zenith = np.clip(radius / center * 90.0, 0, 90)
    azimuth = (np.degrees(np.arctan2(dx, -dy)) + 360.0) % 360.0
    source_x = np.floor(azimuth / 360.0 * category_rgb.shape[1]).astype(int) % category_rgb.shape[1]
    source_y = np.floor(zenith / 90.0 * category_rgb.shape[0]).astype(int)
    source_y = np.clip(source_y, 0, category_rgb.shape[0] - 1)
    sampled = category_rgb[source_y, source_x]
    output = np.full((size, size, 3), 248, dtype=np.uint8)
    output[inside] = sampled[inside]
    image = Image.fromarray(output)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((size // 2, 8), "N", fill="black", anchor="ma", font=font)
    draw.text((size - 8, size // 2), "E", fill="black", anchor="rm", font=font)
    draw.text((size // 2, size - 8), "S", fill="black", anchor="ms", font=font)
    draw.text((8, size // 2), "W", fill="black", anchor="lm", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def solar_position(local_dt: datetime, latitude: float, longitude: float, utc_offset: float = 8.0):
    days = 366 if local_dt.year % 4 == 0 and (local_dt.year % 100 != 0 or local_dt.year % 400 == 0) else 365
    hour = local_dt.hour + local_dt.minute / 60.0
    gamma = 2 * math.pi / days * (local_dt.timetuple().tm_yday - 1 + (hour - 12) / 24.0)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma)
        - 0.040849 * math.sin(2 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma)
        + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma)
        + 0.00148 * math.sin(3 * gamma)
    )
    time_offset = eqtime + 4 * longitude - 60 * utc_offset
    true_solar_minutes = (hour * 60 + time_offset) % 1440
    hour_angle = math.radians(true_solar_minutes / 4 - 180)
    lat = math.radians(latitude)
    cos_zenith = math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(decl) * math.cos(hour_angle)
    zenith = math.acos(max(-1.0, min(1.0, cos_zenith)))
    elevation = 90 - math.degrees(zenith)
    azimuth = (
        math.degrees(
            math.atan2(
                math.sin(hour_angle),
                math.cos(hour_angle) * math.sin(lat) - math.tan(decl) * math.cos(lat),
            )
        )
        + 180
    ) % 360
    return elevation, azimuth


def sample_sun_ray(mask: np.ndarray, azimuth: float, elevation: float, radius: int = 3):
    if elevation <= 0:
        return "NIGHT", "night", "", "", ""
    height, width = mask.shape
    x = int(round((azimuth % 360.0) / 360.0 * width)) % width
    y = int(round((90.0 - elevation) / 90.0 * (height - 1)))
    y = max(0, min(height - 1, y))
    xs = [(x + offset) % width for offset in range(-radius, radius + 1)]
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    categories = []
    for yy in range(y0, y1):
        for xx in xs:
            categories.append(category_for_class(int(mask[yy, xx])))
    category = Counter(categories).most_common(1)[0][0]
    if category == "unknown":
        state = "UNKNOWN"
    elif category == "sky":
        state = "SUNLIT"
    elif category == "tree":
        state = "SHADED_TREE"
    else:
        state = "SHADED_SOLID"
    return state, category, int(mask[y, x]), x, y


def dates_in_year(year: int):
    current = date(year, 1, 1)
    end = date(year + 1, 1, 1)
    while current < end:
        yield current
        current += timedelta(days=1)


def analyse(root: Path, manifest, model_path: Path, runtime_path: Path | None, year: int):
    session = create_onnx_session(model_path, runtime_path)
    mask_root = root / "data" / "masks"
    overlay_root = root / "figures" / "fisheye"
    representative_rows = []
    view_rows = []
    segment_hour_counts = defaultdict(Counter)
    all_segment_summary = []
    daily_gzip = root / "outputs" / "segment_shade_daily_hour.csv.gz"
    daily_gzip.parent.mkdir(parents=True, exist_ok=True)
    daily_fields = [
        "route_id", "segment_id", "image_id", "analysis_date", "local_hour",
        "sun_elevation_deg", "sun_azimuth_deg", "shade_state", "ray_category",
        "ray_class_id", "ray_pixel_x", "ray_pixel_y", "qc_status",
    ]
    with gzip.open(daily_gzip, "wt", newline="", encoding="utf-8") as gz_handle:
        daily_writer = csv.DictWriter(gz_handle, fieldnames=daily_fields)
        daily_writer.writeheader()
        valid_manifest = [row for row in manifest if row.get("panorama_file")]
        for index, row in enumerate(valid_manifest, 1):
            seg_id = row["seg_id"]
            route_id = row["route_id"]
            image_id = row["image_id"]
            panorama_path = root / row["panorama_file"]
            mask_path = mask_root / route_id / f"{seg_id}__KV{image_id}__upper_classes.png"
            if mask_path.exists():
                mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
            else:
                mask = segment_upper_hemisphere(session, panorama_path)
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(mask, mode="L").save(mask_path, optimize=True)
            category_rgb = category_image(mask)
            overlay_path = overlay_root / route_id / f"{seg_id}__KV{image_id}__fisheye.png"
            if not overlay_path.exists():
                save_fisheye(category_rgb, overlay_path)
            factors = view_factors(mask)
            view_row = {
                "route_id": route_id,
                "segment_id": seg_id,
                "image_id": image_id,
                "segment_length_m": round(float(row["length_m"]), 2),
                "latitude": row["image_lat"],
                "longitude": row["image_lon"],
                "sky_view_factor": round(factors["sky"], 6),
                "tree_view_factor": round(factors["tree"], 6),
                "building_view_factor": round(factors["building"], 6),
                "roof_view_factor": round(factors["roof"], 6),
                "other_solid_view_factor": round(factors["other_solid"], 6),
                "unknown_view_factor": round(factors["unknown"], 6),
                "mask_file": str(mask_path.relative_to(root)).replace("\\", "/"),
                "fisheye_file": str(overlay_path.relative_to(root)).replace("\\", "/"),
                "qc_status": row["qc_status"],
            }
            view_rows.append(view_row)
            latitude = float(row["image_lat"])
            longitude = float(row["image_lon"])
            rep_set = set(REPRESENTATIVE_DATES)
            for day in dates_in_year(year):
                date_text = day.isoformat()
                for hour in WALKING_HOURS:
                    local_dt = datetime(day.year, day.month, day.day, hour)
                    elevation, azimuth = solar_position(local_dt, latitude, longitude)
                    state, category, class_id, x, y = sample_sun_ray(mask, azimuth, elevation)
                    output_row = {
                        "route_id": route_id,
                        "segment_id": seg_id,
                        "image_id": image_id,
                        "analysis_date": date_text,
                        "local_hour": hour,
                        "sun_elevation_deg": round(elevation, 3),
                        "sun_azimuth_deg": round(azimuth, 3),
                        "shade_state": state,
                        "ray_category": category,
                        "ray_class_id": class_id,
                        "ray_pixel_x": x,
                        "ray_pixel_y": y,
                        "qc_status": row["qc_status"],
                    }
                    daily_writer.writerow(output_row)
                    if date_text in rep_set:
                        representative_rows.append({**output_row, **{key: view_row[key] for key in ["sky_view_factor", "tree_view_factor", "building_view_factor", "roof_view_factor"]}})
                    if state != "NIGHT":
                        segment_hour_counts[(seg_id, hour)]["solar_above_horizon"] += 1
                        if state == "UNKNOWN":
                            segment_hour_counts[(seg_id, hour)]["unknown"] += 1
                        else:
                            segment_hour_counts[(seg_id, hour)]["valid"] += 1
                            segment_hour_counts[(seg_id, hour)][state] += 1
            print(f"Analyse {index:03d}/{len(valid_manifest)} {seg_id}", flush=True)
    write_csv(root / "outputs" / "point_view_factors.csv", view_rows)
    write_csv(root / "outputs" / "segment_shade_by_hour_representative_days.csv", representative_rows)

    view_by_segment = {row["segment_id"]: row for row in view_rows}
    manifest_by_segment = {row["seg_id"]: row for row in manifest}
    for segment in manifest:
        seg_id = segment["seg_id"]
        for hour in WALKING_HOURS:
            counts = segment_hour_counts.get((seg_id, hour), Counter())
            daylight = counts.get("solar_above_horizon", 0)
            valid = counts.get("valid", 0)
            unknown = counts.get("unknown", 0)
            shaded = counts.get("SHADED_TREE", 0) + counts.get("SHADED_SOLID", 0)
            summary = {
                "route_id": segment["route_id"],
                "segment_id": seg_id,
                "local_hour": hour,
                "segment_length_m": round(float(segment["length_m"]), 2),
                "solar_above_horizon_days": daylight,
                "valid_days": valid,
                "unknown_days": unknown,
                "valid_day_fraction": round(valid / daylight, 6) if daylight else "",
                "sunlit_days": counts.get("SUNLIT", 0),
                "tree_shade_days": counts.get("SHADED_TREE", 0),
                "solid_shade_days": counts.get("SHADED_SOLID", 0),
                "annual_shade_fraction": round(shaded / valid, 6) if valid else "",
                "annual_tree_shade_fraction": round(counts.get("SHADED_TREE", 0) / valid, 6) if valid else "",
                "annual_solid_shade_fraction": round(counts.get("SHADED_SOLID", 0) / valid, 6) if valid else "",
                "valid_geometry": 1 if seg_id in view_by_segment else 0,
                "qc_status": manifest_by_segment[seg_id]["qc_status"],
            }
            if seg_id in view_by_segment:
                summary.update({key: view_by_segment[seg_id][key] for key in ["sky_view_factor", "tree_view_factor", "building_view_factor", "roof_view_factor", "unknown_view_factor"]})
            else:
                summary.update({key: "" for key in ["sky_view_factor", "tree_view_factor", "building_view_factor", "roof_view_factor", "unknown_view_factor"]})
            all_segment_summary.append(summary)
    write_csv(root / "outputs" / "segment_annual_hour_summary.csv", all_segment_summary)

    route_hour_rows = []
    route_summary_rows = []
    by_route_hour = defaultdict(list)
    for row in all_segment_summary:
        by_route_hour[(row["route_id"], int(row["local_hour"]))].append(row)
    routes = sorted({row["route_id"] for row in all_segment_summary})
    for route_id in routes:
        route_hours = []
        for hour in WALKING_HOURS:
            rows = by_route_hour[(route_id, hour)]
            total_length = sum(float(row["segment_length_m"]) for row in rows)
            valid_rows = [row for row in rows if row["annual_shade_fraction"] != "" and row["valid_day_fraction"] != ""]
            valid_length = sum(float(row["segment_length_m"]) * float(row["valid_day_fraction"]) for row in valid_rows)
            def weighted(field):
                return sum(float(row["segment_length_m"]) * float(row["valid_day_fraction"]) * float(row[field]) for row in valid_rows) / valid_length if valid_length else None
            shade = weighted("annual_shade_fraction")
            tree = weighted("annual_tree_shade_fraction")
            solid = weighted("annual_solid_shade_fraction")
            output = {
                "route_id": route_id,
                "local_hour": hour,
                "route_length_m": round(total_length, 2),
                "valid_length_m": round(valid_length, 2),
                "valid_coverage_fraction": round(valid_length / total_length, 6) if total_length else 0,
                "annual_shade_fraction": round(shade, 6) if shade is not None else "",
                "annual_tree_shade_fraction": round(tree, 6) if tree is not None else "",
                "annual_solid_shade_fraction": round(solid, 6) if solid is not None else "",
                "result_status": "PROVISIONAL_PARTIAL_VERTICAL_FOV_AND_HEADING_UNVALIDATED" if valid_length else "NO_RESULT",
            }
            route_hour_rows.append(output)
            if shade is not None:
                route_hours.append(output)
        if route_hours:
            route_summary_rows.append(
                {
                    "route_id": route_id,
                    "annual_walking_hour_shade_fraction": round(sum(float(r["annual_shade_fraction"]) for r in route_hours) / len(route_hours), 6),
                    "morning_shade_fraction": round(sum(float(r["annual_shade_fraction"]) for r in route_hours if int(r["local_hour"]) <= 10) / 4, 6),
                    "evening_shade_fraction": round(sum(float(r["annual_shade_fraction"]) for r in route_hours if int(r["local_hour"]) >= 16) / 4, 6),
                    "minimum_valid_coverage_fraction": min(float(r["valid_coverage_fraction"]) for r in route_hours),
                    "result_status": "PROVISIONAL_PARTIAL_VERTICAL_FOV_AND_HEADING_UNVALIDATED",
                }
            )
    write_csv(root / "outputs" / "route_annual_hour_summary.csv", route_hour_rows)
    write_csv(root / "outputs" / "route_annual_summary.csv", route_summary_rows)
    build_heat_handover(root, all_segment_summary, manifest, view_by_segment)
    build_measurement_plan(root, all_segment_summary, manifest)
    draw_route_chart(root, route_hour_rows)
    draw_sample_chart(root, representative_rows)
    return route_summary_rows


def build_heat_handover(root, segment_hour_rows, manifest, view_by_segment):
    manifest_by_segment = {row["seg_id"]: row for row in manifest}
    rows = []
    for row in segment_hour_rows:
        meta = manifest_by_segment[row["segment_id"]]
        rows.append(
            {
                "route_id": row["route_id"],
                "segment_id": row["segment_id"],
                "local_hour": row["local_hour"],
                "segment_length_m": row["segment_length_m"],
                "segment_mid_lat": meta["mid_lat"],
                "segment_mid_lon": meta["mid_lon"],
                "panorama_lat": meta.get("image_lat", ""),
                "panorama_lon": meta.get("image_lon", ""),
                "annual_shade_fraction": row["annual_shade_fraction"],
                "annual_tree_shade_fraction": row["annual_tree_shade_fraction"],
                "annual_solid_shade_fraction": row["annual_solid_shade_fraction"],
                "valid_day_fraction": row["valid_day_fraction"],
                "sky_view_factor": row["sky_view_factor"],
                "tree_view_factor": row["tree_view_factor"],
                "building_view_factor": row["building_view_factor"],
                "roof_view_factor": row["roof_view_factor"],
                "unknown_view_factor": row["unknown_view_factor"],
                "recommended_heat_model_use": "Use shade fraction or date-specific shade_state as the solar-exposure predictor; retain QC status.",
                "shade_result_status": row["qc_status"],
                "heading_validation_status": "NOT_FIELD_VALIDATED",
                "vertical_field_of_view_status": "PARTIAL_SPHERE_UNOBSERVED_PIXELS_EXCLUDED",
            }
        )
    write_csv(root / "handover_to_heat_model" / "shade_predictors_by_segment_hour.csv", rows)


def build_measurement_plan(root, segment_hour_rows, manifest):
    meta = {row["seg_id"]: row for row in manifest}
    grouped = defaultdict(list)
    for row in segment_hour_rows:
        if row["annual_shade_fraction"] != "":
            grouped[row["segment_id"]].append(float(row["annual_shade_fraction"]))
    candidates = []
    for seg_id, values in grouped.items():
        row = meta[seg_id]
        candidates.append({
            "segment_id": seg_id,
            "route_id": row["route_id"],
            "latitude": row["mid_lat"],
            "longitude": row["mid_lon"],
            "mean_annual_shade_fraction": sum(values) / len(values),
            "distance_to_panorama_m": row.get("distance_to_segment_midpoint_m", ""),
            "qc_status": row["qc_status"],
        })
    candidates.sort(key=lambda row: row["mean_annual_shade_fraction"])
    selected = []
    targets = [0.05, 0.25, 0.45, 0.65, 0.82, 0.95]
    used_routes = set()
    for target in targets:
        index = int(round(target * (len(candidates) - 1))) if candidates else 0
        ordered = sorted(candidates, key=lambda row: abs(candidates.index(row) - index))
        choice = next((row for row in ordered if row["segment_id"] not in {x["segment_id"] for x in selected} and row["route_id"] not in used_routes), None)
        if choice is None:
            choice = next((row for row in ordered if row["segment_id"] not in {x["segment_id"] for x in selected}), None)
        if choice:
            selected.append(choice)
            used_routes.add(choice["route_id"])
    plan = []
    tier_size = max(1, math.ceil(len(selected) / 3))
    for rank, row in enumerate(selected, 1):
        shade = row["mean_annual_shade_fraction"]
        if rank <= tier_size:
            category = "Lower modeled shade"
        elif rank <= tier_size * 2:
            category = "Medium modeled shade"
        else:
            category = "Higher modeled shade"
        plan.append({
            "spot_id": f"M{rank:02d}",
            **row,
            "shade_category": category,
            "recommended_measurement_hours": "08:00, 10:00, 16:00, 18:00",
            "field_protocol": "Record handheld WBGT, air temperature, relative humidity, wind context, exact time, direct-sun state, and photo. Keep the meter at pedestrian torso height and follow its stabilization guidance.",
            "calibration_use": "Join to NEA station WBGT by timestamp, then regress handheld WBGT on station WBGT plus shade state, sky view factor, time and route context.",
        })
    write_csv(root / "handover_to_heat_model" / "measurement_spot_plan.csv", plan)


def draw_route_chart(root: Path, rows):
    by_route = defaultdict(dict)
    for row in rows:
        if row["annual_shade_fraction"] != "":
            by_route[row["route_id"]][int(row["local_hour"])] = float(row["annual_shade_fraction"])
    width, height = 1700, 1050
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((width // 2, 25), "Annual mean route shade by walking hour", fill="black", anchor="ma", font=font)
    draw.text((width // 2, 48), "Provisional: KartaView panorama headings require field validation", fill="#8A2D2D", anchor="ma", font=font)
    routes = sorted(by_route)
    margin_x, top = 90, 90
    panel_w, panel_h = 300, 410
    gap_x, gap_y = 25, 45
    for index, route_id in enumerate(routes):
        col, row_index = index % 5, index // 5
        x0 = margin_x + col * (panel_w + gap_x)
        y0 = top + row_index * (panel_h + gap_y)
        draw.text((x0 + panel_w // 2, y0), route_id, fill="black", anchor="ma", font=font)
        base_y = y0 + panel_h - 40
        chart_top = y0 + 35
        draw.line((x0 + 30, chart_top, x0 + 30, base_y), fill="#444444", width=2)
        draw.line((x0 + 30, base_y, x0 + panel_w - 10, base_y), fill="#444444", width=2)
        for i, hour in enumerate(WALKING_HOURS):
            value = by_route[route_id].get(hour, 0)
            bx0 = x0 + 40 + i * 30
            bx1 = bx0 + 20
            by = base_y - value * (base_y - chart_top)
            color = "#2E8B57" if value >= 0.7 else "#F0B44D" if value >= 0.35 else "#D9684A"
            draw.rectangle((bx0, by, bx1, base_y), fill=color)
            draw.text(((bx0 + bx1) / 2, base_y + 8), str(hour), fill="#333333", anchor="ma", font=font)
        for value in [0, 0.5, 1.0]:
            yy = base_y - value * (base_y - chart_top)
            draw.text((x0 + 25, yy), f"{int(value*100)}", fill="#555555", anchor="rm", font=font)
    output = root / "figures" / "all_routes_annual_shade_by_hour.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def draw_sample_chart(root: Path, rows):
    if not rows:
        return
    first_segment = rows[0]["segment_id"]
    selected = [row for row in rows if row["segment_id"] == first_segment and row["analysis_date"] == "2026-06-21"]
    selected.sort(key=lambda row: int(row["local_hour"]))
    width, height = 1500, 650
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((width // 2, 35), f"Shade by hour sample  {first_segment}  21 June 2026", fill="black", anchor="ma", font=font)
    draw.text((width // 2, 58), "North-aligned KartaView 360 panorama with SegFormer ADE20K classification", fill="#555555", anchor="ma", font=font)
    x0, x1, y = 130, width - 100, 310
    draw.line((x0, y, x1, y), fill="#333333", width=3)
    for index, row in enumerate(selected):
        x = x0 + index * (x1 - x0) / max(1, len(selected) - 1)
        state = row["shade_state"]
        color = {"SUNLIT": "#F0A000", "SHADED_TREE": "#2E8B57", "SHADED_SOLID": "#666666", "NIGHT": "#23395D", "UNKNOWN": "#664E7F"}[state]
        draw.ellipse((x - 18, y - 18, x + 18, y + 18), fill=color, outline="white", width=2)
        draw.text((x, y - 35), state.replace("SHADED_", "SHADE "), fill=color, anchor="ms", font=font)
        draw.text((x, y + 32), f"{int(row['local_hour']):02d}:00", fill="#333333", anchor="ma", font=font)
        draw.text((x, y + 52), f"{float(row['sun_elevation_deg']):.1f} deg", fill="#777777", anchor="ma", font=font)
    output = root / "figures" / "sample_shade_by_hour.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def main():
    parser = argparse.ArgumentParser(description="Complete shade-by-hour pipeline for the AMK senior routes")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpkg", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--runtime-path", type=Path)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--radius-m", type=int, default=160)
    parser.add_argument("--skip-acquisition", action="store_true")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    segments = read_segments(args.gpkg)
    manifest_path = args.root / "data" / "panorama_manifest.csv"
    if args.skip_acquisition and manifest_path.exists():
        manifest = read_csv(manifest_path)
    else:
        manifest = acquire_panoramas(segments, args.root, args.radius_m)
    summaries = analyse(args.root, manifest, args.model, args.runtime_path, args.year)
    run_summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "analysis_year": args.year,
        "route_count": len({row["route_id"] for row in segments}),
        "segment_count": len(segments),
        "segments_with_panoramas": sum(1 for row in manifest if row.get("panorama_file")),
        "segments_without_panoramas": sum(1 for row in manifest if not row.get("panorama_file")),
        "walking_hours": WALKING_HOURS,
        "representative_dates": REPRESENTATIVE_DATES,
        "route_summaries": summaries,
        "result_status": "PROVISIONAL_PARTIAL_VERTICAL_FOV_AND_HEADING_UNVALIDATED",
        "important_limitation": "KartaView panorama headings orient true north but have not been independently field-validated. Many SPHERE images do not observe the zenith; black unobserved pixels are explicitly UNKNOWN and excluded. Results are suitable for prototype and heat-model calibration planning, not final audited exposure claims.",
    }
    (args.root / "outputs" / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
