from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from collections import Counter

app = FastAPI(title="Poputchik Matcher (FastAPI)", version="1.0.0")

# -----------------------------
# In-memory state
# -----------------------------
DF: Optional[pd.DataFrame] = None  # raw points
OD: Optional[pd.DataFrame] = None  # heuristic O/D table
READY: bool = False

# Google Drive dataset
GOOGLE_DRIVE_ID = "1AMyT5zsKbAGyNslMMN1yrYRVzlfgPA2H"
CSV_URL = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}"

# Default columns
RIDE_COL = "randomized_id"
LAT_COL = "lat"
LON_COL = "lng"


# -----------------------------
# Helpers
# -----------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def latlon_to_xy(lat, lon, lat0=None):
    if lat0 is None:
        lat0 = np.mean(lat)
    R = 6371000.0
    x = np.radians(lon) * R * np.cos(np.radians(lat0))
    y = np.radians(lat) * R
    return x, y


def convex_hull(points):
    pts = sorted(points.tolist())
    if len(pts) <= 1:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def farthest_pair(hull):
    if len(hull) < 2:
        return 0, 0
    d2 = ((hull[:, None, :] - hull[None, :, :]) ** 2).sum(2)
    i, j = np.unravel_index(d2.argmax(), d2.shape)
    return i, j


def grid_cell(lat, lon, prec=3):
    return (round(lat, prec), round(lon, prec))


def build_od(df, ride_col="ride_id", lat_col="lat", lon_col="lon"):
    rows = []
    for rid, g in df.groupby(ride_col):
        lat, lon = g[lat_col].values, g[lon_col].values
        if len(lat) < 2:
            continue
        x, y = latlon_to_xy(lat, lon)
        hull = convex_hull(np.c_[x, y])
        if len(hull) < 2:
            start, end = (lat[0], lon[0]), (lat[-1], lon[-1])
        else:
            idx = [np.argmin((x - vx) ** 2 + (y - vy) ** 2) for vx, vy in hull]
            i, j = farthest_pair(hull)
            p, q = (lat[idx[i]], lon[idx[i]]), (lat[idx[j]], lon[idx[j]])
            counts = Counter([grid_cell(a, b) for a, b in zip(lat, lon)])
            start, end = (
                (p, q) if counts[grid_cell(*p)] >= counts[grid_cell(*q)] else (q, p)
            )
        dist = float(haversine_km(start[0], start[1], end[0], end[1]))
        rows.append(
            dict(
                ride_id=rid,
                start_lat=float(start[0]),
                start_lon=float(start[1]),
                end_lat=float(end[0]),
                end_lon=float(end[1]),
                straight_line_km=dist,
            )
        )
    return pd.DataFrame(rows)


def direction_vec(lat1, lon1, lat2, lon2):
    lat1 = np.asarray(lat1)
    lon1 = np.asarray(lon1)
    lat2 = np.asarray(lat2)
    lon2 = np.asarray(lon2)
    lat0 = np.mean(lat1)
    R = 6371000
    sx = np.radians(lon1) * R * np.cos(np.radians(lat0))
    sy = np.radians(lat1) * R
    ex = np.radians(lon2) * R * np.cos(np.radians(lat0))
    ey = np.radians(lat2) * R
    v = np.c_[ex - sx, ey - sy]
    norm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v / norm


def match_rides(od, origin_radius=2.0, dest_radius=2.0, min_cos=0.5, top_k=3):
    v = direction_vec(od.start_lat, od.start_lon, od.end_lat, od.end_lon)
    ids = od.ride_id.values
    pairs = []
    for i in range(len(od)):
        ds = haversine_km(
            od.start_lat.iloc[i],
            od.start_lon.iloc[i],
            od.start_lat.values,
            od.start_lon.values,
        )
        de = haversine_km(
            od.end_lat.iloc[i], od.end_lon.iloc[i], od.end_lat.values, od.end_lon.values
        )
        dot = (v[i] * v).sum(1)
        mask = (
            (ds <= origin_radius)
            & (de <= dest_radius)
            & (dot >= min_cos)
            & (ids != ids[i])
        )
        if not mask.any():
            continue
        s_prox = np.clip(1 - ds / origin_radius, 0, 1)
        e_prox = np.clip(1 - de / dest_radius, 0, 1)
        score = 0.6 * dot + 0.2 * s_prox + 0.2 * e_prox
        idx = np.argsort(score[mask])[::-1][:top_k]
        for j in np.where(mask)[0][idx]:
            pairs.append(
                dict(
                    ride_a=str(ids[i]),
                    ride_b=str(ids[j]),
                    start_km=float(ds[j]),
                    end_km=float(de[j]),
                    direction_cos=float(dot[j]),
                    score=float(score[j]),
                )
            )
    return pd.DataFrame(pairs)


# -----------------------------
# API models
# -----------------------------
class BuildODRequest(BaseModel):
    ride_col: Optional[str] = Field(default=None, description="Ride ID column name")
    lat_col: Optional[str] = Field(default=None, description="Latitude column name")
    lon_col: Optional[str] = Field(default=None, description="Longitude column name")


class MatchParams(BaseModel):
    origin_radius_km: float = 2.0
    dest_radius_km: float = 2.0
    min_direction_cos: float = 0.5
    top_k: int = 5


class Candidate(BaseModel):
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float


class MatchRequest(BaseModel):
    params: MatchParams = MatchParams()
    candidates: List[Candidate]


# -----------------------------
# API endpoints
# -----------------------------
@app.get("/health")
def health():
    return {"ok": True, "ready": READY, "rows": 0 if DF is None else len(DF)}


@app.post("/build_od")
def api_build_od(req: BuildODRequest):
    """
    Load CSV from Google Drive and build heuristic O/D from unordered points.
    """
    global DF, OD, READY, RIDE_COL, LAT_COL, LON_COL

    try:
        DF = pd.read_csv(CSV_URL)
    except Exception as e:
        raise HTTPException(500, f"Failed to load CSV from Google Drive: {e}")

    # allow override of column names
    if req.ride_col:
        RIDE_COL = req.ride_col
    if req.lat_col:
        LAT_COL = req.lat_col
    if req.lon_col:
        LON_COL = req.lon_col

    for c in [RIDE_COL, LAT_COL, LON_COL]:
        if c not in DF.columns:
            raise HTTPException(400, f"Column '{c}' not found in CSV.")

    OD = build_od(DF, ride_col=RIDE_COL, lat_col=LAT_COL, lon_col=LON_COL)
    READY = True
    return {
        "ok": True,
        "num_rides": int(len(OD)),
        "ride_col": RIDE_COL,
        "lat_col": LAT_COL,
        "lon_col": LON_COL,
    }


@app.post("/match_candidates")
def api_match_candidates(req: MatchRequest) -> Dict[str, Any]:
    """
    Send candidate start/end coords; returns best-matching rides from dataset.
    """
    global OD, READY
    if not READY or OD is None:
        raise HTTPException(400, "Not ready. Call /build_od first.")

    # Build small OD-like table for candidates
    cand_rows = [
        {
            "ride_id": f"cand_{i}",
            "start_lat": c.start_lat,
            "start_lon": c.start_lon,
            "end_lat": c.end_lat,
            "end_lon": c.end_lon,
        }
        for i, c in enumerate(req.candidates)
    ]
    CAND = pd.DataFrame(cand_rows)

    # Combine and run matching
    FULL = pd.concat(
        [CAND, OD[["ride_id", "start_lat", "start_lon", "end_lat", "end_lon"]]],
        ignore_index=True,
    )
    matches = match_rides(
        FULL,
        origin_radius=req.params.origin_radius_km,
        dest_radius=req.params.dest_radius_km,
        min_cos=req.params.min_direction_cos,
        top_k=req.params.top_k,
    )

    # Filter to edges where source is a candidate
    cand_ids = set(CAND["ride_id"])
    out = matches[matches["ride_a"].isin(cand_ids)].copy()

    if out.empty:
        return {"ok": True, "num_matches": 0, "matches": []}

    # Add candidate_index for convenience
    out["candidate_index"] = np.nan
    for i in range(len(CAND)):
        cid = f"cand_{i}"
        mask = out["ride_a"] == cid
        if mask.any():
            out.loc[mask, "candidate_index"] = i

    # Attach OD preview for matched dataset rides
    od_map = OD.set_index("ride_id")[
        ["start_lat", "start_lon", "end_lat", "end_lon"]
    ].to_dict(orient="index")
    enriched = []
    for rec in out.to_dict(orient="records"):
        rb = rec["ride_b"]
        rec["ride_b_od"] = od_map.get(rb, {})
        enriched.append(rec)

    return {"ok": True, "num_matches": len(enriched), "matches": enriched}
