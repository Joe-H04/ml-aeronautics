"""
Step 7: FastAPI service for aircraft trajectory reconstruction.

Endpoints:
  GET  /health                  - health check + model status
  GET  /flights                 - list available flight IDs from local parquet files
  GET  /opensky/flights         - list live OpenSky flights in a map area
  GET  /opensky/reconstruct/... - reconstruct a live OpenSky track
  GET  /reconstruct/{flight_id} - reconstruct a stored flight (icao24_firstSeen)
  POST /reconstruct             - reconstruct from raw track points

Gap-filling uses the trained LSTM model (12.8% better than great-circle baseline).
Falls back to great-circle interpolation when LSTM context requirements can't be met.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from baseline import interpolate_great_circle
from model import LSTMTrajectoryModel, great_circle_km
from opensky_client import OpenSkyClient, OpenSkyError, OpenSkyHTTPError

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TRACKS_DIR = Path("data/clean/tracks")
WEIGHTS_PATH = Path("weights/lstm_residual_trajectory.keras")
CONTEXT_LENGTH = 20
GAP_LENGTH = 10
GAP_THRESHOLD_SECONDS = 120  # time gap larger than this triggers gap-filling

# ---------------------------------------------------------------------------
# Startup: load LSTM model once
# ---------------------------------------------------------------------------

_model: Optional[LSTMTrajectoryModel] = None
_opensky = OpenSkyClient()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    if WEIGHTS_PATH.exists():
        try:
            _model = LSTMTrajectoryModel.load(
                str(WEIGHTS_PATH),
                context_length=CONTEXT_LENGTH,
                gap_length=GAP_LENGTH,
            )
        except Exception as exc:
            print(f"Warning: could not load LSTM weights: {exc}")
    else:
        print(f"Warning: weights not found at {WEIGHTS_PATH}; using great-circle fallback.")
    yield


app = FastAPI(
    title="Aircraft Trajectory Reconstruction API",
    description="ADS-B/ADS-C gap-filling pipeline — LSTM + great-circle fallback.",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class TrackPoint(BaseModel):
    time: float = Field(..., description="Unix timestamp (seconds)")
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    altitude_m: float = Field(..., description="Barometric altitude in metres")


class ReconstructedPoint(BaseModel):
    time: float
    latitude: float
    longitude: float
    altitude_m: float
    source: Literal["original", "lstm", "great_circle"]


class ReconstructionResponse(BaseModel):
    flight_id: str
    icao24: str
    total_points: int
    original_points: int
    reconstructed_points: int
    gaps_found: int
    gaps_filled: int
    track: List[ReconstructedPoint]


class RawTrackRequest(BaseModel):
    flight_id: str = Field(default="unknown", description="Optional label for this flight")
    points: List[TrackPoint] = Field(..., min_length=1)


class ComparisonPoint(BaseModel):
    time: float
    latitude: float
    longitude: float
    altitude_m: float


class DemoResponse(BaseModel):
    flight_id: str
    icao24: str
    gap_index: int
    total_gaps: int
    full_track: List[ReconstructedPoint]
    truth: List[ComparisonPoint]
    lstm: Optional[List[ComparisonPoint]]
    great_circle: List[ComparisonPoint]
    lstm_mean_error_km: Optional[float]
    gc_mean_error_km: Optional[float]


class OpenSkyFlight(BaseModel):
    icao24: str
    callsign: str
    origin_country: str
    time_position: Optional[int]
    last_contact: int
    longitude: float
    latitude: float
    altitude_m: Optional[float]
    velocity_mps: Optional[float]
    on_ground: bool


class OpenSkyFlightsResponse(BaseModel):
    auth_mode: Literal["oauth", "anonymous"]
    count: int
    flights: List[OpenSkyFlight]


# ---------------------------------------------------------------------------
# Core gap-filling logic
# ---------------------------------------------------------------------------


def _great_circle_fill(p_before: np.ndarray, alt_before: float,
                       p_after: np.ndarray, alt_after: float,
                       gap_times: np.ndarray) -> List[ReconstructedPoint]:
    n = len(gap_times)
    fractions = np.linspace(1 / (n + 1), n / (n + 1), n)
    result = []
    for i, (frac, t) in enumerate(zip(fractions, gap_times)):
        lat, lon = interpolate_great_circle(
            float(p_before[0]), float(p_before[1]),
            float(p_after[0]), float(p_after[1]),
            frac,
        )
        alt = float(alt_before + (alt_after - alt_before) * frac)
        result.append(ReconstructedPoint(
            time=float(t), latitude=lat, longitude=lon,
            altitude_m=alt, source="great_circle",
        ))
    return result


def _lstm_fill(positions: np.ndarray, altitudes: np.ndarray, times: np.ndarray,
               gap_start_idx: int) -> Optional[List[ReconstructedPoint]]:
    """Fill GAP_LENGTH points starting at gap_start_idx using the LSTM."""
    if _model is None:
        return None

    before_end = gap_start_idx
    after_start = gap_start_idx  # gap points are synthetic; context borders gap

    if before_end < CONTEXT_LENGTH or (len(positions) - after_start) < CONTEXT_LENGTH:
        return None

    before_pos = positions[before_end - CONTEXT_LENGTH: before_end]
    before_alt = altitudes[before_end - CONTEXT_LENGTH: before_end]
    before_t = times[before_end - CONTEXT_LENGTH: before_end]

    after_pos = positions[after_start: after_start + CONTEXT_LENGTH]
    after_alt = altitudes[after_start: after_start + CONTEXT_LENGTH]
    after_t = times[after_start: after_start + CONTEXT_LENGTH]

    t_gap_start = float(times[before_end - 1])
    t_gap_end = float(times[after_start])
    gap_times = np.linspace(t_gap_start, t_gap_end, GAP_LENGTH + 2)[1:-1]

    try:
        lats, lons, alts = _model.predict_gap(
            before_pos, before_alt, before_t,
            after_pos, after_alt, after_t,
        )
    except Exception:
        return None

    return [
        ReconstructedPoint(
            time=float(gap_times[i]),
            latitude=float(lats[i]),
            longitude=float(lons[i]),
            altitude_m=float(alts[i]),
            source="lstm",
        )
        for i in range(GAP_LENGTH)
    ]


def reconstruct_track(
    positions: np.ndarray,
    altitudes: np.ndarray,
    times: np.ndarray,
    flight_id: str,
    icao24: str,
) -> ReconstructionResponse:
    """Detect gaps and fill them. Returns the full reconstructed track."""
    order = np.argsort(times)
    positions, altitudes, times = positions[order], altitudes[order], times[order]

    result: List[ReconstructedPoint] = []
    gaps_found = 0
    gaps_filled = 0

    for i in range(len(times)):
        result.append(ReconstructedPoint(
            time=float(times[i]),
            latitude=float(positions[i, 0]),
            longitude=float(positions[i, 1]),
            altitude_m=float(altitudes[i]),
            source="original",
        ))

        if i == len(times) - 1:
            break

        dt = float(times[i + 1]) - float(times[i])
        if dt <= GAP_THRESHOLD_SECONDS:
            continue

        gaps_found += 1

        # Try LSTM first; it needs CONTEXT_LENGTH points on each side
        # At this point `i+1` is the after-side start in the original array
        filled = _lstm_fill(positions, altitudes, times, gap_start_idx=i + 1)

        if filled is None:
            # Fall back to great-circle with GAP_LENGTH fill points
            gap_times = np.linspace(float(times[i]), float(times[i + 1]), GAP_LENGTH + 2)[1:-1]
            filled = _great_circle_fill(
                positions[i], float(altitudes[i]),
                positions[i + 1], float(altitudes[i + 1]),
                gap_times,
            )

        result.extend(filled)
        gaps_filled += 1

    original_pts = len(times)
    reconstructed_pts = len(result) - original_pts

    return ReconstructionResponse(
        flight_id=flight_id,
        icao24=icao24,
        total_points=len(result),
        original_points=original_pts,
        reconstructed_points=reconstructed_pts,
        gaps_found=gaps_found,
        gaps_filled=gaps_filled,
        track=result,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_parquet(path: Path):
    df = pd.read_parquet(path).sort_values("time").reset_index(drop=True)
    df = df.dropna(subset=["latitude", "longitude", "baro_altitude"])
    if df.empty:
        return None, None, None, None
    positions = df[["latitude", "longitude"]].to_numpy(dtype=np.float64)
    altitudes = df["baro_altitude"].to_numpy(dtype=np.float64)
    times = df["time"].to_numpy(dtype=np.float64)
    icao24 = str(df["icao24"].iloc[0]) if "icao24" in df.columns else path.stem.split("_")[0]
    return positions, altitudes, times, icao24


def _load_opensky_track(icao24: str, time_seconds: int):
    track = _opensky.get_track(icao24, time_seconds=time_seconds)
    raw_path = track.get("path") or []

    cleaned = []
    last_altitude = 0.0
    for point in raw_path:
        if len(point) < 4:
            continue

        point_time, lat, lon, baro_altitude = point[:4]
        if point_time is None or lat is None or lon is None:
            continue

        altitude = float(baro_altitude) if baro_altitude is not None else last_altitude
        last_altitude = altitude
        cleaned.append((float(point_time), float(lat), float(lon), altitude))

    cleaned.sort(key=lambda point: point[0])

    if len(cleaned) < 2:
        raise HTTPException(
            status_code=422,
            detail="OpenSky returned too few valid track points to reconstruct.",
        )

    times = np.array([point[0] for point in cleaned], dtype=np.float64)
    positions = np.array([[point[1], point[2]] for point in cleaned], dtype=np.float64)
    altitudes = np.array([point[3] for point in cleaned], dtype=np.float64)
    track_icao24 = str(track.get("icao24") or icao24).strip().lower()
    callsign = str(track.get("callsign") or track.get("calllsign") or "").strip()
    return positions, altitudes, times, track_icao24, callsign


def _raise_from_opensky(exc: OpenSkyError):
    if isinstance(exc, OpenSkyHTTPError):
        status_code = exc.status_code or 502
        if status_code in {400, 401, 403, 404, 429}:
            raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    raise HTTPException(status_code=502, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

_TEMPLATES = Path(__file__).parent / "templates"


@app.get("/", response_class=FileResponse, include_in_schema=False)
def ui():
    return FileResponse(_TEMPLATES / "index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "lstm_loaded": _model is not None,
        "opensky_auth_mode": _opensky.auth_mode,
        "opensky_auth_configured": _opensky.is_configured,
        "tracks_dir": str(TRACKS_DIR),
        "tracks_available": len(list(TRACKS_DIR.glob("*.parquet"))) if TRACKS_DIR.exists() else 0,
    }


@app.get("/flights", response_model=List[str])
def list_flights():
    """Return all available flight IDs (parquet file stems)."""
    if not TRACKS_DIR.exists():
        raise HTTPException(status_code=404, detail=f"Tracks directory not found: {TRACKS_DIR}")
    return sorted(p.stem for p in TRACKS_DIR.glob("*.parquet"))


@app.get("/opensky/flights", response_model=OpenSkyFlightsResponse)
def list_opensky_flights(
    query: str = "",
    limit: int = 25,
    lamin: Optional[float] = None,
    lamax: Optional[float] = None,
    lomin: Optional[float] = None,
    lomax: Optional[float] = None,
):
    """List live OpenSky flights, optionally filtered to the current map bounds."""
    if any(value is None for value in (lamin, lamax, lomin, lomax)) and any(
        value is not None for value in (lamin, lamax, lomin, lomax)
    ):
        raise HTTPException(
            status_code=422,
            detail="lamin, lamax, lomin, and lomax must be provided together.",
        )

    try:
        flights = _opensky.list_current_flights(
            lamin=lamin,
            lamax=lamax,
            lomin=lomin,
            lomax=lomax,
            limit=limit,
            query=query,
        )
    except OpenSkyError as exc:
        _raise_from_opensky(exc)

    return OpenSkyFlightsResponse(
        auth_mode=_opensky.auth_mode,
        count=len(flights),
        flights=[OpenSkyFlight(**flight) for flight in flights],
    )


@app.get("/reconstruct/{flight_id}", response_model=ReconstructionResponse)
def reconstruct_flight(flight_id: str):
    """
    Reconstruct a stored flight by its ID (e.g. `a1b2c3_1678900000`).

    The ID matches the parquet filename stem under data/clean/tracks/.
    Gaps longer than {GAP_THRESHOLD_SECONDS}s are filled with LSTM (or great-circle fallback).
    """
    path = TRACKS_DIR / f"{flight_id}.parquet"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Flight not found: {flight_id}")

    positions, altitudes, times, icao24 = _load_parquet(path)
    if positions is None:
        raise HTTPException(status_code=422, detail="Track is empty after cleaning.")

    return reconstruct_track(positions, altitudes, times, flight_id, icao24)


@app.get("/opensky/reconstruct/{icao24}", response_model=ReconstructionResponse)
def reconstruct_opensky_flight(icao24: str, time: int = 0):
    """
    Reconstruct a track fetched live from OpenSky instead of from the local parquet folder.

    `time=0` asks OpenSky for the live track. Passing `last_contact` from `/opensky/flights`
    is often a little more robust when selecting a current aircraft from the list.
    """
    try:
        positions, altitudes, times, track_icao24, callsign = _load_opensky_track(icao24, time)
    except OpenSkyError as exc:
        _raise_from_opensky(exc)

    flight_label = f"opensky_{track_icao24}_{int(times[0])}"
    if callsign:
        flight_label = f"{flight_label}_{callsign.strip().replace(' ', '')}"

    return reconstruct_track(positions, altitudes, times, flight_label, track_icao24)


@app.post("/reconstruct", response_model=ReconstructionResponse)
def reconstruct_raw(body: RawTrackRequest):
    """
    Reconstruct a trajectory from raw track points.

    Send a list of {time, latitude, longitude, altitude_m} points.
    Returns the same points with LSTM-filled (or great-circle-filled) gap segments inserted.
    """
    if len(body.points) < 2:
        raise HTTPException(status_code=422, detail="At least 2 track points required.")

    times = np.array([p.time for p in body.points], dtype=np.float64)
    positions = np.array([[p.latitude, p.longitude] for p in body.points], dtype=np.float64)
    altitudes = np.array([p.altitude_m for p in body.points], dtype=np.float64)
    icao24 = body.flight_id.split("_")[0]

    return reconstruct_track(positions, altitudes, times, body.flight_id, icao24)


# ---------------------------------------------------------------------------
# Demo / comparison endpoint
# ---------------------------------------------------------------------------

_N_DEMO_GAPS = 10  # evenly-spaced gap positions exposed per flight


@app.get("/demo/{flight_id}", response_model=DemoResponse)
def demo_gap(flight_id: str, gap: int = 0):
    """
    Artificially hide GAP_LENGTH points from a dense flight and compare
    LSTM reconstruction vs great-circle baseline against the ground truth.

    Use `?gap=0..9` to browse up to 10 evenly-spaced positions along the flight.
    Requires the flight to have at least 2*CONTEXT_LENGTH + GAP_LENGTH points.
    """
    path = TRACKS_DIR / f"{flight_id}.parquet"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Flight not found: {flight_id}")

    positions, altitudes, times, icao24 = _load_parquet(path)
    if positions is None:
        raise HTTPException(status_code=422, detail="Track is empty after cleaning.")

    window = 2 * CONTEXT_LENGTH + GAP_LENGTH
    if len(positions) < window:
        raise HTTPException(
            status_code=422,
            detail=f"Flight has only {len(positions)} points; need ≥ {window} for comparison demo.",
        )

    # Evenly space up to _N_DEMO_GAPS positions across the valid range
    max_start = len(positions) - window
    n_gaps = min(_N_DEMO_GAPS, max_start + 1)
    starts = [int(round(max_start * i / max(n_gaps - 1, 1))) for i in range(n_gaps)]

    gap = max(0, min(gap, n_gaps - 1))
    start = starts[gap]

    before_slc = slice(start, start + CONTEXT_LENGTH)
    gap_slc    = slice(start + CONTEXT_LENGTH, start + CONTEXT_LENGTH + GAP_LENGTH)
    after_slc  = slice(start + CONTEXT_LENGTH + GAP_LENGTH, start + window)

    before_pos   = positions[before_slc]
    before_alt   = altitudes[before_slc]
    before_times = times[before_slc]
    gap_pos      = positions[gap_slc]
    gap_alt      = altitudes[gap_slc]
    gap_times    = times[gap_slc]
    after_pos    = positions[after_slc]
    after_alt    = altitudes[after_slc]
    after_times  = times[after_slc]

    def _pt(i: int, src: str) -> ReconstructedPoint:
        return ReconstructedPoint(
            time=float(times[i]), latitude=float(positions[i, 0]),
            longitude=float(positions[i, 1]), altitude_m=float(altitudes[i]),
            source=src,  # type: ignore[arg-type]
        )

    # Full track (gap marked as original so UI draws it as background)
    full_track = [_pt(i, "original") for i in range(len(positions))]

    # Ground truth
    truth = [
        ComparisonPoint(
            time=float(gap_times[i]), latitude=float(gap_pos[i, 0]),
            longitude=float(gap_pos[i, 1]), altitude_m=float(gap_alt[i]),
        )
        for i in range(GAP_LENGTH)
    ]

    # LSTM reconstruction
    lstm_fill: Optional[List[ComparisonPoint]] = None
    lstm_error: Optional[float] = None
    if _model is not None:
        try:
            lats, lons, alts = _model.predict_gap(
                before_pos, before_alt, before_times,
                after_pos, after_alt, after_times,
            )
            lstm_fill = [
                ComparisonPoint(
                    time=float(gap_times[i]), latitude=float(lats[i]),
                    longitude=float(lons[i]), altitude_m=float(alts[i]),
                )
                for i in range(GAP_LENGTH)
            ]
            errs = great_circle_km(gap_pos[:, 0], gap_pos[:, 1], lats, lons)
            lstm_error = float(np.mean(errs))
        except Exception:
            pass

    # Great-circle reconstruction
    fractions = np.linspace(1 / (GAP_LENGTH + 1), GAP_LENGTH / (GAP_LENGTH + 1), GAP_LENGTH)
    gc_lats = np.array([
        interpolate_great_circle(
            float(before_pos[-1, 0]), float(before_pos[-1, 1]),
            float(after_pos[0, 0]),  float(after_pos[0, 1]),
            float(f),
        )[0]
        for f in fractions
    ])
    gc_lons = np.array([
        interpolate_great_circle(
            float(before_pos[-1, 0]), float(before_pos[-1, 1]),
            float(after_pos[0, 0]),  float(after_pos[0, 1]),
            float(f),
        )[1]
        for f in fractions
    ])
    gc_alts = before_alt[-1] + (after_alt[0] - before_alt[-1]) * fractions
    gc_fill = [
        ComparisonPoint(
            time=float(gap_times[i]), latitude=float(gc_lats[i]),
            longitude=float(gc_lons[i]), altitude_m=float(gc_alts[i]),
        )
        for i in range(GAP_LENGTH)
    ]
    gc_errs = great_circle_km(gap_pos[:, 0], gap_pos[:, 1], gc_lats, gc_lons)
    gc_error = float(np.mean(gc_errs))

    return DemoResponse(
        flight_id=flight_id,
        icao24=icao24,
        gap_index=gap,
        total_gaps=n_gaps,
        full_track=full_track,
        truth=truth,
        lstm=lstm_fill,
        great_circle=gc_fill,
        lstm_mean_error_km=lstm_error,
        gc_mean_error_km=gc_error,
    )
