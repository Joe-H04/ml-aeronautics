import argparse
import math
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://opensky-network.org/api"
TOKEN_URL = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
DEFAULT_AIRPORTS = ["EDDF", "EGLL", "LFPG", "EHAM", "LEMD", "LIRF", "LOWW", "LSZH", "EKCH", "EDDM"]
FLIGHT_COLUMNS = [
    "icao24", "firstSeen", "estDepartureAirport", "lastSeen", "estArrivalAirport", "callsign",
    "estDepartureAirportHorizDistance", "estDepartureAirportVertDistance",
    "estArrivalAirportHorizDistance", "estArrivalAirportVertDistance",
    "departureAirportCandidatesCount", "arrivalAirportCandidatesCount",
]
TRACK_COLUMNS = ["time", "latitude", "longitude", "baro_altitude", "true_track", "on_ground"]
ADSC_POSITION_TAGS = {"07", "09", "10", "18", "19", "20"}
ADSC_FLIGHT_COLUMNS = FLIGHT_COLUMNS + ["airport", "direction"]
FEET_TO_METERS = 0.3048
HEADER_RE = re.compile(
    r"^Registration:\s+(?P<registration>.*?)\s+ICAO ID:\s+"
    r"(?P<icao24>[0-9a-fA-F]+)\s+ATSU Address:\s+(?P<atsu>.+)$"
)
TAG_RE = re.compile(r"^Tag\s+(?P<tag>\d{2})\b")
FLOAT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
TOKEN = {"value": None, "exp": 0.0}
SESSION = requests.Session()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--airports", nargs="+", default=DEFAULT_AIRPORTS)
    p.add_argument("--hours", type=int, default=4)
    p.add_argument("--output", default="data/clean")
    p.add_argument("--no-tracks", action="store_true")
    p.add_argument("--local-adsc", type=Path, help="Path to decoded ADS-C text data to import locally.")
    p.add_argument("--adsc-gap-hours", type=float, default=6.0)
    p.add_argument("--adsc-min-points", type=int, default=5)
    p.add_argument("--adsc-max-speed-kt", type=float, default=700.0)
    p.add_argument("--adsc-max-records", type=int, help="Debug limit for decoded ADS-C position reports.")
    return p.parse_args()


def auth_headers():
    client_id = os.getenv("OPENSKY_CLIENT_ID")
    client_secret = os.getenv("OPENSKY_CLIENT_SECRET")
    if not client_id or not client_secret:
        return {}
    if time.time() >= TOKEN["exp"]:
        r = requests.post(
            TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
            },
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        TOKEN["value"] = data["access_token"]
        TOKEN["exp"] = time.time() + data.get("expires_in", 1800) - 60
    return {"Authorization": f"Bearer {TOKEN['value']}"}


def get(path, **params):
    try:
        r = SESSION.get(f"{BASE_URL}{path}", headers=auth_headers(), params=params, timeout=30)
        
        if r.status_code == 401 and TOKEN["value"]:
            TOKEN["exp"] = 0
            r = SESSION.get(f"{BASE_URL}{path}", headers=auth_headers(), params=params, timeout=30)
            
        if r.status_code in (404, 500, 502, 503, 504):
            return None
            
        r.raise_for_status()
        return r.json()
        
    except requests.exceptions.RequestException:
        return None


def windows(start, end, step=7200):
    while start < end:
        yield start, min(start + step, end)
        start += step


def collect_flights(airports, start, end, output_dir):
    print(f"\n[PHASE 1] INGESTION: Fetching Flight Metadata...")
    keep = {a.upper() for a in airports}
    chunks = []
    for begin, finish in windows(start, end):
        data = get("/flights/all", begin=begin, end=finish)
        df = pd.DataFrame(data or [], columns=FLIGHT_COLUMNS)
        if df.empty:
            continue
        
        arrivals = df[df["estArrivalAirport"].isin(keep)].copy()
        if not arrivals.empty:
            arrivals["airport"] = arrivals["estArrivalAirport"]
            arrivals["direction"] = "arrivals"
            chunks.append(arrivals)
            
        departures = df[df["estDepartureAirport"].isin(keep)].copy()
        if not departures.empty:
            departures["airport"] = departures["estDepartureAirport"]
            departures["direction"] = "departures"
            chunks.append(departures)

    flights = pd.concat(chunks, ignore_index=True).drop_duplicates(
        subset=["icao24", "firstSeen", "lastSeen", "airport", "direction"]
    ) if chunks else pd.DataFrame()

    if not flights.empty:
        out = output_dir / "flights"
        out.mkdir(parents=True, exist_ok=True)
        flights.to_parquet(out / f"flights_{datetime.now():%Y%m%d_%H%M%S}.parquet", index=False)
        print(f" -> Success: Fetched {len(flights)} flight records from OpenSky.")
    return flights


def clean_and_segment_track(df):
    """
    Trajectory Segmentation and Cleaning with detailed step-by-step logs.
    """
    if df.empty:
        return df, 0

    initial_count = len(df)

    # 1. SEGMENTATION: Chronological Sequencing
    df_clean = df.sort_values(by=["time"]).copy()

    # 2. CLEANING: Duplicate Timestamp Filter
    df_clean = df_clean.drop_duplicates(subset=["time"])

    # 3. CLEANING: Invalid Coordinate/Altitude Filter
    df_clean = df_clean.dropna(subset=["latitude", "longitude", "baro_altitude"])

    # 4. CLEANING: Ground & Space Filter
    df_clean = df_clean[(df_clean["baro_altitude"] < 15000) & (df_clean["on_ground"] == False)]

    final_count = len(df_clean)
    return df_clean, final_count


def normalize_icao24(value):
    cleaned = re.sub(r"[^0-9a-fA-F]", "", str(value or "")).lower()
    if not cleaned:
        return ""
    return cleaned.zfill(6) if len(cleaned) < 6 else cleaned


def parse_float_value(line):
    match = FLOAT_RE.search(line)
    return float(match.group(0)) if match else None


def parse_adsc_timestamp(value):
    dt = datetime.fromisoformat(value.strip())
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def build_adsc_reports(record, positions):
    if not record:
        return []

    icao24 = normalize_icao24(record.get("icao24"))
    if not icao24:
        return []

    reports = []
    for position in positions:
        required = ("time", "latitude", "longitude", "altitude_ft")
        if any(position.get(field) is None for field in required):
            continue

        reports.append(
            {
                "time": int(position["time"]),
                "latitude": float(position["latitude"]),
                "longitude": float(position["longitude"]),
                "baro_altitude": int(round(float(position["altitude_ft"]) * FEET_TO_METERS)),
                "true_track": record.get("true_track"),
                "on_ground": False,
                "icao24": icao24,
                "callsign": str(record.get("callsign") or "").strip(),
                "registration": str(record.get("registration") or "").strip(),
                "atsu": str(record.get("atsu") or "").strip(),
                "channel_frequency": record.get("channel_frequency"),
                "source_tag": position.get("source_tag"),
            }
        )
    return reports


def iter_adsc_position_reports(path, max_records=None):
    record = None
    positions = []
    current_tag = None
    current_position = None
    emitted = 0

    def emit_current_record():
        nonlocal emitted
        for report in build_adsc_reports(record, positions):
            if max_records is not None and emitted >= max_records:
                return
            emitted += 1
            yield report

    with Path(path).open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            header = HEADER_RE.match(stripped)
            if header:
                yield from emit_current_record()
                if max_records is not None and emitted >= max_records:
                    return

                record = header.groupdict()
                positions = []
                current_tag = None
                current_position = None
                continue

            if record is None or not stripped:
                continue

            if stripped.startswith("Channel Frequency:"):
                record["channel_frequency"] = parse_float_value(stripped)
                continue

            tag = TAG_RE.match(stripped)
            if tag:
                current_tag = tag.group("tag")
                current_position = None
                if current_tag in ADSC_POSITION_TAGS:
                    current_position = {"source_tag": current_tag}
                    positions.append(current_position)
                continue

            if current_position is not None:
                if stripped.startswith("Latitude:"):
                    current_position["latitude"] = parse_float_value(stripped)
                elif stripped.startswith("Longitude:"):
                    current_position["longitude"] = parse_float_value(stripped)
                elif stripped.startswith("Altitude:"):
                    current_position["altitude_ft"] = parse_float_value(stripped)
                elif stripped.startswith("Timestamp:"):
                    current_position["time"] = parse_adsc_timestamp(stripped.split(":", 1)[1])
                continue

            if current_tag == "12" and stripped.startswith("Flight ID:"):
                record["callsign"] = stripped.split(":", 1)[1].strip()
            elif current_tag == "14" and stripped.startswith("True track:"):
                record["true_track"] = parse_float_value(stripped)

    yield from emit_current_record()


def bearing_degrees(lat1, lon1, lat2, lon2):
    if lat1 == lat2 and lon1 == lon2:
        return 0.0

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)
    x = math.sin(delta_lambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def haversine_nm(lat1, lon1, lat2, lon2):
    radius_nm = 3440.065
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return radius_nm * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def fill_true_track(df):
    if df.empty:
        return df

    parsed_tracks = pd.to_numeric(df.get("true_track"), errors="coerce")
    filled = []
    for i, value in enumerate(parsed_tracks):
        if pd.notna(value):
            filled.append(int(round(value)) % 360)
            continue

        if len(df) == 1:
            filled.append(0)
            continue

        if i < len(df) - 1:
            current = df.iloc[i]
            target = df.iloc[i + 1]
        else:
            current = df.iloc[i - 1]
            target = df.iloc[i]

        filled.append(
            int(
                round(
                    bearing_degrees(
                        current["latitude"],
                        current["longitude"],
                        target["latitude"],
                        target["longitude"],
                    )
                )
            )
            % 360
        )

    df = df.copy()
    df["true_track"] = pd.Series(filled, index=df.index, dtype="int64")
    return df


def format_adsc_track(rows, icao24, callsign):
    raw_df = pd.DataFrame(rows, columns=TRACK_COLUMNS)
    clean_df, _ = clean_and_segment_track(raw_df)
    if clean_df.empty:
        return clean_df

    clean_df = fill_true_track(clean_df)
    clean_df["time"] = clean_df["time"].astype("int64")
    clean_df["latitude"] = clean_df["latitude"].astype("float64")
    clean_df["longitude"] = clean_df["longitude"].astype("float64")
    clean_df["baro_altitude"] = clean_df["baro_altitude"].round().astype("int64")
    clean_df["on_ground"] = clean_df["on_ground"].astype("bool")
    clean_df["icao24"] = icao24
    clean_df["callsign"] = callsign
    clean_df["flight_start"] = float(clean_df["time"].min())
    clean_df["flight_end"] = float(clean_df["time"].max())
    return clean_df


def adsc_flight_row(icao24, callsign, first_seen, last_seen, atsu):
    return {
        "icao24": icao24,
        "firstSeen": int(first_seen),
        "estDepartureAirport": "",
        "lastSeen": int(last_seen),
        "estArrivalAirport": "",
        "callsign": callsign,
        "estDepartureAirportHorizDistance": pd.NA,
        "estDepartureAirportVertDistance": pd.NA,
        "estArrivalAirportHorizDistance": pd.NA,
        "estArrivalAirportVertDistance": pd.NA,
        "departureAirportCandidatesCount": pd.NA,
        "arrivalAirportCandidatesCount": pd.NA,
        "airport": atsu or "ADS-C",
        "direction": "adsc",
    }


def collect_adsc_tracks(
    input_path,
    output_dir,
    gap_hours=6.0,
    min_points=5,
    max_speed_kt=700.0,
    max_records=None,
):
    print(f"\n[LOCAL ADS-C] INGESTION: Reading decoded ADS-C data from {input_path}...")

    tracks_out = output_dir / "tracks"
    flights_out = output_dir / "flights"
    tracks_out.mkdir(parents=True, exist_ok=True)
    flights_out.mkdir(parents=True, exist_ok=True)

    gap_seconds = int(gap_hours * 3600)
    active = {}
    flight_rows = []
    total_raw_points = 0
    total_clean_points = 0
    total_tracks = 0
    skipped_tracks = 0

    def should_split(segment, report):
        if report["callsign"] and segment["callsign"] and report["callsign"] != segment["callsign"]:
            return True

        dt = report["time"] - segment["last_time"]
        if dt > gap_seconds:
            return True

        if dt > 0 and max_speed_kt:
            distance_nm = haversine_nm(
                segment["last_latitude"],
                segment["last_longitude"],
                report["latitude"],
                report["longitude"],
            )
            speed_kt = distance_nm / (dt / 3600)
            if speed_kt > max_speed_kt:
                return True

        return False

    def flush_segment(icao24):
        nonlocal total_clean_points, total_tracks, skipped_tracks
        segment = active.pop(icao24, None)
        if not segment:
            return

        callsign = segment["callsign"]
        if len(segment["rows"]) < min_points:
            skipped_tracks += 1
            return

        clean_df = format_adsc_track(segment["rows"], icao24, callsign)
        if len(clean_df) < min_points:
            skipped_tracks += 1
            return

        first_seen = int(clean_df["time"].min())
        last_seen = int(clean_df["time"].max())
        clean_df.to_parquet(tracks_out / f"{icao24}_{first_seen}.parquet", index=False)

        flight_rows.append(adsc_flight_row(icao24, callsign, first_seen, last_seen, segment["atsu"]))
        total_clean_points += len(clean_df)
        total_tracks += 1

    for report in iter_adsc_position_reports(input_path, max_records=max_records):
        total_raw_points += 1
        icao24 = report["icao24"]

        if total_raw_points % 5000 == 0:
            stale_cutoff = report["time"] - gap_seconds
            for stale_icao24, segment in list(active.items()):
                if segment["last_time"] < stale_cutoff:
                    flush_segment(stale_icao24)

        segment = active.get(icao24)
        if segment is not None and should_split(segment, report):
            flush_segment(icao24)
            segment = None

        if segment is None:
            segment = {
                "rows": [],
                "callsign": report["callsign"],
                "registration": report["registration"],
                "atsu": report["atsu"],
                "last_time": report["time"],
                "last_latitude": report["latitude"],
                "last_longitude": report["longitude"],
            }
            active[icao24] = segment
        elif report["callsign"] and not segment["callsign"]:
            segment["callsign"] = report["callsign"]

        segment["rows"].append({column: report[column] for column in TRACK_COLUMNS})
        segment["last_time"] = report["time"]
        segment["last_latitude"] = report["latitude"]
        segment["last_longitude"] = report["longitude"]
        if report["atsu"]:
            segment["atsu"] = report["atsu"]

        if total_raw_points % 100000 == 0:
            print(f" -> Parsed {total_raw_points:,} ADS-C position reports...")

    for icao24 in list(active):
        flush_segment(icao24)

    if flight_rows:
        flights_df = pd.DataFrame(flight_rows, columns=ADSC_FLIGHT_COLUMNS)
        flights_df["firstSeen"] = flights_df["firstSeen"].astype("int64")
        flights_df["lastSeen"] = flights_df["lastSeen"].astype("int64")
        for column in [
            "estDepartureAirportHorizDistance",
            "estDepartureAirportVertDistance",
            "estArrivalAirportHorizDistance",
            "estArrivalAirportVertDistance",
        ]:
            flights_df[column] = flights_df[column].astype("Float64")
        for column in ["departureAirportCandidatesCount", "arrivalAirportCandidatesCount"]:
            flights_df[column] = flights_df[column].astype("Int64")

        flights_df.to_parquet(
            flights_out / f"flights_adsc_{datetime.now():%Y%m%d_%H%M%S}.parquet",
            index=False,
        )

    print(f"\n[SUMMARY] Local ADS-C Pipeline Results:")
    print(f" -> Total Raw Points Ingested: {total_raw_points}")
    print(f" -> Total Clean Points Retained: {total_clean_points}")
    print(f" -> Track Files Written: {total_tracks}")
    print(f" -> Short/Empty Tracks Skipped: {skipped_tracks}")
    if total_raw_points > 0:
        removal_rate = ((total_raw_points - total_clean_points) / total_raw_points) * 100
        print(f" -> Noise Removal Rate: {removal_rate:.1f}%")


def collect_tracks(flights, output_dir):
    print(f"\n[PHASE 2] CLEANING: Processing Trajectories...")
    out = output_dir / "tracks"
    out.mkdir(parents=True, exist_ok=True)
    
    unique_flights = flights.drop_duplicates(subset=["icao24", "firstSeen", "lastSeen"])
    total_flights = len(unique_flights)
    
    total_raw_points = 0
    total_clean_points = 0
    
    for i, (_, row) in enumerate(unique_flights.iterrows(), 1):
        icao24 = str(row["icao24"]).lower()
        first_seen = int(row["firstSeen"])
        
        track_data = get("/tracks/all", icao24=icao24, time=first_seen) or {}
        path_data = track_data.get("path")
        
        if not path_data:
            continue
            
        raw_df = pd.DataFrame(path_data, columns=TRACK_COLUMNS)
        raw_count = len(raw_df)
        total_raw_points += raw_count
        
        clean_df, clean_count = clean_and_segment_track(raw_df)
        total_clean_points += clean_count
        
        if not clean_df.empty:
            clean_df["icao24"] = icao24
            clean_df["callsign"] = track_data.get("callsign") or str(row.get("callsign") or "").strip()
            clean_df["flight_start"] = track_data.get("startTime")
            clean_df["flight_end"] = track_data.get("endTime")
            clean_df.to_parquet(out / f"{icao24}_{first_seen}.parquet", index=False)
            
        # Log progress every 5 flights to keep the terminal clean but informative
        if i % 5 == 0 or i == total_flights:
            print(f" -> Progress: [{i}/{total_flights}] flights processed.")

    print(f"\n[SUMMARY] Pipeline Results:")
    print(f" -> Total Raw Points Ingested: {total_raw_points}")
    print(f" -> Total Clean Points Retained: {total_clean_points}")
    if total_raw_points > 0:
        removal_rate = ((total_raw_points - total_clean_points) / total_raw_points) * 100
        print(f" -> Noise Removal Rate: {removal_rate:.1f}%")


def main():
    args = parse_args()
    output_dir = Path(args.output)

    if args.local_adsc:
        collect_adsc_tracks(
            args.local_adsc,
            output_dir,
            gap_hours=args.adsc_gap_hours,
            min_points=args.adsc_min_points,
            max_speed_kt=args.adsc_max_speed_kt,
            max_records=args.adsc_max_records,
        )
        print(f"\nPipeline finished! Clean ADS-C datasets are ready in: {output_dir}\n")
        return
    
    end = datetime.now(timezone.utc).replace(microsecond=0)
    start = end - timedelta(hours=args.hours)
    
    flights = collect_flights(args.airports, int(start.timestamp()), int(end.timestamp()), output_dir)
    
    if not flights.empty and not args.no_tracks:
        collect_tracks(flights, output_dir)
        print(f"\nPipeline finished! Clean datasets are ready in: {output_dir}\n")
    else:
        print("No flights found for the selected time window.")


if __name__ == "__main__":
    main()
