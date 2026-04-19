import argparse
import os
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
TOKEN = {"value": None, "exp": 0.0}
SESSION = requests.Session()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--airports", nargs="+", default=DEFAULT_AIRPORTS)
    p.add_argument("--hours", type=int, default=4)
    p.add_argument("--output", default="data/clean")
    p.add_argument("--no-tracks", action="store_true")
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