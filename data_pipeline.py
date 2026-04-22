"""Ingest flight trajectories from the OpenSky Trino historical database.

One query per day-partition (the `day` partition column in `flights_data4`),
filtered to the configured airports, with the embedded `track` column unnested
to per-point rows. Output schema is compatible with the downstream trainers
(baseline.py, model.py, train_lstm.py).

Flights are split deterministically into train / test directories via a CRC32
hash on (icao24, firstseen) so the same flight always lands in the same split
across runs.

Read RULES.MD before changing this file — ignoring OpenSky's efficient-use
guidelines gets the account banned.
"""

import argparse
import shutil
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
from pyopensky.trino import Trino

DEFAULT_AIRPORTS = ["EDDF", "EGLL", "LFPG", "EHAM", "LEMD",
                    "LIRF", "LOWW", "LSZH", "EKCH", "EDDM"]
METADATA_COLS = ["icao24", "firstseen", "lastseen",
                 "estdepartureairport", "estarrivalairport", "callsign"]
TRACK_RENAME = {
    "tp_time": "time",
    "tp_lat": "latitude",
    "tp_lon": "longitude",
    "tp_alt": "baro_altitude",
    "tp_heading": "true_track",
    "tp_onground": "on_ground",
}
MAX_ALTITUDE_M = 15000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--airports", nargs="+", default=DEFAULT_AIRPORTS)
    p.add_argument("--days", type=int, default=2,
                   help="Number of past UTC days to ingest (excludes today)")
    p.add_argument("--output", default="data/clean",
                   help="Root for flights/, tracks/ (train)")
    p.add_argument("--test-output", default="data/clean/test_tracks",
                   help="Dir for held-out test flight parquets")
    p.add_argument("--test-ratio", type=float, default=0.2,
                   help="Fraction of flights routed to test dir "
                        "(deterministic hash, 0 disables)")
    p.add_argument("--clean", action="store_true",
                   help="Wipe tracks/ and test-output dirs before writing")
    p.add_argument("--no-tracks", action="store_true")
    return p.parse_args()


def day_partitions(days: int) -> Iterable[int]:
    today = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    for i in range(days, 0, -1):
        yield int((today - timedelta(days=i)).timestamp())


def fetch_day(trino: Trino, day: int, airports: list[str]) -> pd.DataFrame:
    """One partitioned query per day: flights + unnested trajectories."""
    airport_list = ",".join(f"'{a.upper()}'" for a in airports)
    sql = f"""
    SELECT f.icao24, f.firstseen, f.lastseen,
           f.estdepartureairport, f.estarrivalairport, f.callsign,
           tp_time, tp_lat, tp_lon, tp_alt, tp_heading, tp_onground
    FROM flights_data4 f
    CROSS JOIN UNNEST(f.track) AS t(
        tp_time, tp_lat, tp_lon, tp_alt, tp_heading, tp_onground
    )
    WHERE f.day = {day}
      AND (f.estdepartureairport IN ({airport_list})
           OR f.estarrivalairport IN ({airport_list}))
    """
    return trino.query(sql)


def clean_track(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("time").drop_duplicates("time")
    df = df.dropna(subset=["latitude", "longitude", "baro_altitude"])
    on_ground = df["on_ground"].astype("boolean").fillna(False)
    return df[(df["baro_altitude"] < MAX_ALTITUDE_M) & (~on_ground)]


def assign_to_test(icao24: str, firstseen: int, ratio: float) -> bool:
    if ratio <= 0:
        return False
    key = f"{icao24}|{int(firstseen)}".encode()
    return (zlib.crc32(key) % 100) < int(ratio * 100)


def write_tracks(raw: pd.DataFrame, train_dir: Path, test_dir: Path,
                 test_ratio: float) -> tuple[int, int, int, int]:
    train_dir.mkdir(parents=True, exist_ok=True)
    if test_ratio > 0:
        test_dir.mkdir(parents=True, exist_ok=True)
    df = raw.rename(columns=TRACK_RENAME)

    raw_total = clean_total = train_written = test_written = 0
    for (icao24, first), group in df.groupby(["icao24", "firstseen"], sort=False):
        raw_total += len(group)
        clean = clean_track(group)
        clean_total += len(clean)
        if clean.empty:
            continue

        callsign = (group["callsign"].dropna().iloc[0]
                    if group["callsign"].notna().any() else "")
        clean = clean[["time", "latitude", "longitude", "baro_altitude",
                       "true_track", "on_ground"]].copy()
        clean["icao24"] = icao24
        clean["callsign"] = str(callsign).strip()
        clean["flight_start"] = int(first)
        clean["flight_end"] = int(group["lastseen"].iloc[0])

        target_dir = (test_dir
                      if assign_to_test(icao24, int(first), test_ratio)
                      else train_dir)
        clean.to_parquet(
            target_dir / f"{icao24}_{int(first)}.parquet", index=False
        )
        if target_dir is test_dir:
            test_written += 1
        else:
            train_written += 1

    return raw_total, clean_total, train_written, test_written


def write_flights(flights: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"flights_{datetime.now():%Y%m%d_%H%M%S}.parquet"
    flights.to_parquet(path, index=False)
    return path


def main():
    args = parse_args()
    trino = Trino()
    out_dir = Path(args.output)
    train_dir = out_dir / "tracks"
    test_dir = Path(args.test_output)
    airports = [a.upper() for a in args.airports]

    if args.clean:
        for d in (train_dir, test_dir):
            if d.exists():
                print(f"[CLEAN] wiping {d}")
                shutil.rmtree(d)

    print(f"[PIPELINE] ingesting {args.days} day(s) for airports {airports}")
    print(f"           train tracks -> {train_dir}")
    print(f"           test  tracks -> {test_dir}  (ratio={args.test_ratio})")

    flight_metas = []
    total_raw = total_clean = total_train = total_test = 0

    for day in day_partitions(args.days):
        day_iso = datetime.fromtimestamp(day, tz=timezone.utc).date().isoformat()
        print(f"\n[DAY {day_iso}] querying flights_data4 (day={day})")

        df = fetch_day(trino, day, airports)
        if df.empty:
            print("  no flights returned")
            continue

        meta = (df[METADATA_COLS]
                .drop_duplicates(["icao24", "firstseen", "lastseen"]))
        flight_metas.append(meta)
        print(f"  flights={len(meta)}  raw track points={len(df)}")

        if not args.no_tracks:
            raw, clean, n_train, n_test = write_tracks(
                df, train_dir, test_dir, args.test_ratio
            )
            total_raw += raw
            total_clean += clean
            total_train += n_train
            total_test += n_test
            print(f"  wrote train={n_train}  test={n_test}  "
                  f"({raw} -> {clean} points)")

    if not flight_metas:
        print("\nNo flights found across the selected days.")
        return

    all_meta = (pd.concat(flight_metas, ignore_index=True)
                .drop_duplicates(["icao24", "firstseen", "lastseen"]))
    flights_path = write_flights(all_meta, out_dir / "flights")

    print("\n[SUMMARY]")
    print(f"  unique flights:       {len(all_meta)}")
    print(f"  train track files:    {total_train}")
    print(f"  test  track files:    {total_test}")
    print(f"  raw points:           {total_raw}")
    print(f"  clean points:         {total_clean}")
    if total_raw > 0:
        rate = (total_raw - total_clean) / total_raw * 100
        print(f"  noise removed:        {rate:.1f}%")
    print(f"  flights metadata ->   {flights_path}")


if __name__ == "__main__":
    main()
