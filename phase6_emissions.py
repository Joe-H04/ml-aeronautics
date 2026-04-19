"""
Phase 6: Distance and emissions estimation from reconstructed trajectories.
Compares how accurately the baseline and smoother estimate the
distance and CO2 of the GAP segment for each flight.
"""
import numpy as np
import pandas as pd
from pathlib import Path

from baseline import haversine_distance, interpolate_great_circle
from model import FusionTrajectoryModel


# ---------- Configuration ----------

TRACKS_DIR = Path("data/clean/tracks")
GAP_DURATION_SEC = 10 * 60
MIN_BEFORE_AFTER_POINTS = 10
MIN_TRUTH_POINTS = 5

# Simple emissions model (cruise narrow-body):
#   ~3.0 kg fuel per km
#   ~3.16 kg CO2 per kg fuel
# So: co2_kg = distance_km * 9.48
FUEL_KG_PER_KM = 3.0
CO2_KG_PER_FUEL_KG = 3.16
CO2_KG_PER_KM = FUEL_KG_PER_KM * CO2_KG_PER_FUEL_KG  # ~9.48


# ---------- Helpers ----------

def path_length_km(lats, lons):
    total = 0.0
    for i in range(len(lats) - 1):
        total += haversine_distance(lats[i], lons[i], lats[i + 1], lons[i + 1])
    return total


def km_to_co2_kg(km):
    return km * CO2_KG_PER_KM


def evaluate_one_flight_emissions(parquet_path):
    """
    Compute distance and CO2 for the truth gap, baseline reconstruction,
    and smoother reconstruction. Returns a dict or None if the flight
    can't be evaluated.
    """
    df = pd.read_parquet(parquet_path)
    df = df.sort_values("time").reset_index(drop=True)

    mid_time = (df["time"].min() + df["time"].max()) / 2
    gap_start = mid_time - GAP_DURATION_SEC / 2
    gap_end   = mid_time + GAP_DURATION_SEC / 2

    before = df[df["time"] < gap_start]
    truth  = df[(df["time"] >= gap_start) & (df["time"] <= gap_end)]
    after  = df[df["time"] > gap_end]

    if len(before) < MIN_BEFORE_AFTER_POINTS:
        return None
    if len(after) < MIN_BEFORE_AFTER_POINTS:
        return None
    if len(truth) < MIN_TRUTH_POINTS:
        return None

    # Truth gap distance and CO2
    truth_lats = truth["latitude"].values
    truth_lons = truth["longitude"].values
    truth_dist = path_length_km(truth_lats, truth_lons)
    truth_co2  = km_to_co2_kg(truth_dist)

    # Baseline gap reconstruction
    a_b_lat = before["latitude"].iloc[-1]
    a_b_lon = before["longitude"].iloc[-1]
    a_b_t   = before["time"].iloc[-1]
    a_a_lat = after["latitude"].iloc[0]
    a_a_lon = after["longitude"].iloc[0]
    a_a_t   = after["time"].iloc[0]
    span = a_a_t - a_b_t

    base_lats, base_lons = [], []
    for t in truth["time"].values:
        f = (t - a_b_t) / span
        lat, lon = interpolate_great_circle(a_b_lat, a_b_lon, a_a_lat, a_a_lon, f)
        base_lats.append(lat)
        base_lons.append(lon)
    base_dist = path_length_km(np.array(base_lats), np.array(base_lons))
    base_co2  = km_to_co2_kg(base_dist)

    # Smoother gap reconstruction
    fusion = FusionTrajectoryModel()
    result = fusion.reconstruct_gap(
        before_lat=before["latitude"].values,
        before_lon=before["longitude"].values,
        before_alt=before["baro_altitude"].values,
        before_times=before["time"].values,
        after_lat=after["latitude"].values,
        after_lon=after["longitude"].values,
        after_alt=after["baro_altitude"].values,
        after_times=after["time"].values,
        gap_times=truth["time"].values,
        method="smoother",
    )
    if "smoother" not in result:
        return None
    sm_lats, sm_lons, _ = result["smoother"]
    sm_dist = path_length_km(sm_lats, sm_lons)
    sm_co2  = km_to_co2_kg(sm_dist)
    # ---------- Full-trajectory totals ----------
    # Build the complete reconstructed flight by stitching:
    # before-gap + reconstructed-gap + after-gap
    before_lats = before["latitude"].values
    before_lons = before["longitude"].values
    after_lats  = after["latitude"].values
    after_lons  = after["longitude"].values

    # Total distance using the TRUE gap (the perfect reconstruction)
    truth_full_lats = np.concatenate([before_lats, truth_lats, after_lats])
    truth_full_lons = np.concatenate([before_lons, truth_lons, after_lons])
    truth_full_dist = path_length_km(truth_full_lats, truth_full_lons)

    # Total distance using the BASELINE reconstruction
    base_full_lats = np.concatenate([before_lats, np.array(base_lats), after_lats])
    base_full_lons = np.concatenate([before_lons, np.array(base_lons), after_lons])
    base_full_dist = path_length_km(base_full_lats, base_full_lons)

    # Total distance using the SMOOTHER reconstruction
    sm_full_lats = np.concatenate([before_lats, sm_lats, after_lats])
    sm_full_lons = np.concatenate([before_lons, sm_lons, after_lons])
    sm_full_dist = path_length_km(sm_full_lats, sm_full_lons)

    return {
        "flight": parquet_path.stem,
        "truth_dist_km": truth_dist,
        "truth_co2_kg":  truth_co2,
        "base_dist_km":  base_dist,
        "base_co2_kg":   base_co2,
        "base_dist_err": base_dist - truth_dist,
        "base_co2_err":  base_co2 - truth_co2,
        "sm_dist_km":    sm_dist,
        "sm_co2_kg":     sm_co2,
        "sm_dist_err":   sm_dist - truth_dist,
        "sm_co2_err":    sm_co2 - truth_co2,
        "truth_full_dist_km":   truth_full_dist,
        "truth_full_co2_kg":    km_to_co2_kg(truth_full_dist),
        "base_full_dist_km":    base_full_dist,
        "base_full_dist_err":   base_full_dist - truth_full_dist,
        "base_full_co2_err":    km_to_co2_kg(base_full_dist - truth_full_dist),
        "sm_full_dist_km":      sm_full_dist,
        "sm_full_dist_err":     sm_full_dist - truth_full_dist,
        "sm_full_co2_err":      km_to_co2_kg(sm_full_dist - truth_full_dist),
    }


# ---------- Main loop ----------

def main():
    track_files = sorted(TRACKS_DIR.glob("*.parquet"))
    print(f"Found {len(track_files)} track files")
    print(f"Gap duration: {GAP_DURATION_SEC // 60} minutes")
    print(f"Emissions model: {CO2_KG_PER_KM:.2f} kg CO2 per km of flight\n")

    rows = []
    for i, f in enumerate(track_files, 1):
        try:
            r = evaluate_one_flight_emissions(f)
            if r is None:
                continue
            rows.append(r)
            print(f"  [{i:2d}] {f.name:38s}  "
                  f"full={r['truth_full_dist_km']:7.1f} km   "
                  f"base_err={r['base_full_dist_err']:+6.2f} km   "
                  f"sm_err={r['sm_full_dist_err']:+6.2f} km   "
                  f"sm_co2_err={r['sm_full_co2_err']:+7.1f} kg")
        except Exception as e:
            print(f"  [{i:2d}] {f.name:38s}  ERROR: {e}")

    print(f"\nEvaluated: {len(rows)} flights")
    if not rows:
        return

    df = pd.DataFrame(rows)

    print("\n" + "=" * 70)
    print("PHASE 6 SUMMARY: distance & emissions estimation accuracy")
    print("=" * 70)
    print(f"Number of flights: {len(df)}")
    print(f"Gap duration: {GAP_DURATION_SEC // 60} minutes")
    print(f"Emissions model: {CO2_KG_PER_KM:.2f} kg CO2 per km\n")

    print("--- GAP SEGMENT ONLY ---")
    print(f"  baseline   mean |dist err| = {df['base_dist_err'].abs().mean():6.2f} km   "
          f"mean |CO2 err| = {df['base_co2_err'].abs().mean():7.1f} kg")
    print(f"  smoother   mean |dist err| = {df['sm_dist_err'].abs().mean():6.2f} km   "
          f"mean |CO2 err| = {df['sm_co2_err'].abs().mean():7.1f} kg")

    print("\n--- FULL FLIGHT (before + reconstructed gap + after) ---")
    print(f"  baseline   mean |dist err| = {df['base_full_dist_err'].abs().mean():6.2f} km   "
          f"mean |CO2 err| = {df['base_full_co2_err'].abs().mean():7.1f} kg")
    print(f"  smoother   mean |dist err| = {df['sm_full_dist_err'].abs().mean():6.2f} km   "
          f"mean |CO2 err| = {df['sm_full_co2_err'].abs().mean():7.1f} kg")

    print("\n--- WORST-CASE FLIGHTS (smoother CO2 error) ---")
    worst = df.reindex(df['sm_full_co2_err'].abs().sort_values(ascending=False).index).head(3)
    for _, row in worst.iterrows():
        print(f"  {row['flight']:30s}  full={row['truth_full_dist_km']:7.1f} km   "
              f"smoother CO2 err = {row['sm_full_co2_err']:+7.1f} kg")

    # CSV output
    csv_path = Path("phase6_emissions_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nFull per-flight results saved to {csv_path}")


if __name__ == "__main__":
    main()