"""
Phase 5 (multi-flight): Evaluate reconstruction across all track files.
This script is the multi-flight version of evaluate.py.
"""
import numpy as np
import pandas as pd
from pathlib import Path

from baseline import haversine_distance, interpolate_great_circle
from model import FusionTrajectoryModel


# ---------- Configuration ----------

TRACKS_DIR = Path("data/clean/tracks")
GAP_DURATION_SEC = 10 * 60        # 10 minutes (smaller so more flights qualify)
MIN_BEFORE_AFTER_POINTS = 10      # need at least this many on each side
MIN_TRUTH_POINTS = 5              # need at least this many hidden points


# ---------- Helpers ----------

def errors_km(true_lats, true_lons, pred_lats, pred_lons):
    n = min(len(true_lats), len(pred_lats))
    return np.array([
        haversine_distance(true_lats[i], true_lons[i], pred_lats[i], pred_lons[i])
        for i in range(n)
    ])


def path_length_km(lats, lons):
    total = 0.0
    for i in range(len(lats) - 1):
        total += haversine_distance(lats[i], lons[i], lats[i + 1], lons[i + 1])
    return total


def evaluate_one_flight(parquet_path):
    """
    Run baseline + smoother + filter on one flight.
    Returns a dict of results, or None if the flight can't be evaluated.
    """
    df = pd.read_parquet(parquet_path)
    df = df.sort_values("time").reset_index(drop=True)

    # Cut a gap from the middle
    mid_time = (df["time"].min() + df["time"].max()) / 2
    gap_start = mid_time - GAP_DURATION_SEC / 2
    gap_end   = mid_time + GAP_DURATION_SEC / 2

    before = df[df["time"] < gap_start]
    truth  = df[(df["time"] >= gap_start) & (df["time"] <= gap_end)]
    after  = df[df["time"] > gap_end]

    # Safety checks
    if len(before) < MIN_BEFORE_AFTER_POINTS:
        return None, "too few points before gap"
    if len(after) < MIN_BEFORE_AFTER_POINTS:
        return None, "too few points after gap"
    if len(truth) < MIN_TRUTH_POINTS:
        return None, "too few hidden truth points"

    # Baseline: great-circle interpolation between anchor points
    a_b_lat = before["latitude"].iloc[-1]
    a_b_lon = before["longitude"].iloc[-1]
    a_b_t   = before["time"].iloc[-1]
    a_a_lat = after["latitude"].iloc[0]
    a_a_lon = after["longitude"].iloc[0]
    a_a_t   = after["time"].iloc[0]
    span = a_a_t - a_b_t

    baseline_lats, baseline_lons = [], []
    for t in truth["time"].values:
        f = (t - a_b_t) / span
        lat, lon = interpolate_great_circle(a_b_lat, a_b_lon, a_a_lat, a_a_lon, f)
        baseline_lats.append(lat)
        baseline_lons.append(lon)
    baseline_lats = np.array(baseline_lats)
    baseline_lons = np.array(baseline_lons)

    # Kalman filter and smoother
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

    truth_lats = truth["latitude"].values
    truth_lons = truth["longitude"].values
    truth_length = path_length_km(truth_lats, truth_lons)

    out = {
        "flight": parquet_path.stem,
        "n_truth_points": len(truth),
        "truth_length_km": truth_length,
    }

    # Baseline metrics
    e = errors_km(truth_lats, truth_lons, baseline_lats, baseline_lons)
    out["baseline_median"] = np.median(e)
    out["baseline_p95"]    = np.percentile(e, 95)
    out["baseline_path_err"] = path_length_km(baseline_lats, baseline_lons) - truth_length

    # Smoother metrics
    if "smoother" in result:
        sm_lats, sm_lons, _ = result["smoother"]
        e = errors_km(truth_lats, truth_lons, sm_lats, sm_lons)
        out["smoother_median"] = np.median(e)
        out["smoother_p95"]    = np.percentile(e, 95)
        out["smoother_path_err"] = path_length_km(sm_lats, sm_lons) - truth_length

    # Filter metrics
    if "kalman" in result:
        kf_lats, kf_lons, _ = result["kalman"]
        e = errors_km(truth_lats, truth_lons, kf_lats, kf_lons)
        out["kalman_median"] = np.median(e)
        out["kalman_p95"]    = np.percentile(e, 95)
        out["kalman_path_err"] = path_length_km(kf_lats, kf_lons) - truth_length

    return out, None


# ---------- Main loop ----------

def main():
    track_files = sorted(TRACKS_DIR.glob("*.parquet"))
    print(f"Found {len(track_files)} track files\n")
    print(f"Gap duration: {GAP_DURATION_SEC // 60} minutes\n")

    rows = []
    skipped = []

    for i, f in enumerate(track_files, 1):
        try:
            result, reason = evaluate_one_flight(f)
            if result is None:
                skipped.append((f.name, reason))
                print(f"  [{i:2d}] {f.name:40s}  SKIP: {reason}")
                continue
            rows.append(result)
            print(f"  [{i:2d}] {f.name:40s}  "
                  f"baseline={result['baseline_median']:6.2f}  "
                  f"smoother={result['smoother_median']:6.2f}  "
                  f"kalman={result['kalman_median']:6.2f} km")
        except Exception as e:
            skipped.append((f.name, f"ERROR: {e}"))
            print(f"  [{i:2d}] {f.name:40s}  ERROR: {e}")

    print(f"\nEvaluated: {len(rows)} flights")
    print(f"Skipped:   {len(skipped)} flights")
    if not rows:
        print("\nNo flights successfully evaluated. Cannot compute summary.")
        return

    # ---------- Summary across all flights ----------
    df_results = pd.DataFrame(rows)

    print("\n" + "=" * 70)
    print("SUMMARY ACROSS ALL EVALUATED FLIGHTS")
    print("=" * 70)
    print(f"Number of flights: {len(df_results)}")
    print(f"Gap duration: {GAP_DURATION_SEC // 60} minutes\n")

    print(f"{'Method':<12} {'mean median':>12} {'med median':>12} {'mean p95':>12} {'med path err':>14}")
    print("-" * 64)

    for method in ["baseline", "smoother", "kalman"]:
        col_med = f"{method}_median"
        col_p95 = f"{method}_p95"
        col_pe  = f"{method}_path_err"
        if col_med not in df_results.columns:
            continue
        print(f"{method:<12} "
              f"{df_results[col_med].mean():>9.2f} km  "
              f"{df_results[col_med].median():>9.2f} km  "
              f"{df_results[col_p95].mean():>9.2f} km  "
              f"{df_results[col_pe].abs().median():>11.2f} km")

    # How often does each method WIN (lowest median error)?
    print("\nWin counts (lowest median error per flight):")
    for method in ["baseline", "smoother", "kalman"]:
        col = f"{method}_median"
        if col not in df_results.columns:
            continue
        # Compare to all other available methods on each row
        other_cols = [f"{m}_median" for m in ["baseline", "smoother", "kalman"]
                      if f"{m}_median" in df_results.columns and m != method]
        wins = (df_results[col] < df_results[other_cols].min(axis=1)).sum()
        print(f"  {method:<12} won on {wins:2d} of {len(df_results)} flights")

    # ---------- Save CSV ----------
    csv_path = Path("evaluation_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\nFull per-flight results saved to {csv_path}")


if __name__ == "__main__":
    main()