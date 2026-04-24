"""Evaluate baseline (great-circle) vs Kalman on held-out gap segments."""
import glob
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from baseline import fill_trajectory_gaps, haversine_distance, interpolate_great_circle
from model import FusionTrajectoryModel


TEST_TRACK_DIR = "data/clean/test_tracks"
GAP_DURATION_SEC = 20 * 60

if len(sys.argv) > 1:
    TRACK_FILE = sys.argv[1]
else:
    candidates = sorted(glob.glob(f"{TEST_TRACK_DIR}/*.parquet"))
    if not candidates:
        raise SystemExit(
            f"No parquet flights in {TEST_TRACK_DIR}/. "
            "Run data_pipeline.py first (test-ratio>0) or pass a file as arg."
        )
    TRACK_FILE = max(candidates, key=lambda p: pd.read_parquet(p, columns=["time"]).shape[0])

df = pd.read_parquet(TRACK_FILE)
df = df.sort_values("time").reset_index(drop=True)
print(f"Loaded {len(df)} points from {TRACK_FILE}")
print(f"  Duration: {(df['time'].max() - df['time'].min()) / 60:.1f} min")

mid_time = (df["time"].min() + df["time"].max()) / 2
gap_start = mid_time - GAP_DURATION_SEC / 2
gap_end   = mid_time + GAP_DURATION_SEC / 2

before = df[df["time"] <  gap_start].copy()
truth  = df[(df["time"] >= gap_start) & (df["time"] <= gap_end)].copy()
after  = df[df["time"] >  gap_end].copy()

print(f"  Before gap: {len(before)} points")
print(f"  Hidden (truth): {len(truth)} points")
print(f"  After gap: {len(after)} points")

if len(truth) == 0:
    raise SystemExit("Truth segment is empty — pick a longer flight or shorter gap.")


anchor_before_lat = before["latitude"].iloc[-1]
anchor_before_lon = before["longitude"].iloc[-1]
anchor_before_time = before["time"].iloc[-1]

anchor_after_lat = after["latitude"].iloc[0]
anchor_after_lon = after["longitude"].iloc[0]
anchor_after_time = after["time"].iloc[0]

total_gap_seconds = anchor_after_time - anchor_before_time

baseline_lats = []
baseline_lons = []
for t in truth["time"].values:
    fraction = (t - anchor_before_time) / total_gap_seconds
    lat, lon = interpolate_great_circle(
        anchor_before_lat, anchor_before_lon,
        anchor_after_lat, anchor_after_lon,
        fraction,
    )
    baseline_lats.append(lat)
    baseline_lons.append(lon)

baseline_lats = np.array(baseline_lats)
baseline_lons = np.array(baseline_lons)

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


def report(name, errs, pred_lats=None, pred_lons=None, truth_length=None):
    if len(errs) == 0:
        print(f"  {name}: NO POINTS PRODUCED")
        return
    line = (f"  {name:12s}  median={np.median(errs):7.2f} km   "
            f"p95={np.percentile(errs, 95):7.2f} km   "
            f"max={errs.max():7.2f} km")
    if pred_lats is not None and truth_length is not None:
        pred_length = path_length_km(pred_lats, pred_lons)
        diff = pred_length - truth_length
        pct = (diff / truth_length) * 100 if truth_length > 0 else 0
        line += f"   path_err={diff:+6.2f} km ({pct:+5.1f}%)"
    print(line)


truth_length = path_length_km(truth["latitude"].values, truth["longitude"].values)
print(f"\nTruth path length in gap: {truth_length:.2f} km")

errs_baseline = errors_km(
    truth["latitude"].values, truth["longitude"].values,
    baseline_lats, baseline_lons,
)
report("baseline", errs_baseline, baseline_lats, baseline_lons, truth_length)
if "smoother" in result:
    sm_lats, sm_lons, sm_alts = result["smoother"]
    errs_sm = errors_km(
        truth["latitude"].values, truth["longitude"].values, sm_lats, sm_lons
    )
    report("smoother", errs_sm, sm_lats, sm_lons, truth_length)

if "kalman" in result:
    kf_lats, kf_lons, kf_alts = result["kalman"]
    errs_kf = errors_km(
        truth["latitude"].values, truth["longitude"].values, kf_lats, kf_lons
    )
    report("kalman", errs_kf, kf_lats, kf_lons, truth_length)


fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(before["longitude"], before["latitude"], "b.", label="before gap", markersize=4)
ax.plot(after["longitude"],  after["latitude"],  "b.", label="after gap",  markersize=4)
ax.plot(truth["longitude"],  truth["latitude"],  "g-", label="TRUTH (hidden)", linewidth=2)

if "smoother" in result:
    sm_lats, sm_lons, _ = result["smoother"]
    ax.plot(sm_lons, sm_lats, "r-",  label="Kalman smoother", linewidth=2)
if "kalman" in result:
    kf_lats, kf_lons, _ = result["kalman"]
    ax.plot(kf_lons, kf_lats, "m--", label="Kalman filter", linewidth=1.5)
ax.plot(baseline_lons, baseline_lats, color="orange", linestyle="--",
        label="great-circle baseline", linewidth=1.5)

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title(f"Phase 5: gap reconstruction comparison ({GAP_DURATION_SEC // 60} min gap)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("phase5_comparison.png", dpi=120)
print("\nPlot saved to phase5_comparison.png")
plt.show()
