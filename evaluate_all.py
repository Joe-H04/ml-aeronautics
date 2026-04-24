"""Evaluate the trained LSTM gap-filler on held-out test tracks."""

from pathlib import Path

import numpy as np
import pandas as pd

from baseline import interpolate_great_circle
from model import LSTMTrajectoryModel, great_circle_km


TRACKS_DIR = Path("data/clean/test_tracks")
WEIGHTS_PATH = Path("weights/lstm_residual_trajectory.keras")
OUTPUT_CSV = Path("evaluation_results.csv")

CONTEXT_LENGTH = 20
GAP_LENGTH = 10
GAPS_PER_FLIGHT = 3
SEED = 42


def load_track(path):
    df = pd.read_parquet(path).sort_values("time").reset_index(drop=True)
    df = df.dropna(subset=["latitude", "longitude", "baro_altitude"])

    min_len = 2 * CONTEXT_LENGTH + GAP_LENGTH
    if len(df) < min_len:
        return None

    return (
        df[["latitude", "longitude"]].to_numpy(dtype=np.float64),
        df["baro_altitude"].to_numpy(dtype=np.float64),
        df["time"].to_numpy(dtype=np.float64),
    )


def path_length_km(positions):
    if len(positions) < 2:
        return 0.0
    return float(np.sum(
        great_circle_km(
            positions[:-1, 0],
            positions[:-1, 1],
            positions[1:, 0],
            positions[1:, 1],
        )
    ))


def interpolate_baselines(p_before, p_after):
    fractions = np.linspace(
        1 / (GAP_LENGTH + 1),
        GAP_LENGTH / (GAP_LENGTH + 1),
        GAP_LENGTH,
        dtype=np.float64,
    )

    great_circle = np.array([
        interpolate_great_circle(
            p_before[0],
            p_before[1],
            p_after[0],
            p_after[1],
            fraction,
        )
        for fraction in fractions
    ])

    linear = p_before + (p_after - p_before) * fractions[:, np.newaxis]
    return great_circle, linear


def error_summary(truth, prediction):
    errors = great_circle_km(
        truth[:, 0],
        truth[:, 1],
        prediction[:, 0],
        prediction[:, 1],
    )
    return {
        "mean_error_km": float(np.mean(errors)),
        "median_error_km": float(np.median(errors)),
        "p95_error_km": float(np.percentile(errors, 95)),
        "path_error_km": float(path_length_km(prediction) - path_length_km(truth)),
    }


def evaluate_window(model, path, positions, altitudes, times, start):
    window = 2 * CONTEXT_LENGTH + GAP_LENGTH

    before_slc = slice(start, start + CONTEXT_LENGTH)
    gap_slc = slice(start + CONTEXT_LENGTH, start + CONTEXT_LENGTH + GAP_LENGTH)
    after_slc = slice(start + CONTEXT_LENGTH + GAP_LENGTH, start + window)

    truth = positions[gap_slc]

    lstm_lat, lstm_lon, _ = model.predict_gap(
        positions[before_slc],
        altitudes[before_slc],
        times[before_slc],
        positions[after_slc],
        altitudes[after_slc],
        times[after_slc],
    )
    lstm_prediction = np.column_stack([lstm_lat, lstm_lon])

    p_before = positions[start + CONTEXT_LENGTH - 1]
    p_after = positions[start + CONTEXT_LENGTH + GAP_LENGTH]
    great_circle, linear = interpolate_baselines(p_before, p_after)

    row = {
        "flight": path.stem,
        "start_index": int(start),
        "truth_path_km": path_length_km(truth),
    }

    for method, prediction in [
        ("lstm", lstm_prediction),
        ("great_circle", great_circle),
        ("linear", linear),
    ]:
        summary = error_summary(truth, prediction)
        for metric, value in summary.items():
            row[f"{method}_{metric}"] = value

    return row


def main():
    if not WEIGHTS_PATH.exists():
        raise SystemExit(
            f"No trained LSTM weights found at {WEIGHTS_PATH}. "
            "Run `python train_lstm.py` first."
        )

    print(f"Loading LSTM weights from {WEIGHTS_PATH}")
    model = LSTMTrajectoryModel.load(
        str(WEIGHTS_PATH),
        context_length=CONTEXT_LENGTH,
        gap_length=GAP_LENGTH,
    )

    rng = np.random.default_rng(SEED)
    rows = []
    eligible_flights = 0
    skipped_flights = 0
    window = 2 * CONTEXT_LENGTH + GAP_LENGTH

    track_files = sorted(TRACKS_DIR.glob("*.parquet"))
    print(f"Found {len(track_files)} track files")
    print(f"Gap setup: {CONTEXT_LENGTH} before + {GAP_LENGTH} hidden + {CONTEXT_LENGTH} after")

    for path in track_files:
        track = load_track(path)
        if track is None:
            skipped_flights += 1
            continue

        eligible_flights += 1
        positions, altitudes, times = track
        starts = rng.integers(0, len(positions) - window + 1, size=GAPS_PER_FLIGHT)

        for start in starts:
            rows.append(evaluate_window(model, path, positions, altitudes, times, int(start)))

        if eligible_flights % 100 == 0:
            print(f"  evaluated {eligible_flights} eligible flights...")

    print(f"\nEligible flights: {eligible_flights}")
    print(f"Skipped flights:  {skipped_flights}")
    print(f"Gap samples:      {len(rows)}")

    if not rows:
        print("No eligible windows found.")
        return

    results = pd.DataFrame(rows)
    methods = ["lstm", "great_circle", "linear"]

    print("\n" + "=" * 76)
    print("LSTM GAP-FILLING EVALUATION")
    print("=" * 76)
    print(f"{'Method':<14} {'mean':>10} {'median':>10} {'p95':>10} {'med path err':>14}")
    print("-" * 76)

    for method in methods:
        print(
            f"{method:<14} "
            f"{results[f'{method}_mean_error_km'].mean():>8.3f} km  "
            f"{results[f'{method}_mean_error_km'].median():>8.3f} km  "
            f"{results[f'{method}_p95_error_km'].mean():>8.3f} km  "
            f"{results[f'{method}_path_error_km'].abs().median():>11.3f} km"
        )

    win_cols = [f"{method}_mean_error_km" for method in methods]
    wins = results[win_cols].idxmin(axis=1).str.replace("_mean_error_km", "")
    print("\nWin counts (lowest mean error per hidden gap):")
    for method in methods:
        print(f"  {method:<14} {int((wins == method).sum())} of {len(results)}")

    baseline = results["great_circle_mean_error_km"].mean()
    lstm = results["lstm_mean_error_km"].mean()
    if baseline > 0:
        gain = (baseline - lstm) / baseline * 100
        print(f"\nLSTM vs great-circle baseline: {gain:+.1f}%")

    results.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved per-gap results to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
