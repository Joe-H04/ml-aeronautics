"""Distance and emissions sensitivity from LSTM-reconstructed tracks."""

from pathlib import Path

import numpy as np
import pandas as pd

from baseline import interpolate_great_circle
from model import LSTMTrajectoryModel, great_circle_km


TRACKS_DIR = Path("data/clean/tracks")
WEIGHTS_PATH = Path("weights/lstm_residual_trajectory.keras")
OUTPUT_CSV = Path("phase6_emissions_results.csv")

CONTEXT_LENGTH = 20
GAP_LENGTH = 10

# Emissions proxy: 3.0 kg fuel per km, 3.16 kg CO2 per kg fuel.
FUEL_KG_PER_KM = 3.0
CO2_KG_PER_FUEL_KG = 3.16
CO2_KG_PER_KM = FUEL_KG_PER_KM * CO2_KG_PER_FUEL_KG


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


def km_to_co2_kg(km):
    return km * CO2_KG_PER_KM


def interpolation_baselines(p_before, p_after):
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


def replace_gap(full_positions, gap_slc, reconstruction):
    reconstructed = full_positions.copy()
    reconstructed[gap_slc] = reconstruction
    return reconstructed


def evaluate_one_flight(model, path):
    track = load_track(path)
    if track is None:
        return None

    positions, altitudes, times = track
    window = 2 * CONTEXT_LENGTH + GAP_LENGTH
    start = (len(positions) - window) // 2

    before_slc = slice(start, start + CONTEXT_LENGTH)
    gap_slc = slice(start + CONTEXT_LENGTH, start + CONTEXT_LENGTH + GAP_LENGTH)
    after_slc = slice(start + CONTEXT_LENGTH + GAP_LENGTH, start + window)

    truth_gap = positions[gap_slc]
    truth_gap_dist = path_length_km(truth_gap)
    truth_full_dist = path_length_km(positions)

    lstm_lat, lstm_lon, _ = model.predict_gap(
        positions[before_slc],
        altitudes[before_slc],
        times[before_slc],
        positions[after_slc],
        altitudes[after_slc],
        times[after_slc],
    )
    lstm_gap = np.column_stack([lstm_lat, lstm_lon])

    p_before = positions[start + CONTEXT_LENGTH - 1]
    p_after = positions[start + CONTEXT_LENGTH + GAP_LENGTH]
    great_circle_gap, linear_gap = interpolation_baselines(p_before, p_after)

    row = {
        "flight": path.stem,
        "truth_gap_dist_km": truth_gap_dist,
        "truth_gap_co2_kg": km_to_co2_kg(truth_gap_dist),
        "truth_full_dist_km": truth_full_dist,
        "truth_full_co2_kg": km_to_co2_kg(truth_full_dist),
    }

    for method, gap_positions in [
        ("lstm", lstm_gap),
        ("great_circle", great_circle_gap),
        ("linear", linear_gap),
    ]:
        gap_dist = path_length_km(gap_positions)
        full_dist = path_length_km(replace_gap(positions, gap_slc, gap_positions))

        row[f"{method}_gap_dist_km"] = gap_dist
        row[f"{method}_gap_dist_err_km"] = gap_dist - truth_gap_dist
        row[f"{method}_gap_co2_err_kg"] = km_to_co2_kg(gap_dist - truth_gap_dist)
        row[f"{method}_full_dist_km"] = full_dist
        row[f"{method}_full_dist_err_km"] = full_dist - truth_full_dist
        row[f"{method}_full_co2_err_kg"] = km_to_co2_kg(full_dist - truth_full_dist)

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

    rows = []
    track_files = sorted(TRACKS_DIR.glob("*.parquet"))
    print(f"Found {len(track_files)} track files")
    print(f"Emissions proxy: {CO2_KG_PER_KM:.2f} kg CO2 per km\n")

    for i, path in enumerate(track_files, 1):
        result = evaluate_one_flight(model, path)
        if result is None:
            continue
        rows.append(result)
        if len(rows) % 100 == 0:
            print(f"  evaluated {len(rows)} eligible flights...")

    print(f"\nEvaluated: {len(rows)} flights")
    if not rows:
        return

    results = pd.DataFrame(rows)

    print("\n" + "=" * 76)
    print("LSTM DISTANCE AND EMISSIONS SENSITIVITY")
    print("=" * 76)

    for scope in ["gap", "full"]:
        print(f"\n--- {scope.upper()} TRAJECTORY ---")
        for method in ["lstm", "great_circle", "linear"]:
            dist_col = f"{method}_{scope}_dist_err_km"
            co2_col = f"{method}_{scope}_co2_err_kg"
            print(
                f"  {method:<14} "
                f"mean |dist err| = {results[dist_col].abs().mean():8.3f} km   "
                f"mean |CO2 err| = {results[co2_col].abs().mean():9.1f} kg"
            )

    results.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved per-flight results to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
