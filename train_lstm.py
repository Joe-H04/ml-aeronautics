"""
Train the bidirectional LSTM trajectory gap-filler on OpenSky ADS-B flights.

Loads parquet trajectories from data/clean/tracks/, splits by flight (no
leakage across train/val), trains a bidirectional LSTM that reconstructs a
GAP_LENGTH-point gap given CONTEXT_LENGTH points on each side, saves the
weights, and benchmarks on held-out gaps against the great-circle baseline.
"""

import glob
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping

from baseline import interpolate_great_circle
from model import LSTMTrajectoryModel


CONTEXT_LENGTH = 20
GAP_LENGTH = 10
EPOCHS = 50
BATCH_SIZE = 32
TRACK_DIR = "data/clean/tracks"
WEIGHTS_PATH = "weights/lstm_residual_trajectory.keras"
SEED = 42


def load_trajectories(track_dir: str
                      ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    trajectories = []
    min_len = 2 * CONTEXT_LENGTH + GAP_LENGTH
    for f in sorted(glob.glob(f"{track_dir}/*.parquet")):
        df = pd.read_parquet(f).sort_values("time").reset_index(drop=True)
        df = df.dropna(subset=["latitude", "longitude", "baro_altitude"])
        if len(df) < min_len:
            continue
        positions = df[["latitude", "longitude"]].to_numpy(dtype=np.float64)
        altitudes = df["baro_altitude"].to_numpy(dtype=np.float64)
        times = df["time"].to_numpy(dtype=np.float64)
        trajectories.append((positions, altitudes, times))
    return trajectories


def great_circle_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def evaluate_on_gaps(model: LSTMTrajectoryModel,
                     trajectories: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                     gaps_per_flight: int = 15):
    """Sample multiple gap positions from each held-out flight."""
    rng = np.random.default_rng(SEED)
    lstm_errs, great_circle_errs, linear_errs = [], [], []
    window = 2 * CONTEXT_LENGTH + GAP_LENGTH

    eligible = [i for i, (p, _, _) in enumerate(trajectories) if len(p) >= window]
    samples = [(i, s) for i in eligible
               for s in rng.integers(0, len(trajectories[i][0]) - window + 1,
                                     size=gaps_per_flight)]

    for idx, start in samples:
        positions, altitudes, times = trajectories[idx]
        start = int(start)

        before_slc = slice(start, start + CONTEXT_LENGTH)
        gap_slc = slice(start + CONTEXT_LENGTH, start + CONTEXT_LENGTH + GAP_LENGTH)
        after_slc = slice(start + CONTEXT_LENGTH + GAP_LENGTH, start + window)

        truth = positions[gap_slc]

        # LSTM reconstruction (uses both before and after context)
        lstm_lat, lstm_lon, _ = model.predict_gap(
            positions[before_slc], altitudes[before_slc], times[before_slc],
            positions[after_slc], altitudes[after_slc], times[after_slc],
        )
        lstm_errs.append(np.mean(great_circle_km(truth[:, 0], truth[:, 1], lstm_lat, lstm_lon)))

        # Great-circle baseline: interpolate between the two known endpoints of the gap
        p_before = positions[start + CONTEXT_LENGTH - 1]
        p_after = positions[start + CONTEXT_LENGTH + GAP_LENGTH]
        gc = np.array([
            interpolate_great_circle(p_before[0], p_before[1], p_after[0], p_after[1],
                                     (j + 1) / (GAP_LENGTH + 1))
            for j in range(GAP_LENGTH)
        ])
        great_circle_errs.append(np.mean(great_circle_km(truth[:, 0], truth[:, 1], gc[:, 0], gc[:, 1])))

        # Linear lat/lon interpolation — what "naive interpolation" in the project PDF means
        frac = np.linspace(1 / (GAP_LENGTH + 1), GAP_LENGTH / (GAP_LENGTH + 1), GAP_LENGTH)
        lin_lat = p_before[0] + (p_after[0] - p_before[0]) * frac
        lin_lon = p_before[1] + (p_after[1] - p_before[1]) * frac
        linear_errs.append(np.mean(great_circle_km(truth[:, 0], truth[:, 1], lin_lat, lin_lon)))

    return {
        "lstm": float(np.mean(lstm_errs)),
        "great_circle": float(np.mean(great_circle_errs)),
        "linear": float(np.mean(linear_errs)),
        "n_samples": len(samples),
    }


def main():
    np.random.seed(SEED)

    print(f"[1] Loading trajectories from {TRACK_DIR}/")
    trajectories = load_trajectories(TRACK_DIR)
    total_points = sum(len(p) for p, _, _ in trajectories)
    print(f"    Usable flights: {len(trajectories)}  total points: {total_points}")

    perm = np.random.permutation(len(trajectories))
    split = int(len(trajectories) * 0.8)
    train_trajs = [trajectories[i] for i in perm[:split]]
    val_trajs = [trajectories[i] for i in perm[split:]]
    print(f"    Train flights: {len(train_trajs)}  Val flights: {len(val_trajs)}")

    print(f"\n[2] Building bidirectional LSTM "
          f"(context_length={CONTEXT_LENGTH}, gap_length={GAP_LENGTH})")
    lstm = LSTMTrajectoryModel(context_length=CONTEXT_LENGTH, gap_length=GAP_LENGTH)
    lstm.model.summary()

    print("\n[3] Preparing training windows")
    X_train, y_train = lstm.prepare_training_data(train_trajs)
    X_val, y_val = lstm.prepare_training_data(val_trajs)
    print(f"    X_train={X_train.shape}  y_train={y_train.shape}")
    print(f"    X_val  ={X_val.shape}  y_val  ={y_val.shape}")

    if len(X_train) == 0:
        print("No training windows — aborting")
        return

    print(f"\n[4] Training ({EPOCHS} epochs, batch={BATCH_SIZE})")
    history = lstm.train(
        X_train, y_train,
        epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2,
        validation_data=(X_val, y_val) if len(X_val) else None,
        callbacks=[
            EarlyStopping(
                monitor="val_loss" if len(X_val) else "loss",
                patience=6,
                restore_best_weights=True,
            )
        ],
    )

    Path(WEIGHTS_PATH).parent.mkdir(parents=True, exist_ok=True)
    lstm.save(WEIGHTS_PATH)
    print(f"\n[5] Saved weights -> {WEIGHTS_PATH}")

    print("\n[6] Held-out gap benchmark (mean great-circle error per predicted point)")
    results = evaluate_on_gaps(lstm, val_trajs)
    print(f"    samples:                {results['n_samples']}")
    print(f"    LSTM gap-fill:          {results['lstm']:.3f} km")
    print(f"    Great-circle baseline:  {results['great_circle']:.3f} km")
    print(f"    Linear baseline:        {results['linear']:.3f} km")

    if results["great_circle"] > 0:
        gain = (results["great_circle"] - results["lstm"]) / results["great_circle"] * 100
        print(f"\n    LSTM vs great-circle baseline: {gain:+.1f}%")


if __name__ == "__main__":
    main()
