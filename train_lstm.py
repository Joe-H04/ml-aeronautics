"""Train the bidirectional LSTM trajectory gap-filler on OpenSky ADS-B flights.

Trains on data/clean/tracks/ (split 80/20 into train/val), then benchmarks on
the held-out data/clean/test_tracks/ flights which were never seen during training.
"""

import argparse
import glob
import platform
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping

try:
    import tensorflow as tf
except ImportError:
    tf = None

from baseline import interpolate_great_circle
from model import LSTMTrajectoryModel


CONTEXT_LENGTH = 20
GAP_LENGTH = 10
EPOCHS = 50
BATCH_SIZE = 128
WINDOW_STRIDE = 4
TRACK_DIR = "data/clean/tracks"
TEST_TRACK_DIR = "data/clean/test_tracks"
WEIGHTS_PATH = "weights/lstm_residual_trajectory.keras"
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the bidirectional LSTM trajectory gap-filler."
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Training epochs (default: {EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Mini-batch size (default: {BATCH_SIZE})")
    parser.add_argument("--window-stride", type=int, default=WINDOW_STRIDE,
                        help=f"Use every Nth training window (default: {WINDOW_STRIDE})")
    parser.add_argument("--mixed-precision", dest="mixed_precision", action="store_true",
                        help="Enable TensorFlow mixed precision on GPU")
    parser.add_argument("--no-mixed-precision", dest="mixed_precision", action="store_false",
                        help="Disable TensorFlow mixed precision")
    parser.set_defaults(mixed_precision=True)
    return parser.parse_args()


def configure_runtime(args: argparse.Namespace) -> None:
    if tf is None:
        return

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    if gpus and args.mixed_precision:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")


def describe_runtime(args: argparse.Namespace) -> None:
    print(f"[0] Python {platform.python_version()} on {platform.system()}")

    if tf is None:
        print("    TensorFlow is not installed in this environment.")
        return

    print(f"    TensorFlow {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        gpu_names = ", ".join(gpu.name for gpu in gpus)
        print(f"    GPU devices: {gpu_names}")
        policy_name = tf.keras.mixed_precision.global_policy().name
        print(f"    Mixed precision: {policy_name}")
        print(f"    Training config: epochs={args.epochs}, batch={args.batch_size}, "
              f"window_stride={args.window_stride}")
        return

    print("    No TensorFlow GPU detected; training will run on CPU.")
    print(f"    Training config: epochs={args.epochs}, batch={args.batch_size}, "
          f"window_stride={args.window_stride}")
    if platform.system() == "Windows":
        print("    Native Windows TensorFlow 2.11+ does not use NVIDIA CUDA GPUs.")
        print("    Use WSL2 + Linux TensorFlow for RTX 5070/5070 Ti acceleration.")


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

        lstm_lat, lstm_lon, _ = model.predict_gap(
            positions[before_slc], altitudes[before_slc], times[before_slc],
            positions[after_slc], altitudes[after_slc], times[after_slc],
        )
        lstm_errs.append(np.mean(great_circle_km(truth[:, 0], truth[:, 1], lstm_lat, lstm_lon)))

        p_before = positions[start + CONTEXT_LENGTH - 1]
        p_after = positions[start + CONTEXT_LENGTH + GAP_LENGTH]
        gc = np.array([
            interpolate_great_circle(p_before[0], p_before[1], p_after[0], p_after[1],
                                     (j + 1) / (GAP_LENGTH + 1))
            for j in range(GAP_LENGTH)
        ])
        great_circle_errs.append(np.mean(great_circle_km(truth[:, 0], truth[:, 1], gc[:, 0], gc[:, 1])))

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
    args = parse_args()
    configure_runtime(args)
    describe_runtime(args)
    np.random.seed(SEED)

    print(f"[1] Loading trajectories from {TRACK_DIR}/ (train) "
          f"and {TEST_TRACK_DIR}/ (held-out test)")
    trajectories = load_trajectories(TRACK_DIR)
    test_trajs = load_trajectories(TEST_TRACK_DIR)
    total_points = sum(len(p) for p, _, _ in trajectories)
    test_points = sum(len(p) for p, _, _ in test_trajs)
    print(f"    Train/val flights: {len(trajectories)}  total points: {total_points}")
    print(f"    Test flights:      {len(test_trajs)}  total points: {test_points}")

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
    X_train, y_train = lstm.prepare_training_data(
        train_trajs,
        window_stride=args.window_stride,
    )
    X_val, y_val = lstm.prepare_training_data(
        val_trajs,
        window_stride=args.window_stride,
    )
    print(f"    X_train={X_train.shape}  y_train={y_train.shape}")
    print(f"    X_val  ={X_val.shape}  y_val  ={y_val.shape}")

    if len(X_train) == 0:
        print("No training windows — aborting")
        return

    print(f"\n[4] Training ({args.epochs} epochs, batch={args.batch_size}, "
          f"window_stride={args.window_stride})")
    history = lstm.train(
        X_train, y_train,
        epochs=args.epochs, batch_size=args.batch_size, verbose=1,
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

    print("\n[6] Validation-set gap benchmark (sanity check, seen during training split)")
    val_results = evaluate_on_gaps(lstm, val_trajs)
    print(f"    samples:                {val_results['n_samples']}")
    print(f"    LSTM gap-fill:          {val_results['lstm']:.3f} km")
    print(f"    Great-circle baseline:  {val_results['great_circle']:.3f} km")
    print(f"    Linear baseline:        {val_results['linear']:.3f} km")

    if not test_trajs:
        print(f"\n[7] No flights in {TEST_TRACK_DIR}/ — skipping held-out test benchmark")
        return

    print(f"\n[7] Held-out TEST benchmark on {TEST_TRACK_DIR}/ "
          "(flights never seen during training)")
    test_results = evaluate_on_gaps(lstm, test_trajs)
    print(f"    samples:                {test_results['n_samples']}")
    print(f"    LSTM gap-fill:          {test_results['lstm']:.3f} km")
    print(f"    Great-circle baseline:  {test_results['great_circle']:.3f} km")
    print(f"    Linear baseline:        {test_results['linear']:.3f} km")

    if test_results["great_circle"] > 0:
        gain = (test_results["great_circle"] - test_results["lstm"]) / test_results["great_circle"] * 100
        print(f"\n    LSTM vs great-circle baseline (test): {gain:+.1f}%")


if __name__ == "__main__":
    main()
