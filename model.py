"""
Trajectory gap-filling models for aircraft reconstruction.

The project model is a bidirectional LSTM gap-filler trained on held-out
trajectory windows. Kalman filter/smoother classes are kept as optional
physics-based comparison methods.

The LSTM learns a residual correction on top of an interpolation baseline so
it only has to learn where simple interpolation is wrong.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# Try importing ML libraries; graceful fallback if not available
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


def great_circle_km(lat1, lon1, lat2, lon2):
    """Vectorized great-circle distance in kilometers."""
    radius_km = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    return radius_km * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


@dataclass
class TrajectoryMetrics:
    """Metrics for evaluating trajectory reconstruction quality."""
    mae_lat: float  # Mean Absolute Error in latitude
    mae_lon: float  # Mean Absolute Error in longitude
    mae_alt: float  # Mean Absolute Error in altitude
    rmse_position: float  # Root Mean Square Error in position (km)
    velocity_smoothness: float  # Measure of velocity continuity


class ConstantVelocityKalmanFilter:
    """
    Kalman Filter with constant-velocity motion model.
    
    Assumes aircraft maintain constant velocity between observations.
    Better than interpolation because it accounts for velocity dynamics.
    
    State vector: [lat, lon, alt, v_lat, v_lon, v_alt]
    where v_* are velocities (degrees/sec and m/sec)
    """
    
    def __init__(self, process_noise: float = 1e-6, measurement_noise: float = 1e-5):
        """
        Initialize Kalman filter.
        
        Args:
            process_noise: How much we expect velocity to change (Q matrix)
            measurement_noise: How much we trust position measurements (R matrix)
        """
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.state = None
        self.covariance = None
    
    def initialize_state(self, lat: float, lon: float, alt: float):
        """Initialize filter with first observation."""
        self.state = np.array([lat, lon, alt, 0, 0, 0], dtype=float)
        self.covariance = np.eye(6) * 0.1
    
    def predict(self, dt: float) -> np.ndarray:
        """
        Prediction step: extrapolate position based on velocity.
        
        Args:
            dt: Time step (seconds)
        
        Returns:
            Predicted state
        """
        # Motion model: position = position + velocity * dt
        F = np.eye(6)
        F[0, 3] = dt  # lat += v_lat * dt
        F[1, 4] = dt  # lon += v_lon * dt
        F[2, 5] = dt  # alt += v_alt * dt
        
        self.state = F @ self.state
        
        # Update covariance: P = F * P * F^T + Q
        Q = np.eye(6) * self.process_noise
        Q[3:6, 3:6] *= 100  # Higher uncertainty in velocity changes
        self.covariance = F @ self.covariance @ F.T + Q
        
        return self.state
    
    def update(self, measurement: np.ndarray):
        """
        Update step: correct state based on measurement.
        
        Args:
            measurement: [lat, lon, alt] observation
        """
        # Only measure position, not velocity
        H = np.zeros((3, 6))
        H[:3, :3] = np.eye(3)
        
        # Measurement residual
        y = measurement - H @ self.state
        
        # Kalman gain
        R = np.eye(3) * self.measurement_noise
        S = H @ self.covariance @ H.T + R
        K = self.covariance @ H.T @ np.linalg.inv(S)
        
        # Update state and covariance
        self.state = self.state + K @ y
        self.covariance = (np.eye(6) - K @ H) @ self.covariance
    
    def filter_trajectory(self, timestamps: np.ndarray, positions: np.ndarray,
                         altitudes: np.ndarray,
                         observed: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter a complete trajectory.

        Args:
            timestamps: Array of timestamps (seconds from start)
            positions: Nx2 array of [lat, lon] positions
            altitudes: Array of altitudes
            observed: Boolean mask — True for real observations, False for gap points
                      (gap points are predict-only, no measurement update).
                      If None, all points are treated as observed.

        Returns:
            Filtered (latitudes, longitudes, altitudes)
        """
        if observed is None:
            observed = np.ones(len(timestamps), dtype=bool)

        filtered_lats = []
        filtered_lons = []
        filtered_alts = []

        for i, (lat, lon, alt) in enumerate(zip(positions[:, 0], positions[:, 1], altitudes)):
            if i == 0:
                self.initialize_state(lat, lon, alt)
            else:
                dt = timestamps[i] - timestamps[i - 1]
                self.predict(dt)

            if observed[i]:
                self.update(np.array([lat, lon, alt]))

            filtered_lats.append(self.state[0])
            filtered_lons.append(self.state[1])
            filtered_alts.append(self.state[2])

        return np.array(filtered_lats), np.array(filtered_lons), np.array(filtered_alts)


class KalmanSmoother:
    """
    Rauch-Tuch-Striebel Smoother.
    
    Improvements over filter:
    • Bidirectional: uses past AND future observations
    • Better for gap reconstruction: can use context from both sides
    • More accurate: processes all data multiple times
    
    Process:
    1. Forward pass: filter (past → future)
    2. Backward pass: smooth (future → past)
    """
    
    def __init__(self, process_noise: float = 1e-6, measurement_noise: float = 1e-5):
        self.kf = ConstantVelocityKalmanFilter(process_noise, measurement_noise)
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
    
    def smooth_trajectory(self, timestamps: np.ndarray, positions: np.ndarray,
                         altitudes: np.ndarray,
                         observed: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply smoother to trajectory.

        Args:
            timestamps: Array of timestamps (seconds from start)
            positions: Nx2 array of [lat, lon] positions
            altitudes: Array of altitudes
            observed: Boolean mask — True for real observations, False for gap points
                      (gap points are predict-only, no measurement update).
                      If None, all points are treated as observed.

        Returns:
            Smoothed (latitudes, longitudes, altitudes)
        """
        n = len(timestamps)

        if observed is None:
            observed = np.ones(n, dtype=bool)

        # Forward pass: standard Kalman filter
        states_fwd = []
        covs_fwd = []

        for i, (lat, lon, alt) in enumerate(zip(positions[:, 0], positions[:, 1], altitudes)):
            if i == 0:
                self.kf.initialize_state(lat, lon, alt)
            else:
                dt = timestamps[i] - timestamps[i - 1]
                self.kf.predict(dt)

            if observed[i]:
                self.kf.update(np.array([lat, lon, alt]))

            states_fwd.append(self.kf.state.copy())
            covs_fwd.append(self.kf.covariance.copy())
        
        states_fwd = np.array(states_fwd)
        covs_fwd = np.array(covs_fwd)
        
        # Backward pass: smooth
        states_smooth = states_fwd.copy()
        
        for i in range(n - 2, -1, -1):
            dt = timestamps[i + 1] - timestamps[i]
            
            # Predicted next state
            F = np.eye(6)
            F[0, 3] = dt
            F[1, 4] = dt
            F[2, 5] = dt
            
            state_pred_next = F @ states_fwd[i]
            
            # Smoother gain
            Q = np.eye(6) * self.process_noise
            Q[3:6, 3:6] *= 100
            cov_pred_next = F @ covs_fwd[i] @ F.T + Q
            
            D = covs_fwd[i] @ F.T @ np.linalg.inv(cov_pred_next)
            
            # Smoothed state
            states_smooth[i] = states_fwd[i] + D @ (states_smooth[i + 1] - state_pred_next)
        
        return (states_smooth[:, 0], states_smooth[:, 1], states_smooth[:, 2])


POSITION_SCALE = 10.0
ALTITUDE_SCALE_METERS = 10000.0
TIME_SCALE_SECONDS = 3600.0


def create_lstm_model(sequence_length: int = 40, output_length: int = 10,
                      feature_dim: int = 4) -> 'keras.Model':
    """
    Create a bidirectional LSTM sequence model for trajectory gap filling.
    
    Architecture:
    • Input: Sequences of past positions/velocities
    • LSTM layers: Learn temporal patterns
    • Dense output: Predict future positions
    
    Args:
        sequence_length: Number of context timesteps
        output_length: Number of gap timesteps to reconstruct
        feature_dim: Number of features (lat, lon, alt, time_delta)
    
    Returns:
        Compiled Keras model
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow not available. Install with: pip install tensorflow")
    
    model = keras.Sequential([
        layers.Input(shape=(sequence_length, feature_dim)),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Dropout(0.2),
        layers.Bidirectional(layers.LSTM(32, return_sequences=False)),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(
            output_length * 3,
            kernel_initializer="zeros",
            bias_initializer="zeros",
        ),
        layers.Reshape((output_length, 3))
    ])
    
    model.compile(optimizer='adam', loss=keras.losses.Huber(delta=1.0), metrics=['mae'])
    return model


class LSTMTrajectoryModel:
    """LSTM-based trajectory reconstruction model."""

    def __init__(self, context_length: int = 20, gap_length: int = 5,
                 sequence_length: Optional[int] = None):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available for LSTM model")

        if sequence_length is not None:
            context_length = sequence_length

        self.context_length = int(context_length)
        self.gap_length = int(gap_length)
        self.sequence_length = self.context_length * 2
        self.feature_dim = 4
        self.model = create_lstm_model(
            sequence_length=self.sequence_length,
            output_length=self.gap_length,
            feature_dim=self.feature_dim,
        )
        self.fitted = False

    @staticmethod
    def _safe_time_delta(current: float, previous: float) -> float:
        delta = float(current) - float(previous)
        return delta if np.isfinite(delta) else 0.0

    def _build_context_features(
        self,
        before_positions: np.ndarray,
        before_altitudes: np.ndarray,
        before_times: np.ndarray,
        after_positions: np.ndarray,
        after_altitudes: np.ndarray,
        after_times: np.ndarray,
    ) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        before_positions = np.asarray(before_positions, dtype=np.float64)[-self.context_length:]
        before_altitudes = np.asarray(before_altitudes, dtype=np.float64)[-self.context_length:]
        before_times = np.asarray(before_times, dtype=np.float64)[-self.context_length:]
        after_positions = np.asarray(after_positions, dtype=np.float64)[:self.context_length]
        after_altitudes = np.asarray(after_altitudes, dtype=np.float64)[:self.context_length]
        after_times = np.asarray(after_times, dtype=np.float64)[:self.context_length]

        if len(before_positions) != self.context_length or len(after_positions) != self.context_length:
            raise ValueError(
                f"Expected {self.context_length} points before and after the gap"
            )

        anchor_lat = float(before_positions[-1, 0])
        anchor_lon = float(before_positions[-1, 1])
        anchor_alt = float(before_altitudes[-1])
        anchor_time = float(before_times[-1])

        features = []
        for i, (position, altitude, timestamp) in enumerate(
            zip(before_positions, before_altitudes, before_times)
        ):
            dt = 0.0 if i == 0 else self._safe_time_delta(timestamp, before_times[i - 1])
            features.append([
                (float(position[0]) - anchor_lat) * POSITION_SCALE,
                (float(position[1]) - anchor_lon) * POSITION_SCALE,
                (float(altitude) - anchor_alt) / ALTITUDE_SCALE_METERS,
                dt / TIME_SCALE_SECONDS,
            ])

        for i, (position, altitude, timestamp) in enumerate(
            zip(after_positions, after_altitudes, after_times)
        ):
            previous_time = anchor_time if i == 0 else after_times[i - 1]
            dt = self._safe_time_delta(timestamp, previous_time)
            features.append([
                (float(position[0]) - anchor_lat) * POSITION_SCALE,
                (float(position[1]) - anchor_lon) * POSITION_SCALE,
                (float(altitude) - anchor_alt) / ALTITUDE_SCALE_METERS,
                dt / TIME_SCALE_SECONDS,
            ])

        return np.asarray(features, dtype=np.float32), (anchor_lat, anchor_lon, anchor_alt)

    @staticmethod
    def _linear_gap_baseline(
        before_positions: np.ndarray,
        before_altitudes: np.ndarray,
        after_positions: np.ndarray,
        after_altitudes: np.ndarray,
        gap_length: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        p_before = np.asarray(before_positions, dtype=np.float64)[-1]
        p_after = np.asarray(after_positions, dtype=np.float64)[0]
        alt_before = float(np.asarray(before_altitudes, dtype=np.float64)[-1])
        alt_after = float(np.asarray(after_altitudes, dtype=np.float64)[0])

        fractions = np.linspace(
            1 / (gap_length + 1),
            gap_length / (gap_length + 1),
            gap_length,
            dtype=np.float64,
        )
        baseline_positions = p_before + (p_after - p_before) * fractions[:, np.newaxis]
        baseline_altitudes = alt_before + (alt_after - alt_before) * fractions
        return baseline_positions, baseline_altitudes

    @staticmethod
    def _build_residual_target(
        gap_positions: np.ndarray,
        gap_altitudes: np.ndarray,
        baseline_positions: np.ndarray,
        baseline_altitudes: np.ndarray,
    ) -> np.ndarray:
        gap_positions = np.asarray(gap_positions, dtype=np.float64)
        gap_altitudes = np.asarray(gap_altitudes, dtype=np.float64)
        baseline_positions = np.asarray(baseline_positions, dtype=np.float64)
        baseline_altitudes = np.asarray(baseline_altitudes, dtype=np.float64)

        return np.column_stack([
            (gap_positions[:, 0] - baseline_positions[:, 0]) * POSITION_SCALE,
            (gap_positions[:, 1] - baseline_positions[:, 1]) * POSITION_SCALE,
            (gap_altitudes - baseline_altitudes) / ALTITUDE_SCALE_METERS,
        ]).astype(np.float32)

    def prepare_training_data(self, trajectories: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                             output_length: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data for LSTM.

        Args:
            trajectories: List of (positions, altitudes, times) tuples
            output_length: Optional compatibility parameter; must match gap_length

        Returns:
            (X_train, y_train) ready for model training
        """
        if output_length is not None and output_length != self.gap_length:
            raise ValueError("output_length must match the model gap_length")

        X, y = [], []
        window_length = 2 * self.context_length + self.gap_length

        for positions, altitudes, times in trajectories:
            if len(positions) < window_length:
                continue

            for start in range(len(positions) - window_length + 1):
                before_slc = slice(start, start + self.context_length)
                gap_slc = slice(
                    start + self.context_length,
                    start + self.context_length + self.gap_length,
                )
                after_slc = slice(
                    start + self.context_length + self.gap_length,
                    start + window_length,
                )

                features, _ = self._build_context_features(
                    positions[before_slc], altitudes[before_slc], times[before_slc],
                    positions[after_slc], altitudes[after_slc], times[after_slc],
                )
                baseline_positions, baseline_altitudes = self._linear_gap_baseline(
                    positions[before_slc], altitudes[before_slc],
                    positions[after_slc], altitudes[after_slc],
                    self.gap_length,
                )
                target = self._build_residual_target(
                    positions[gap_slc],
                    altitudes[gap_slc],
                    baseline_positions,
                    baseline_altitudes,
                )

                X.append(features)
                y.append(target)

        if not X:
            return (
                np.empty((0, self.sequence_length, self.feature_dim), dtype=np.float32),
                np.empty((0, self.gap_length, 3), dtype=np.float32),
            )

        return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             epochs: int = 50, batch_size: int = 32, verbose: int = 0,
             validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
             callbacks: Optional[List['keras.callbacks.Callback']] = None,
             validation_split: float = 0.2):
        """Train the LSTM model."""
        fit_kwargs = {}
        if validation_data is not None:
            fit_kwargs["validation_data"] = validation_data
        else:
            fit_kwargs["validation_split"] = validation_split
        if callbacks is not None:
            fit_kwargs["callbacks"] = callbacks

        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            **fit_kwargs,
        )
        self.fitted = True
        return history

    def save(self, path: str):
        """Save the trained Keras model."""
        self.model.save(path)

    @classmethod
    def load(cls, path: str, context_length: Optional[int] = None,
             gap_length: Optional[int] = None) -> "LSTMTrajectoryModel":
        """Load a saved Keras LSTM gap-filler."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available for LSTM model")

        loaded_model = keras.models.load_model(path)
        input_steps = int(loaded_model.input_shape[1])
        output_steps = int(loaded_model.output_shape[1])

        if context_length is None:
            if input_steps % 2 != 0:
                raise ValueError("Saved model input length is not split into two contexts")
            context_length = input_steps // 2
        if gap_length is None:
            gap_length = output_steps

        instance = cls(context_length=context_length, gap_length=gap_length)
        if instance.sequence_length != input_steps or instance.gap_length != output_steps:
            raise ValueError(
                "Saved LSTM model shape does not match the requested context/gap lengths"
            )

        instance.model = loaded_model
        instance.fitted = True
        return instance

    def predict_gap(
        self,
        before_positions: np.ndarray,
        before_altitudes: np.ndarray,
        before_times: np.ndarray,
        after_positions: np.ndarray,
        after_altitudes: np.ndarray,
        after_times: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict trajectory to fill a gap.

        Args:
            before_positions: Positions before the gap
            before_altitudes: Altitudes before the gap
            before_times: Timestamps before the gap
            after_positions: Positions after the gap
            after_altitudes: Altitudes after the gap
            after_times: Timestamps after the gap

        Returns:
            (predicted_lats, predicted_lons, predicted_alts)
        """
        if not self.fitted:
            raise ValueError("Model not trained yet")

        features, _ = self._build_context_features(
            before_positions, before_altitudes, before_times,
            after_positions, after_altitudes, after_times,
        )
        predictions = self.model.predict(features[np.newaxis, :], verbose=0)[0]
        baseline_positions, baseline_altitudes = self._linear_gap_baseline(
            before_positions, before_altitudes,
            after_positions, after_altitudes,
            self.gap_length,
        )

        predicted_lats = baseline_positions[:, 0] + predictions[:, 0] / POSITION_SCALE
        predicted_lons = baseline_positions[:, 1] + predictions[:, 1] / POSITION_SCALE
        predicted_alts = baseline_altitudes + predictions[:, 2] * ALTITUDE_SCALE_METERS

        return predicted_lats, predicted_lons, predicted_alts


class FusionTrajectoryModel:
    """
    Main fusion model combining multiple approaches.
    
    Provides interface to:
    • Baseline: Great-circle interpolation
    • Kalman Filter: Physics-based filtering
    • Kalman Smoother: Bidirectional smoothing
    • LSTM: Machine learning approach
    """
    
    def __init__(self, use_lstm: bool = False,
                 lstm_weights_path: Optional[str] = None,
                 lstm_context_length: int = 20,
                 lstm_gap_length: int = 10):
        self.kf = ConstantVelocityKalmanFilter()
        self.smoother = KalmanSmoother()
        self.lstm_model = None
        
        if use_lstm:
            try:
                if lstm_weights_path:
                    self.lstm_model = LSTMTrajectoryModel.load(
                        lstm_weights_path,
                        context_length=lstm_context_length,
                        gap_length=lstm_gap_length,
                    )
                else:
                    self.lstm_model = LSTMTrajectoryModel(
                        context_length=lstm_context_length,
                        gap_length=lstm_gap_length,
                    )
            except ImportError:
                print("⚠ TensorFlow not available. LSTM model disabled.")
    
    def reconstruct_gap(self, before_lat: np.ndarray, before_lon: np.ndarray,
                       before_alt: np.ndarray, before_times: np.ndarray,
                       after_lat: np.ndarray, after_lon: np.ndarray,
                       after_alt: np.ndarray, after_times: np.ndarray,
                       gap_times: np.ndarray,
                       method: str = 'smoother') -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Reconstruct a gap in trajectory using specified method.
        
        Args:
            before_*: Trajectory data before gap
            after_*: Trajectory data after gap
            gap_times: Times to fill
            method: 'lstm', 'kalman', 'smoother', 'both', or 'all'. The
                smoother path also returns the forward Kalman filter for
                comparison.
        
        Returns:
            Dict with reconstruction from each available method
        """
        method = method.lower()
        if method not in ("lstm", "kalman", "smoother", "both", "all"):
            raise ValueError(
                "method must be one of: 'lstm', 'kalman', 'smoother', 'both', 'all'"
            )

        results = {}

        if method in ("lstm", "all"):
            if self.lstm_model is None or not self.lstm_model.fitted:
                raise ValueError("A trained LSTM model is required for method='lstm'")
            if len(gap_times) != self.lstm_model.gap_length:
                raise ValueError(
                    f"LSTM expects {self.lstm_model.gap_length} gap points, "
                    f"got {len(gap_times)}"
                )

            before_positions = np.column_stack([before_lat, before_lon])
            after_positions = np.column_stack([after_lat, after_lon])
            lstm_lats, lstm_lons, lstm_alts = self.lstm_model.predict_gap(
                before_positions, before_alt, before_times,
                after_positions, after_alt, after_times,
            )
            results["lstm"] = (lstm_lats, lstm_lons, lstm_alts)

            if method == "lstm":
                return results

        # Combine context from before and after, with placeholders for gap
        combined_times = np.concatenate([before_times, gap_times, after_times])
        combined_lats = np.concatenate([before_lat, np.zeros(len(gap_times)), after_lat])
        combined_lons = np.concatenate([before_lon, np.zeros(len(gap_times)), after_lon])
        combined_alts = np.concatenate([before_alt, np.zeros(len(gap_times)), after_alt])

        combined_positions = np.column_stack([combined_lats, combined_lons])

        # Build observation mask: True for real data, False for gap points
        observed = np.concatenate([
            np.ones(len(before_times), dtype=bool),
            np.zeros(len(gap_times), dtype=bool),
            np.ones(len(after_times), dtype=bool),
        ])

        # Normalize times to seconds from start
        combined_times = (combined_times - combined_times[0]).astype(float)

        gap_idx = len(before_times)
        gap_len = len(gap_times)

        # Kalman filter
        if method in ("kalman", "smoother", "both", "all"):
            try:
                filtered_lats, filtered_lons, filtered_alts = self.kf.filter_trajectory(
                    combined_times, combined_positions, combined_alts, observed
                )
                results['kalman'] = (
                    filtered_lats[gap_idx:gap_idx + gap_len],
                    filtered_lons[gap_idx:gap_idx + gap_len],
                    filtered_alts[gap_idx:gap_idx + gap_len],
                )
            except Exception as e:
                print(f"Kalman filter error: {e}")

        # Kalman smoother
        if method in ('smoother', 'both', 'all'):
            try:
                smoothed_lats, smoothed_lons, smoothed_alts = self.smoother.smooth_trajectory(
                    combined_times, combined_positions, combined_alts, observed
                )
                results['smoother'] = (
                    smoothed_lats[gap_idx:gap_idx + gap_len],
                    smoothed_lons[gap_idx:gap_idx + gap_len],
                    smoothed_alts[gap_idx:gap_idx + gap_len],
                )
            except Exception as e:
                print(f"Smoother error: {e}")

        return results
    
    def evaluate_reconstruction(self, true_positions: np.ndarray,
                              predicted_positions: np.ndarray,
                              true_alt: Optional[np.ndarray] = None,
                              predicted_alt: Optional[np.ndarray] = None) -> TrajectoryMetrics:
        """
        Evaluate reconstruction quality against ground truth.

        Args:
            true_positions: Nx2 array of [lat, lon] ground truth
            predicted_positions: Nx2 array of [lat, lon] predictions
            true_alt: Array of ground truth altitudes (optional)
            predicted_alt: Array of predicted altitudes (optional)
        """
        lat_error = np.mean(np.abs(true_positions[:, 0] - predicted_positions[:, 0]))
        lon_error = np.mean(np.abs(true_positions[:, 1] - predicted_positions[:, 1]))

        position_errors = great_circle_km(
            true_positions[:, 0],
            true_positions[:, 1],
            predicted_positions[:, 0],
            predicted_positions[:, 1],
        )
        dist_error = np.sqrt(np.mean(position_errors ** 2))

        # Altitude error
        alt_error = 0.0
        if true_alt is not None and predicted_alt is not None:
            alt_error = np.mean(np.abs(true_alt - predicted_alt))

        # Velocity smoothness (lower is smoother)
        velocities = np.diff(predicted_positions, axis=0)
        velocity_changes = np.diff(velocities, axis=0)
        smoothness = np.mean(np.linalg.norm(velocity_changes, axis=1)) if len(velocity_changes) > 0 else 0.0

        return TrajectoryMetrics(
            mae_lat=lat_error,
            mae_lon=lon_error,
            mae_alt=alt_error,
            rmse_position=dist_error,
            velocity_smoothness=smoothness
        )


if __name__ == "__main__":
    print("Improved Fusion Models for Aircraft Trajectory Reconstruction")
    print("=" * 70)
    
    # Demonstration with synthetic data
    print("\n[1] Creating synthetic trajectory with gap...")
    
    # Before gap
    t_before = np.linspace(0, 300, 30)  # 30 points over 5 minutes
    lat_before = 47.5 + 0.01 * np.sin(t_before / 100)
    lon_before = 8.5 + 0.01 * np.cos(t_before / 100)
    alt_before = 10000 + 100 * np.sin(t_before / 150)
    
    # After gap (simulating 2 minutes of flight)
    t_after = t_before + 120
    lat_after = lat_before[-1] + 0.02 + 0.01 * np.sin(t_after / 100)
    lon_after = lon_before[-1] + 0.02 + 0.01 * np.cos(t_after / 100)
    alt_after = alt_before[-1] + 200 + 100 * np.sin(t_after / 150)
    
    print(f"  Before gap: {len(lat_before)} points")
    print(f"  After gap: {len(lat_after)} points")
    print(f"  Gap duration: 120 seconds")
    
    # Test Kalman filter
    print("\n[2] Testing Constant-Velocity Kalman Filter...")
    kf = ConstantVelocityKalmanFilter(process_noise=1e-6, measurement_noise=1e-4)
    
    combined_times = np.concatenate([t_before, t_before[-1] + np.linspace(1, 120, 50)])
    combined_lats = np.concatenate([lat_before, np.zeros(50)])
    combined_lons = np.concatenate([lon_before, np.zeros(50)])
    combined_alts = np.concatenate([alt_before, np.zeros(50)])
    
    combined_positions = np.column_stack([combined_lats, combined_lons])
    
    filt_lats, filt_lons, filt_alts = kf.filter_trajectory(combined_times, combined_positions, combined_alts)
    print(f"  Filtered trajectory: {len(filt_lats)} points")
    
    # Test Kalman smoother
    print("\n[3] Testing Kalman Smoother (RTS)...")
    smoother = KalmanSmoother(process_noise=1e-6, measurement_noise=1e-4)
    smooth_lats, smooth_lons, smooth_alts = smoother.smooth_trajectory(
        combined_times, combined_positions, combined_alts
    )
    print(f"  Smoothed trajectory: {len(smooth_lats)} points")
    
    print("\n[4] Project model:")
    print("  ✓ Constant-Velocity Kalman Filter: Lightweight, physics-based")
    print("  ✓ Kalman Smoother: Better for gaps, bidirectional")
    print("  ✓ LSTM Model: Learns complex patterns (requires training data)")
    
    print("\n[5] Recommended project flow:")
    print("  • Start with Kalman Smoother for trajectory gaps")
    print("  • Use when: gaps are small (<5 min), need fast inference")
    print("  • Train LSTM on complete trajectories for better long-term predictions")
    print("  • Compare against baseline to quantify improvement")
