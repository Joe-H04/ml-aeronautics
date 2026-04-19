"""
Baseline: Great-Circle Interpolation for Aircraft Trajectories

A simple baseline that fills gaps in aircraft trajectories by interpolating
along the great-circle arc (shortest path on Earth's surface) between known points.

This provides a reference point for evaluating more complex trajectory prediction models.
"""

import numpy as np
import pandas as pd
from math import radians, degrees, sin, cos, atan2, sqrt


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on Earth (in kilometers).
    
    Args:
        lat1, lon1: Latitude and longitude of first point (degrees)
        lat2, lon2: Latitude and longitude of second point (degrees)
    
    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth's radius in kilometers
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    return R * c


def interpolate_great_circle(lat1: float, lon1: float, lat2: float, lon2: float, 
                             fraction: float) -> tuple[float, float]:
    """
    Interpolate a point along the great-circle arc between two points.
    
    Args:
        lat1, lon1: Starting point (degrees)
        lat2, lon2: Ending point (degrees)
        fraction: Position along arc, 0 ≤ fraction ≤ 1
                  (0 = start, 1 = end, 0.5 = midpoint)
    
    Returns:
        Tuple of (latitude, longitude) at the interpolated position
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Great-circle distance
    dlon = lon2 - lon1
    a = sin((lat2 - lat1) / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    delta = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    # If points are essentially the same, return starting point
    if delta < 1e-10:
        return degrees(lat1), degrees(lon1)
    
    # Interpolation using spherical trigonometry
    A = sin((1 - fraction) * delta) / sin(delta)
    B = sin(fraction * delta) / sin(delta)
    
    x = A * cos(lat1) * cos(lon1) + B * cos(lat2) * cos(lon2)
    y = A * cos(lat1) * sin(lon1) + B * cos(lat2) * sin(lon2)
    z = A * sin(lat1) + B * sin(lat2)
    
    lat = atan2(z, sqrt(x**2 + y**2))
    lon = atan2(y, x)
    
    return degrees(lat), degrees(lon)


def interpolate_altitude(alt1: float, alt2: float, fraction: float) -> float:
    """
    Linear interpolation for altitude.
    
    Args:
        alt1: Starting altitude (meters)
        alt2: Ending altitude (meters)
        fraction: Position, 0 ≤ fraction ≤ 1
    
    Returns:
        Interpolated altitude (meters)
    """
    return alt1 + (alt2 - alt1) * fraction


def fill_trajectory_gaps(df: pd.DataFrame, max_gap_seconds: int = 300,
                         min_gap_seconds: int = 10,
                         points_per_interval: int = 10) -> pd.DataFrame:
    """
    Fill gaps in a flight trajectory using great-circle interpolation.

    Args:
        df: DataFrame with columns: time, latitude, longitude, baro_altitude
            (must be sorted by time)
        max_gap_seconds: Maximum gap to interpolate (gaps larger are left as-is)
        min_gap_seconds: Minimum gap to interpolate (smaller gaps are left as-is)
        points_per_interval: Number of interpolated points to add per interval

    Returns:
        DataFrame with interpolated points inserted
    """
    if df.empty or len(df) < 2:
        return df

    # Convert Unix timestamps to datetime if needed
    if df['time'].dtype == 'int64':
        df = df.copy()
        df['time'] = pd.to_datetime(df['time'], unit='s')

    result = []
    result.append(df.iloc[0])

    for i in range(len(df) - 1):
        current = df.iloc[i]
        next_point = df.iloc[i + 1]

        # Calculate time gap
        time_diff = (next_point['time'] - current['time']).total_seconds()

        # Only interpolate gaps within [min_gap_seconds, max_gap_seconds]
        if time_diff <= 0 or time_diff < min_gap_seconds or time_diff > max_gap_seconds:
            result.append(next_point)
            continue
        
        # Extract coordinates
        lat1, lon1 = current['latitude'], current['longitude']
        lat2, lon2 = next_point['latitude'], next_point['longitude']
        alt1, alt2 = current['baro_altitude'], next_point['baro_altitude']
        time1, time2 = current['time'], next_point['time']
        
        # Interpolate intermediate points
        for j in range(1, points_per_interval):
            fraction = j / points_per_interval
            
            # Interpolate position along great circle
            interp_lat, interp_lon = interpolate_great_circle(lat1, lon1, lat2, lon2, fraction)
            interp_alt = interpolate_altitude(alt1, alt2, fraction)
            interp_time = time1 + (time2 - time1) * fraction
            
            # Create interpolated row
            interp_row = current.copy()
            interp_row['latitude'] = interp_lat
            interp_row['longitude'] = interp_lon
            interp_row['baro_altitude'] = interp_alt
            interp_row['time'] = interp_time
            interp_row['interpolated'] = True  # Mark as interpolated
            
            result.append(interp_row)
        
        # Add the actual next point
        result.append(next_point)
    
    return pd.DataFrame(result).reset_index(drop=True)


if __name__ == "__main__":
    # Example usage
    print("Baseline: Great-Circle Interpolation for Flight Trajectories")
    print("=" * 60)
    
    # Create sample trajectory with gaps
    sample_data = {
        'time': pd.to_datetime([
            '2024-01-01 10:00:00',
            '2024-01-01 10:05:00',  # 5 min gap
            '2024-01-01 10:10:00',  # 5 min gap
        ]),
        'latitude': [47.5, 48.0, 48.5],
        'longitude': [8.5, 9.0, 9.5],
        'baro_altitude': [10000, 11000, 12000],
    }
    
    df = pd.DataFrame(sample_data)
    
    print("\nOriginal trajectory (3 points, 5-minute gaps):")
    print(df)
    
    # Fill gaps
    df_filled = fill_trajectory_gaps(df, max_gap_seconds=300, points_per_interval=5)
    
    print(f"\nInterpolated trajectory ({len(df_filled)} points total):")
    print(df_filled)
    
    # Show distances
    print("\nGreat-circle distances between consecutive points:")
    for i in range(len(df_filled) - 1):
        curr = df_filled.iloc[i]
        next_p = df_filled.iloc[i + 1]
        dist = haversine_distance(curr['latitude'], curr['longitude'],
                                  next_p['latitude'], next_p['longitude'])
        print(f"  Point {i} → {i+1}: {dist:.2f} km")
