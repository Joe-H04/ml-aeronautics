"""Preview a sample of a parquet file.

Usage:
    python preview_parquet.py <path>               # first 10 rows
    python preview_parquet.py <path> --n 20        # first 20 rows
    python preview_parquet.py <path> --random 15   # 15 random rows
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Preview a parquet file.")
    parser.add_argument("path", type=Path, help="Path to the .parquet file")
    parser.add_argument("--n", type=int, default=10, help="Number of rows to show (default 10)")
    parser.add_argument("--random", type=int, default=0, help="Show N random rows instead of head")
    args = parser.parse_args()

    if not args.path.exists():
        raise SystemExit(f"File not found: {args.path}")

    df = pd.read_parquet(args.path)

    print(f"\nFile:     {args.path}")
    print(f"Shape:    {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Size:     {args.path.stat().st_size / 1024:.1f} KB on disk\n")

    print("Columns and dtypes:")
    for col, dtype in df.dtypes.items():
        print(f"  {col:<20} {dtype}")

    print(f"\nSample ({args.random or args.n} rows):")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        sample = df.sample(args.random, random_state=0) if args.random else df.head(args.n)
        print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
