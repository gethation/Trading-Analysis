from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute mean of close column in spread ratio parquet.")
    ap.add_argument(
        "--path",
        type=Path,
        default=Path("data/spread_ratio_PAXG_XAUT_1m.parquet"),
        help="Path to parquet file (default: data/spread_ratio_PAXG_XAUT_1m.parquet)",
    )
    ap.add_argument(
        "--col",
        type=str,
        default="",
        help="Column name to compute mean. If empty, auto-pick ratio_close then close.",
    )
    args = ap.parse_args()

    path: Path = args.path
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_parquet(path)
    if df.empty:
        print("DataFrame is empty. Mean = NaN")
        return

    # Ensure datetime index if possible (not required for mean, but useful info)
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        pass

    if args.col:
        col = args.col
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")
    else:
        if "ratio_close" in df.columns:
            col = "ratio_close"
        elif "close" in df.columns:
            col = "close"
        else:
            raise ValueError(
                "Cannot find 'ratio_close' or 'close' column. "
                f"Available: {list(df.columns)}"
            )

    s = pd.to_numeric(df[col], errors="coerce").dropna()
    mean_val = float(s.mean()) if len(s) else float("nan")

    # Extra context
    n_total = len(df)
    n_valid = len(s)
    idx_info = ""
    if isinstance(df.index, pd.DatetimeIndex) and df.index.notna().any():
        idx_info = f", range: {df.index.min()} ~ {df.index.max()}"

    print(f"File: {path}")
    print(f"Column: {col}")
    print(f"Rows: total={n_total}, valid={n_valid}{idx_info}")
    print(f"Mean({col}) = {mean_val:.10f}")


if __name__ == "__main__":
    main()