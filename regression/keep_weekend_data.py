from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import ccxt
import ccxt_data_fetcher


def load_df_from_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)

    # 你的資料常見是 index= datetime
    if isinstance(df.index, pd.DatetimeIndex):
        dt = df.index
    elif "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"])
        df = df.set_index(dt).drop(columns=["datetime"])
    else:
        raise ValueError("找不到 datetime：請確認 parquet 是 DatetimeIndex 或有 'datetime' 欄位")

    # 確保 index 是 DatetimeIndex
    df.index = pd.DatetimeIndex(df.index)
    df.index.name = "datetime"
    return df


def ensure_ny_naive_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    把 index 統一成「America/New_York 時區後再去 timezone (naive)」，
    這樣週五/週日判斷是以 NY 的星期與時間為準。
    """
    idx = df.index

    if idx.tz is None:
        # 這裡假設「naive 就是 NY 時間」——符合你前面下載程式的輸出格式
        ny_naive = idx
    else:
        ny_naive = idx.tz_convert("America/New_York").tz_localize(None)

    df = df.copy()
    df.index = pd.DatetimeIndex(ny_naive)
    df.index.name = "datetime"
    return df


def weekend_window_mask_ny(df: pd.DataFrame) -> pd.Series:
    """
    保留 NY 時間：
      - 週五 17:00 ~ 24:00
      - 週六 00:00 ~ 24:00
      - 週日 00:00 ~ 18:00
    週幾：Mon=0 ... Sun=6
    """
    idx = df.index
    dow = idx.dayofweek
    minutes = idx.hour * 60 + idx.minute

    fri = (dow == 4) & (minutes >= 17 * 60 - 20)          # Fri >= 17:00
    sat = (dow == 5)                                  # Sat all day
    sun = (dow == 6) & (minutes < 18 * 60 + 20)           # Sun < 18:00 (exclusive)

    return fri | sat | sun


def filter_weekend_parquet(in_path: str, out_path: str):
    df = load_df_from_parquet(in_path)
    df = ensure_ny_naive_index(df)

    mask = weekend_window_mask_ny(df)
    out = df.loc[mask].sort_index()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, engine="pyarrow")

    print(f"input rows : {len(df)}")
    print(f"output rows: {len(out)}")


from pathlib import Path
import ccxt

# 假設 filter_weekend_parquet 支援傳 Path 或 str
# def filter_weekend_parquet(in_path, out_path): ...


if __name__ == "__main__":
    data_dir = Path("data")

    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })

    df, csv_path, pq_path = ccxt_data_fetcher.download_ohlcv_binance_futures(
        symbol="PAXG/USDT",
        timeframe="1m",
        since="2025-01-01T00:00:00Z",
        until="2026-01-06T00:00:00Z",
        exchange=exchange,
        save_dir=data_dir,
        mark="",
        save_csv=False,
        save_parquet=True,
    )

    in_path = Path(pq_path)
    out_path = in_path.with_name(in_path.stem + "_weekend" + in_path.suffix)

    filter_weekend_parquet(
        in_path=in_path,
        out_path=out_path,
    )

    print("Input :", in_path)
    print("saved ->", out_path)

