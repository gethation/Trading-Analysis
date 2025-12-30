from pathlib import Path
import pandas as pd
import mplfinance as mpf


def load_ohlcv_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)

    # 支援兩種格式：index 是 datetime 或有 datetime 欄位
    if isinstance(df.index, pd.DatetimeIndex):
        pass
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
    else:
        raise ValueError("parquet 需要 DatetimeIndex 或包含 'datetime' 欄位")

    df.index = pd.DatetimeIndex(df.index)
    df = df.sort_index()

    # mplfinance 欄位名要是 Open/High/Low/Close/Volume（大小寫敏感）
    rename_map = {
        "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"
    }
    miss = set(rename_map.keys()) - set(df.columns)
    if miss:
        raise ValueError(f"缺少欄位: {miss}")

    df = df.rename(columns=rename_map)
    return df


def plot_ohlcv(df: pd.DataFrame, title: str = "OHLCV"):
    # 你也可以改成 type='candle' 或 'ohlc'
    mpf.plot(
        df,
        type="candle",
        volume=True,
        title=title,
        style="yahoo",      # 想要更樸素可改 "classic"
        figsize=(14, 8),
        tight_layout=True,
    )


if __name__ == "__main__":
    path = r"data/PAXG_15m_weekend.parquet"
    df = load_ohlcv_parquet(path)
    plot_ohlcv(df, title=f"OHLCV - {Path(path).stem}")
