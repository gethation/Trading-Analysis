from pathlib import Path
import numpy as np
import pandas as pd
import mplfinance as mpf


def load_ohlcv_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)

    if isinstance(df.index, pd.DatetimeIndex):
        pass
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
    else:
        raise ValueError("parquet 需要 DatetimeIndex 或包含 'datetime' 欄位")

    df.index = pd.DatetimeIndex(df.index)
    df = df.sort_index()

    rename_map = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    miss = set(rename_map.keys()) - set(df.columns)
    if miss:
        raise ValueError(f"缺少欄位: {miss}")

    return df.rename(columns=rename_map)


def add_padded_and_blended_ma_weekend(
    df: pd.DataFrame,
    window: int = 500,
    alpha: float = 0.7,
    price_col: str = "Close",
    pad_col: str = "Open",
    session_start_hour: int = 17,
):
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha 必須在 [0, 1]")

    df = df.copy()
    idx = pd.DatetimeIndex(df.index)

    dow = idx.dayofweek
    minutes = idx.hour * 60 + idx.minute

    in_weekend = (
        ((dow == 4) & (minutes >= 17 * 60)) |
        (dow == 5) |
        ((dow == 6) & (minutes < 18 * 60))
    )

    delta_days = (dow - 4).astype("int64")
    delta_days = np.where(delta_days < 0, 0, delta_days)
    session_start = (
        idx.normalize() + pd.Timedelta(hours=session_start_hour)
        - pd.to_timedelta(delta_days, unit="D")
    )
    df["_session_start"] = session_start

    out_ma = pd.Series(index=df.index, dtype="float64")
    out_map = pd.Series(index=df.index, dtype="float64")

    wk = df.loc[in_weekend].copy()
    if wk.empty:
        df[f"MA{window}_padded"] = out_ma
        df[f"MA{window}_blend_a{alpha:g}"] = out_map
        return df.drop(columns=["_session_start"])

    # --- padded MA：不足 window 用 session 第一根 pad_col 補齊 ---
    pad_price = wk.groupby("_session_start")[pad_col].transform("first")
    n = wk.groupby("_session_start").cumcount() + 1
    n_clip = np.minimum(n.to_numpy(), window)

    roll_sum = (
        wk.groupby("_session_start")[price_col]
          .rolling(window=window, min_periods=1)
          .sum()
          .reset_index(level=0, drop=True)
    )

    ma_padded = (roll_sum.to_numpy() + (window - n_clip) * pad_price.to_numpy()) / window
    wk[f"MA{window}_padded"] = ma_padded

    # --- 取每個 session 的「Fri 17:00 那根 Open」當基準（缺就用 session 第一根 Open）---
    def _ref_open_transform(s: pd.Series) -> pd.Series:
        gi = pd.DatetimeIndex(s.index)
        mask = (gi.dayofweek == 4) & (gi.hour == session_start_hour) & (gi.minute == 0)
        ref = s.loc[mask].iloc[0] if mask.any() else s.iloc[0]
        return pd.Series(ref, index=s.index)

    ref_open = wk.groupby("_session_start")["Open"].transform(_ref_open_transform)

    # --- 你的公式：MA' = alpha*MA + (1-alpha)*ref_open ---
    ma_blend = alpha * wk[f"MA{window}_padded"].to_numpy() + (1 - alpha) * ref_open.to_numpy()
    wk[f"MA{window}_blend_a{alpha:g}"] = ma_blend

    out_ma.loc[wk.index] = wk[f"MA{window}_padded"]
    out_map.loc[wk.index] = wk[f"MA{window}_blend_a{alpha:g}"]

    df[f"MA{window}_padded"] = out_ma
    df[f"MA{window}_blend_a{alpha:g}"] = out_map

    return df.drop(columns=["_session_start"])



def plot_ohlcv_with_mas(df: pd.DataFrame, ma_cols: list[str], title: str):
    apds = [mpf.make_addplot(df[c], panel=0) for c in ma_cols]

    mpf.plot(
        df,
        type="candle",
        volume=True,
        addplot=apds,
        title=title,
        style="yahoo",
        figsize=(14, 8),
        tight_layout=True,
    )


if __name__ == "__main__":
    path = r"data/PAXG_1m_weekend.parquet"
    df = load_ohlcv_parquet(path)

    window = 100
    alpha = 0.5  # 你可調：越小越貼近 17:00 close

    df = add_padded_and_blended_ma_weekend(df, window=window, alpha=alpha)

    plot_ohlcv_with_mas(
        df,
        ma_cols=[f"MA{window}_padded", f"MA{window}_blend_a{alpha:g}"],
        title=f"OHLCV + MA(padded) + MA'(alpha blend) - {Path(path).stem}",
    )
