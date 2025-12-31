from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------
# 讀檔 + 欄位統一
# ------------------------
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
    # 若已是大寫就不動
    lower_cols = {c.lower(): c for c in df.columns}
    if "open" in lower_cols and "Open" not in df.columns:
        df = df.rename(columns={lower_cols["open"]: "Open"})
    if "high" in lower_cols and "High" not in df.columns:
        df = df.rename(columns={lower_cols["high"]: "High"})
    if "low" in lower_cols and "Low" not in df.columns:
        df = df.rename(columns={lower_cols["low"]: "Low"})
    if "close" in lower_cols and "Close" not in df.columns:
        df = df.rename(columns={lower_cols["close"]: "Close"})
    if "volume" in lower_cols and "Volume" not in df.columns:
        df = df.rename(columns={lower_cols["volume"]: "Volume"})

    need = {"Open", "High", "Low", "Close", "Volume"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"缺少欄位: {miss}")

    return df


# ------------------------
# 週末 session 定義
# ------------------------

def weekend_entry_mask_ny(idx: pd.DatetimeIndex) -> np.ndarray:
    """Fri>=17:00, Sat 全日, Sun<18:00 (NY)"""
    dow = idx.dayofweek            # 可能是 ndarray 或 Index
    minutes = idx.hour * 60 + idx.minute

    mask = (
        ((dow == 4) & (minutes >= 17 * 60)) |
        (dow == 5) |
        ((dow == 6) & (minutes < 18 * 60))
    )
    return np.asarray(mask, dtype=bool)



def session_start_fri17(idx: pd.DatetimeIndex, start_hour: int = 17) -> pd.Series:
    dow = np.asarray(idx.dayofweek)
    delta_days = dow - 4
    delta_days = np.where(delta_days < 0, 0, delta_days)
    ss = idx.normalize() + pd.Timedelta(hours=start_hour) - pd.to_timedelta(delta_days, unit="D")
    return pd.Series(ss, index=idx)



def bin_floor(values: np.ndarray, bin_size: float) -> np.ndarray:
    k = np.floor(values / bin_size + 1e-12).astype(int)
    return np.round(k * bin_size, 6)


# ------------------------
# 計算 padded MA & alpha_ma
# ------------------------
def add_weekend_alpha_ma(
    df: pd.DataFrame,
    window: int = 500,
    alpha: float = 0.7,
    price_col: str = "Close",
    pad_col: str = "Open",
    start_hour: int = 17,
):
    """
    只在週末區間計算：
      MA_padded(window): 不足 window 用 session 第一根 Open 補齊
      alpha_ma: alpha*MA_padded + (1-alpha)*Fri17:00Open  (缺就用 session 第一根 Open)
    其他時間欄位為 NaN
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha 必須在 [0, 1]")

    df = df.copy()
    idx = pd.DatetimeIndex(df.index)
    in_weekend = weekend_entry_mask_ny(idx)

    df["_session_start"] = session_start_fri17(idx, start_hour=start_hour).values

    out_ma = pd.Series(index=df.index, dtype="float64")
    out_alpha_ma = pd.Series(index=df.index, dtype="float64")

    wk = df.loc[in_weekend].copy()
    if wk.empty:
        df["MA_padded"] = out_ma
        df["alpha_ma"] = out_alpha_ma
        return df.drop(columns=["_session_start"])

    # padded MA：不足用 session 第一根 Open 補
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
    wk["MA_padded"] = ma_padded

    # 參考價：Fri 17:00 那根 Open（缺就用 session 第一根 Open）
    def ref_open_transform(s: pd.Series) -> pd.Series:
        gi = pd.DatetimeIndex(s.index)
        mask = (gi.dayofweek == 4) & (gi.hour == start_hour) & (gi.minute == 0)
        ref = s.loc[mask].iloc[0] if mask.any() else s.iloc[0]
        return pd.Series(ref, index=s.index)

    ref_open = wk.groupby("_session_start")["Open"].transform(ref_open_transform)

    wk["alpha_ma"] = alpha * wk["MA_padded"].to_numpy() + (1 - alpha) * ref_open.to_numpy()

    out_ma.loc[wk.index] = wk["MA_padded"]
    out_alpha_ma.loc[wk.index] = wk["alpha_ma"]

    df["MA_padded"] = out_ma
    df["alpha_ma"] = out_alpha_ma
    return df.drop(columns=["_session_start"])


# ------------------------
# 統計：target 隨時間更新、第一次碰到、18:03 截止
# ------------------------
def compute_stats(
    df: pd.DataFrame,
    bin_size_pct: float = 0.02,
    cutoff_h: int = 18,
    cutoff_m: int = 3,
):
    df = df.sort_index().copy()
    idx = pd.DatetimeIndex(df.index)

    need = {"Open", "High", "Low", "Close", "alpha_ma"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"缺少欄位: {miss}")

    entry_mask = weekend_entry_mask_ny(idx)
    df["_session_start"] = session_start_fri17(idx, start_hour=17).values

    trades = []

    # 只處理有 entry 的 session
    for ss, g in df.loc[entry_mask].groupby("_session_start"):
        cutoff_ts = pd.Timestamp(ss) + pd.Timedelta(days=2, hours=cutoff_h, minutes=cutoff_m)  # Sun 18:03

        full = df[(df["_session_start"] == ss) & (df.index >= ss) & (df.index <= cutoff_ts)].copy()
        if full.empty:
            continue

        times = pd.DatetimeIndex(full.index)

        # cutoff bar：第一根 >= 18:03，沒有就最後一根
        cutoff_pos = int(times.searchsorted(cutoff_ts, side="left"))
        if cutoff_pos >= len(times):
            cutoff_pos = len(times) - 1
        cutoff_time = times[cutoff_pos]
        cutoff_close = float(full["Close"].iloc[cutoff_pos])

        lo = full["Low"].to_numpy(float)
        hi = full["High"].to_numpy(float)
        ma = full["alpha_ma"].to_numpy(float)

        touch = (lo <= ma) & (ma <= hi)

        m = len(full)
        next_touch_after = np.full(m + 1, -1, dtype=int)
        next_pos = -1
        for i in range(m - 1, -1, -1):
            if touch[i]:
                next_pos = i
            next_touch_after[i] = next_pos
        next_touch_after[m] = -1

        # entry positions in full
        entry_times = g.index
        entry_pos = times.searchsorted(entry_times, side="left")

        for p in entry_pos:
            if p < 0 or p >= m:
                continue

            entry_time = times[p]
            entry_close = float(full["Close"].iloc[p])
            entry_ma = float(full["alpha_ma"].iloc[p])

            entry_dev = abs(entry_close - entry_ma) / entry_close * 100.0
            dev_bin = float(bin_floor(np.array([entry_dev]), bin_size_pct)[0])

            tpos = next_touch_after[p + 1] if (p + 1) <= m else -1

            if tpos != -1 and tpos <= cutoff_pos:
                exit_time = times[tpos]
                exit_level = float(full["alpha_ma"].iloc[tpos])  # target 隨時間更新，所以用觸碰當下 alpha_ma
                hit = 1
                wait_bars = int(tpos - p)
                ret_pct = abs(entry_close - exit_level) / entry_close * 100.0
            else:
                exit_time = cutoff_time
                exit_level = cutoff_close  # 未命中：用 cutoff close 當 alpha_ma
                hit = 0
                wait_bars = int(cutoff_pos - p)
                ret_pct = abs(entry_close - exit_level) / entry_close * 100.0

            trades.append({
                "session_start": ss,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "hit": hit,
                "wait_bars": wait_bars,
                "entry_dev_pct": entry_dev,
                "dev_bin_pct": dev_bin,
                "ret_pct": ret_pct,
            })

    trades_df = pd.DataFrame(trades).sort_values("entry_time")
    if trades_df.empty:
        return trades_df, pd.DataFrame()

    summary = (trades_df.groupby("dev_bin_pct")
               .agg(
                   n=("ret_pct", "size"),
                   avg_ret_pct=("ret_pct", "mean"),
                   med_ret_pct=("ret_pct", "median"),
                   hit_rate=("hit", "mean"),
                   avg_wait_bars=("wait_bars", "mean"),
               )
               .sort_index())

    return trades_df, summary


def plot_avg_return_bar(summary):

    if summary is None or summary.empty:
        print("summary is empty")
        return

    x = summary.index.to_numpy(dtype=float)
    y = summary["avg_ret_pct"].to_numpy(dtype=float)
    n = summary["n"].to_numpy(dtype=int)

    step = float(np.min(np.diff(x))) if len(x) >= 2 else 0.02
    width = step * 0.9

    plt.figure()
    bars = plt.bar(x, y, width=width, align="center")
    plt.xlabel("entry deviation bin (%)  (floor, step=0.02)")
    plt.ylabel("average realized return (%)")
    plt.title("Average return by entry deviation bin")
    plt.grid(True, axis="y")

    # 在柱頂標註 n（只標註前幾個或全部都可）
    for rect, nn in zip(bars, n):
        h = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2, h, str(nn),
                 ha="center", va="bottom", fontsize=8)

    plt.show()



# ------------------------
# 主程式
# ------------------------
if __name__ == "__main__":
    path = r"data/PAXG_1m_weekend.parquet"  # 改成你的檔案


    window = 500
    alpha = 0.25
    bin_size_pct = 0.02

    df = load_ohlcv_parquet(path)
    df = add_weekend_alpha_ma(df, window=window, alpha=alpha)
    trades, summary = compute_stats(df, bin_size_pct=bin_size_pct)

    print(summary.head(30))
    plot_avg_return_bar(summary)
