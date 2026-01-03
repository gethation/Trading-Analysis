from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# 讀檔 + 去重 + 欄位統一
# ----------------------------
def load_ohlcv_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)

    if not isinstance(df.index, pd.DatetimeIndex):
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime")
        else:
            raise ValueError("parquet 需要 DatetimeIndex 或包含 'datetime' 欄位")

    df.index = pd.DatetimeIndex(df.index)
    df = df.sort_index()

    if df.index.has_duplicates:
        dup_n = int(df.index.duplicated(keep="last").sum())
        print(f"[WARN] index has duplicates: {dup_n} rows -> keep last")
        df = df[~df.index.duplicated(keep="last")].copy()

    # 欄位統一 Open/High/Low/Close/Volume（支援原本小寫）
    lower_cols = {c.lower(): c for c in df.columns}
    mapping = {}
    if "open" in lower_cols:   mapping[lower_cols["open"]] = "Open"
    if "high" in lower_cols:   mapping[lower_cols["high"]] = "High"
    if "low" in lower_cols:    mapping[lower_cols["low"]] = "Low"
    if "close" in lower_cols:  mapping[lower_cols["close"]] = "Close"
    if "volume" in lower_cols: mapping[lower_cols["volume"]] = "Volume"
    df = df.rename(columns=mapping)

    need = {"Open", "High", "Low", "Close", "Volume"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"缺少欄位: {miss}")

    return df[["Open", "High", "Low", "Close", "Volume"]].copy()


# ----------------------------
# 週末篩選 & session 定義（NY 時間）
# ----------------------------
def mask_entry_weekend(idx: pd.DatetimeIndex) -> np.ndarray:
    """Entry: Fri>=17:00, Sat 全日, Sun<18:00"""
    dow = np.asarray(idx.dayofweek)
    minutes = np.asarray(idx.hour) * 60 + np.asarray(idx.minute)
    return np.asarray(
        ((dow == 4) & (minutes >= 17 * 60)) |
        (dow == 5) |
        ((dow == 6) & (minutes < 18 * 60)),
        dtype=bool
    )

def mask_calc_weekend(idx: pd.DatetimeIndex, cutoff_m: int = 3) -> np.ndarray:
    """追蹤/計算範圍：Sun<18:00+cutoff_m，確保涵蓋到截止"""
    dow = np.asarray(idx.dayofweek)
    minutes = np.asarray(idx.hour) * 60 + np.asarray(idx.minute)
    cutoff_min = 18 * 60 + cutoff_m
    return np.asarray(
        ((dow == 4) & (minutes >= 17 * 60)) |
        (dow == 5) |
        ((dow == 6) & (minutes < cutoff_min)),
        dtype=bool
    )

def session_start_fri17(idx: pd.DatetimeIndex, start_hour: int = 17) -> np.ndarray:
    """每筆映射到該週末 session 的 Fri 17:00"""
    dow = np.asarray(idx.dayofweek)
    delta_days = dow - 4
    delta_days = np.where(delta_days < 0, 0, delta_days)
    ss = idx.normalize() + pd.Timedelta(hours=start_hour) - pd.to_timedelta(delta_days, unit="D")
    return np.asarray(ss)

def bin_floor(values: np.ndarray, bin_size: float) -> np.ndarray:
    k = np.floor(values / bin_size + 1e-12).astype(int)
    return np.round(k * bin_size, 6)


# ----------------------------
# alpha_ma（padded MA + blend 到 Fri17 Open）
# ----------------------------
def add_alpha_ma(df: pd.DataFrame, window: int = 500, alpha: float = 0.7, cutoff_m: int = 3) -> pd.DataFrame:
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha 必須在 [0,1]")

    df = df.copy()
    idx = pd.DatetimeIndex(df.index)

    calc_mask = mask_calc_weekend(idx, cutoff_m=cutoff_m)
    if not calc_mask.any():
        df["alpha_ma"] = np.nan
        return df

    sub = df.loc[calc_mask, ["Open", "Close"]].copy()
    sidx = pd.DatetimeIndex(sub.index)
    sub["_ss"] = session_start_fri17(sidx, start_hour=17)

    out = np.full(len(sub), np.nan, dtype=float)

    for ss, grp in sub.groupby("_ss", sort=True):
        locs = sub.index.get_indexer(grp.index)  # ✅ 保證對齊位置

        close = grp["Close"].to_numpy(float)
        open_ = grp["Open"].to_numpy(float)

        # ref_open：Fri 17:00 那根 Open（缺就用第一根）
        gidx = pd.DatetimeIndex(grp.index)
        is_fri17 = (gidx.dayofweek == 4) & (gidx.hour == 17) & (gidx.minute == 0)
        ref_open = float(open_[np.argmax(is_fri17)]) if is_fri17.any() else float(open_[0])

        # padded MA：rolling sum + padding（用 cumsum 加速）
        csum = np.cumsum(close)
        n = np.arange(1, len(close) + 1, dtype=int)
        roll_sum = np.empty_like(close)

        if len(close) <= window:
            roll_sum[:] = csum
            eff_n = n
        else:
            roll_sum[:window] = csum[:window]
            roll_sum[window:] = csum[window:] - csum[:-window]
            eff_n = np.minimum(n, window)

        ma_padded = (roll_sum + (window - eff_n) * ref_open) / window
        alpha_ma = alpha * ma_padded + (1 - alpha) * ref_open

        out[locs] = alpha_ma

    # 寫回 df
    df["alpha_ma"] = np.nan
    df.loc[sub.index, "alpha_ma"] = out
    return df


# ----------------------------
# 產生 trades（依你規則：target 隨時間更新、第一次 touch、18:03 截止）
# ----------------------------
def build_trades(df: pd.DataFrame, bin_step_pct: float = 0.02, cutoff_m: int = 3, fee_roundtrip: float = 0.0) -> pd.DataFrame:
    df = df.sort_index().copy()
    idx = pd.DatetimeIndex(df.index)

    if "alpha_ma" not in df.columns:
        raise ValueError("df 缺少 alpha_ma（請先 add_alpha_ma）")

    calc_mask = mask_calc_weekend(idx, cutoff_m=cutoff_m)
    work = df.loc[calc_mask, ["High", "Low", "Close", "alpha_ma"]].copy()
    widx = pd.DatetimeIndex(work.index)
    work["_ss"] = session_start_fri17(widx, start_hour=17)

    trades = []

    for ss, g in work.groupby("_ss", sort=True):
        cutoff_ts = pd.Timestamp(ss) + pd.Timedelta(days=2, hours=18, minutes=cutoff_m)
        g = g[g.index <= cutoff_ts].copy()
        if g.empty:
            continue

        times = pd.DatetimeIndex(g.index)
        m = len(g)

        # cutoff bar
        cutoff_pos = int(times.searchsorted(cutoff_ts, side="left"))
        if cutoff_pos >= m:
            cutoff_pos = m - 1
        cutoff_close = float(g["Close"].iloc[cutoff_pos])
        cutoff_time = times[cutoff_pos]

        lo = g["Low"].to_numpy(float)
        hi = g["High"].to_numpy(float)
        ma = g["alpha_ma"].to_numpy(float)
        close = g["Close"].to_numpy(float)

        touch = (lo <= ma) & (ma <= hi)

        # next touch after i
        next_touch = np.full(m + 1, -1, dtype=int)
        nxt = -1
        for i in range(m - 1, -1, -1):
            if touch[i]:
                nxt = i
            next_touch[i] = nxt
        next_touch[m] = -1

        # entry positions：只取 Fri>=17, Sat, Sun<18:00
        e_mask = mask_entry_weekend(times)
        entry_positions = np.where(e_mask)[0]

        for p in entry_positions:
            entry_time = times[p]
            entry_close = close[p]
            entry_ma = ma[p]
            if not np.isfinite(entry_close) or not np.isfinite(entry_ma) or entry_close <= 0:
                continue

            diff = entry_close - entry_ma
            if diff == 0:
                continue

            # 方向：上方做空，下方做多
            side = -np.sign(diff)

            entry_dev_pct = abs(diff) / entry_close * 100.0
            dev_bin = float(bin_floor(np.array([entry_dev_pct]), bin_step_pct)[0])

            tpos = next_touch[p + 1] if (p + 1) <= m else -1

            if tpos != -1 and tpos <= cutoff_pos:
                exit_time = times[tpos]
                exit_price = ma[tpos]      # 觸碰當下 alpha_ma
                hit = 1
                wait_bars = int(tpos - p)
            else:
                exit_time = cutoff_time
                exit_price = cutoff_close  # 未命中：用 cutoff close
                hit = 0
                wait_bars = int(cutoff_pos - p)

            signed_ret = side * (exit_price - entry_close) / entry_close
            signed_ret_net = signed_ret - fee_roundtrip
            ret_abs_pct = abs(exit_price - entry_close) / entry_close * 100.0

            trades.append({
                "session_start": pd.Timestamp(ss),
                "entry_time": entry_time,
                "exit_time": exit_time,
                "hour_bin": int(((entry_time - pd.Timestamp(ss)) // pd.Timedelta(hours=1))),  # 0~49
                "dev_bin_pct": dev_bin,
                "hit": hit,
                "wait_bars": wait_bars,
                "signed_ret_net": float(signed_ret_net),
                "ret_abs_pct": float(ret_abs_pct),
            })

    return pd.DataFrame(trades).sort_values("entry_time")


# ----------------------------
# 2D 統計 + heatmap
# ----------------------------
def make_2d_stats(trades: pd.DataFrame) -> pd.DataFrame:
    g = trades.groupby(["dev_bin_pct", "hour_bin"])
    stats = g.agg(
        n=("hit", "size"),
        hit_rate=("hit", "mean"),
        win_rate=("signed_ret_net", lambda s: (s > 0).mean()),
        avg_wait_bars=("wait_bars", "mean"),
        avg_ret_abs_pct=("ret_abs_pct", "mean"),
        avg_signed_ret_net=("signed_ret_net", "mean"),
    )
    return stats


def plot_heatmap(stats: pd.DataFrame, value_col: str, title: str, vmin=None, vmax=None,
                 scale: float = 1.0, cbar_label: str | None = None):
    """
    x: hour_bin（距離 Fri17 的小時數）
    y: dev_bin_pct
    color: value_col
    scale: 顯示倍率（例如 100 代表顯示成 %）
    """
    pivot = stats[value_col].unstack("hour_bin")  # rows=dev_bin, cols=hour_bin

    y_vals = pivot.index.to_numpy(float)
    x_vals = pivot.columns.to_numpy(int)

    Z = pivot.to_numpy(dtype=float) * scale

    plt.figure(figsize=(14, 6))
    im = plt.imshow(
        Z,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )

    if cbar_label is None:
        cbar_label = value_col
    plt.colorbar(im, label=cbar_label)

    plt.title(title)
    plt.xlabel("hour bin since Fri 17:00 (0..49)")
    plt.ylabel("deviation bin (%)")

    if len(x_vals) > 0:
        xticks = np.arange(0, len(x_vals), 4)
        plt.xticks(xticks, x_vals[xticks])
    if len(y_vals) > 0:
        step = max(1, len(y_vals)//10)
        yticks = np.arange(0, len(y_vals), step)
        plt.yticks(yticks, [f"{y_vals[i]:.2f}" for i in yticks])

    plt.tight_layout()
    plt.show()



# ----------------------------
# main
# ----------------------------
if __name__ == "__main__":
    path = r"data/PAXG_1m_weekend.parquet"  # 改成你的檔案

    window = 1000
    alpha = 0.5

    bin_step_pct = 0.05
    cutoff_m = 5
    fee_roundtrip = 0.06 / 100

    df = load_ohlcv_parquet(path)
    df = add_alpha_ma(df, window=window, alpha=alpha, cutoff_m=cutoff_m)

    trades = build_trades(df, bin_step_pct=bin_step_pct, cutoff_m=cutoff_m, fee_roundtrip=fee_roundtrip)

    trades = trades[trades["hour_bin"] >= 1].copy()


    print("trades:", len(trades))
    print(trades.head())

    stats2d = make_2d_stats(trades)

    # plot_heatmap(stats2d, "hit_rate",
    #          "Hit rate (%) by (time, deviation)",
    #          vmin=0, vmax=100,
    #          scale=100,
    #          cbar_label="hit_rate (%)")
    
    # plot_heatmap(stats2d, "avg_wait_bars",
    #             "Avg wait bars by (time, deviation)")

    plot_heatmap(stats2d, "avg_signed_ret_net",
                "Avg signed return (net, %) by (time, deviation)",
                scale=100,
                cbar_label="avg_signed_ret_net (%)")
    
    plot_heatmap(stats2d, "win_rate",
                "Win rate (%) by (time, deviation)",
                vmin=0, vmax=100,
                scale=100,
                cbar_label="win_rate (%)")

