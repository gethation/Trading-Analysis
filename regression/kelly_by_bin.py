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

    # 去掉重複 datetime（避免後面任何 reindex/對齊問題）
    if df.index.has_duplicates:
        dup_n = int(df.index.duplicated(keep="last").sum())
        print(f"[WARN] index has duplicates: {dup_n} rows -> keep last")
        df = df[~df.index.duplicated(keep="last")].copy()

    # 欄位統一成 Open/High/Low/Close/Volume
    lower_cols = {c.lower(): c for c in df.columns}
    mapping = {}
    for k in ["open", "high", "low", "close", "volume"]:
        if k in lower_cols:
            mapping[lower_cols[k]] = k.capitalize() if k != "volume" else "Volume"
    df = df.rename(columns=mapping)

    need = {"Open", "High", "Low", "Close", "Volume"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"缺少欄位: {miss}")

    return df[["Open", "High", "Low", "Close", "Volume"]].copy()


# ----------------------------
# 週末 window（NY 時間，index 已是 NY naive 或你確定它代表 NY）
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
    """計算/追蹤用：Fri>=17:00, Sat 全日, Sun<18:03（用來讓 alpha_ma/touch 覆蓋到 cutoff）"""
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
    """把每筆映射到該週末 session 的 Fri 17:00（用於 groupby session）"""
    dow = np.asarray(idx.dayofweek)
    delta_days = dow - 4
    delta_days = np.where(delta_days < 0, 0, delta_days)
    ss = idx.normalize() + pd.Timedelta(hours=start_hour) - pd.to_timedelta(delta_days, unit="D")
    return np.asarray(ss)


def bin_floor(values: np.ndarray, bin_size: float) -> np.ndarray:
    """向下分箱：0.31->0.30（bin=0.02）；0.25->0.24"""
    k = np.floor(values / bin_size + 1e-12).astype(int)
    return np.round(k * bin_size, 6)


# ----------------------------
# 1) 計算 alpha_ma（padded MA + blend 到 Fri17 Open）
# ----------------------------
def add_alpha_ma(
    df: pd.DataFrame,
    window: int = 500,
    alpha: float = 0.7,
    cutoff_m: int = 3,
) -> pd.DataFrame:
    """
    在週末計算區間 (Fri>=17, Sat, Sun<18:03) 上：
      MA_padded: 不足 window 用 session 第一根 Open 補
      alpha_ma : alpha*MA_padded + (1-alpha)*Fri17:00 Open（若缺，用 session 第一根 Open）
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha 必須在 [0,1]")

    df = df.copy()
    idx = pd.DatetimeIndex(df.index)

    calc_mask = mask_calc_weekend(idx, cutoff_m=cutoff_m)
    calc_idx = idx[calc_mask]

    alpha_ma = np.full(len(df), np.nan, dtype=float)

    if len(calc_idx) == 0:
        df["alpha_ma"] = alpha_ma
        return df

    ss_all = session_start_fri17(calc_idx, start_hour=17)
    sub = df.loc[calc_mask, ["Open", "Close"]].copy()
    sub["_ss"] = ss_all

    # 逐 session 算 padded MA & alpha_ma
    out = np.empty(len(sub), dtype=float)
    out[:] = np.nan

    # 需要子集的位置映射（避免 index 對齊）
    sub_pos = np.arange(len(sub))

    for ss, grp in sub.groupby("_ss", sort=True):
        pos = grp.index
        locs = sub_pos[sub["_ss"].to_numpy() == ss]  # 對應在 sub 的位置

        close = grp["Close"].to_numpy(dtype=float)
        open_ = grp["Open"].to_numpy(dtype=float)

        # ref_open：Fri 17:00 那根 Open（缺就用第一根 Open）
        gidx = pd.DatetimeIndex(grp.index)
        is_fri17 = (gidx.dayofweek == 4) & (gidx.hour == 17) & (gidx.minute == 0)
        ref_open = float(open_[np.argmax(is_fri17)]) if is_fri17.any() else float(open_[0])

        # padded MA：rolling sum（min_periods=1）+ padding
        # 這裡用 cumsum 快速算「從 session 開始到現在」的前綴和，再搭配 window 做 rolling
        csum = np.cumsum(close)
        n = np.arange(1, len(close) + 1, dtype=int)

        # rolling sum with window
        roll_sum = np.empty_like(close)
        if len(close) <= window:
            roll_sum = csum
            eff_n = n
        else:
            roll_sum[:window] = csum[:window]
            roll_sum[window:] = csum[window:] - csum[:-window]
            eff_n = np.minimum(n, window)

        ma_padded = (roll_sum + (window - eff_n) * ref_open) / window
        alpha_ma_grp = alpha * ma_padded + (1 - alpha) * ref_open

        out[locs] = alpha_ma_grp

    # 寫回 df
    alpha_ma[calc_mask] = out
    df["alpha_ma"] = alpha_ma
    return df


# ----------------------------
# 2) 產生每筆交易並分箱
# ----------------------------
def build_trades(
    df: pd.DataFrame,
    bin_size_pct: float = 0.02,
    cutoff_m: int = 3,
    fee_roundtrip: float = 0.0,   # 以「小數」表示，例如 0.0008 = 0.08%（進出總成本）
) -> pd.DataFrame:
    """
    交易規則（依你描述）：
      - entry 只取 Fri>=17, Sat, Sun<18:00 的每根 bar，用 Close 進場
      - target 隨時間更新：touch 當下用該 bar 的 alpha_ma
      - 第一次 touch 就結算：Low<=alpha_ma<=High
      - 截止：Sun 18:03；未命中則 forced exit，exit_level= cutoff 那根 Close（當 alpha_ma）
      - 方向：Close>alpha_ma 做空；Close<alpha_ma 做多（Close==alpha_ma 略過）
      - 帶符號報酬：side*(exit_price-entry_close)/entry_close - fee_roundtrip
      - bin：entry_dev_pct = abs(Close-alpha_ma)/Close*100，floor 到 0.02%
    """
    df = df.sort_index().copy()
    idx = pd.DatetimeIndex(df.index)

    if "alpha_ma" not in df.columns:
        raise ValueError("df 缺少 alpha_ma（請先呼叫 add_alpha_ma）")

    entry_mask = mask_entry_weekend(idx)
    calc_mask = mask_calc_weekend(idx, cutoff_m=cutoff_m)

    work = df.loc[calc_mask, ["Open", "High", "Low", "Close", "alpha_ma"]].copy()
    widx = pd.DatetimeIndex(work.index)
    work["_ss"] = session_start_fri17(widx, start_hour=17)

    trades = []

    for ss, grp in work.groupby("_ss", sort=True):
        # cutoff ts = ss + 2days + 18:03
        cutoff_ts = pd.Timestamp(ss) + pd.Timedelta(days=2, hours=18, minutes=cutoff_m)

        g = grp[grp.index <= cutoff_ts].copy()
        if g.empty:
            continue

        times = pd.DatetimeIndex(g.index)
        m = len(g)

        # cutoff bar：第一根 >= cutoff_ts，若沒有則最後一根
        cutoff_pos = int(times.searchsorted(cutoff_ts, side="left"))
        if cutoff_pos >= m:
            cutoff_pos = m - 1
        cutoff_close = float(g["Close"].iloc[cutoff_pos])
        cutoff_time = times[cutoff_pos]

        lo = g["Low"].to_numpy(float)
        hi = g["High"].to_numpy(float)
        ma = g["alpha_ma"].to_numpy(float)
        close = g["Close"].to_numpy(float)

        # touch: Low<=alpha_ma<=High （alpha_ma 隨時間更新）
        touch = (lo <= ma) & (ma <= hi)

        # next touch after i（strictly after entry）
        next_touch = np.full(m + 1, -1, dtype=int)
        nxt = -1
        for i in range(m - 1, -1, -1):
            if touch[i]:
                nxt = i
            next_touch[i] = nxt
        next_touch[m] = -1

        # 取 entry 時間點：必須在 entry_mask 內
        # 但 entry_mask 是對原 df；我們要在 g 的時間點上判斷是否屬於 entry window
        e_mask = mask_entry_weekend(times)  # 對這個 session 的 times 判斷
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
            side = -np.sign(diff)  # close>ma -> short(-1), close<ma -> long(+1)

            entry_dev_pct = abs(diff) / entry_close * 100.0
            dev_bin = float(bin_floor(np.array([entry_dev_pct]), bin_size_pct)[0])

            tpos = next_touch[p + 1] if (p + 1) <= m else -1

            if tpos != -1 and tpos <= cutoff_pos:
                exit_time = times[tpos]
                exit_price = ma[tpos]          # 觸碰當下的 alpha_ma
                hit = 1
                wait_bars = int(tpos - p)
            else:
                exit_time = cutoff_time
                exit_price = cutoff_close      # 未命中：用 cutoff close 當 alpha_ma
                hit = 0
                wait_bars = int(cutoff_pos - p)

            signed_ret = side * (exit_price - entry_close) / entry_close
            signed_ret_net = signed_ret - fee_roundtrip

            trades.append({
                "dev_bin_pct": dev_bin,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "hit": hit,
                "wait_bars": wait_bars,
                "side": float(side),
                "entry_dev_pct": float(entry_dev_pct),
                "signed_ret": float(signed_ret),
                "signed_ret_net": float(signed_ret_net),
            })

    return pd.DataFrame(trades).sort_values("entry_time")


# ----------------------------
# 3) 以 bin 統計 Kelly 的 P、B、proportion(f*)
# ----------------------------
def kelly_table_by_bin(trades: pd.DataFrame, use_net: bool = True) -> pd.DataFrame:
    col = "signed_ret_net" if use_net else "signed_ret"

    rows = []
    for b, sub in trades.groupby("dev_bin_pct"):
        r = sub[col].to_numpy(dtype=float)
        r = r[np.isfinite(r)]
        if len(r) == 0:
            continue

        wins = r[r > 0]
        losses = r[r < 0]

        P = len(wins) / len(r)

        # B = 平均贏幅 / 平均輸幅（輸幅取正）
        if len(wins) == 0 or len(losses) == 0:
            B = np.nan
            f = np.nan
        else:
            W = float(np.mean(wins))
            L = float(np.mean(-losses))
            B = (W / L) if L > 0 else np.nan
            f = (B * P - (1 - P)) / B if (B is not np.nan and B > 0) else np.nan

        rows.append({
            "dev_bin_pct": float(b),
            "n": int(len(r)),
            "P": float(P),
            "B": float(B) if np.isfinite(B) else np.nan,
            "proportion_f": float(f) if np.isfinite(f) else np.nan,  # Kelly f*
            "hit_rate": float(sub["hit"].mean()),
            "avg_wait_bars": float(sub["wait_bars"].mean()),
        })

    out = pd.DataFrame(rows).set_index("dev_bin_pct").sort_index()
    return out


# ----------------------------
# 4) 畫三張長條圖：P / B / proportion
# ----------------------------
def _bar_plot(x: np.ndarray, y: np.ndarray, step: float, xlabel: str, ylabel: str, title: str):
    width = step * 0.9
    plt.figure()
    plt.bar(x, y, width=width, align="center")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y")
    plt.show()

def plot_p_b_prop(kelly_tbl: pd.DataFrame, bin_step: float = 0.02, kelly_scale: float = 0.5, clamp01: bool = True):
    if kelly_tbl.empty:
        print("kelly table is empty")
        return

    x = kelly_tbl.index.to_numpy(dtype=float)

    # P
    yP = kelly_tbl["P"].to_numpy(dtype=float)
    _bar_plot(
        x, yP, bin_step,
        xlabel="deviation bin (%)",
        ylabel="P (win probability)",
        title="P by deviation bin"
    )

    # B
    yB = kelly_tbl["B"].to_numpy(dtype=float)
    _bar_plot(
        x, yB, bin_step,
        xlabel="deviation bin (%)",
        ylabel="B (avg win / avg loss)",
        title="B by deviation bin"
    )

    # proportion = Kelly fraction（可選縮放 half-kelly & clamp）
    f = kelly_tbl["proportion_f"].to_numpy(dtype=float) * kelly_scale
    if clamp01:
        f = np.clip(f, 0.0, 1.0)

    _bar_plot(
        x, f, bin_step,
        xlabel="deviation bin (%)",
        ylabel=f"Kelly proportion (scale={kelly_scale}, clamp01={clamp01})",
        title="Kelly proportion by deviation bin"
    )


# ----------------------------
# main
# ----------------------------
if __name__ == "__main__":
    path = r"data/PAXG_5m_weekend.parquet"   # 改成你的檔案

    # alpha_ma 參數
    window = 500
    alpha = 0.75

    # 統計參數
    bin_step_pct = 0.02
    cutoff_m = 3

    fee_roundtrip = 0.06 / 100

    kelly_scale = 1.0
    clamp01 = True

    df = load_ohlcv_parquet(path)
    df = add_alpha_ma(df, window=window, alpha=alpha, cutoff_m=cutoff_m)

    trades = build_trades(df, bin_size_pct=bin_step_pct, cutoff_m=cutoff_m, fee_roundtrip=fee_roundtrip)
    print(trades.head())

    kelly_tbl = kelly_table_by_bin(trades, use_net=True)
    print(kelly_tbl.head(30))

    plot_p_b_prop(kelly_tbl, bin_step=bin_step_pct, kelly_scale=kelly_scale, clamp01=clamp01)
