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
    """計算 alpha_ma 用：Sun<18:00+cutoff_m"""
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
    """向下分箱：0.31->0.30（bin=0.02）；0.25->0.24"""
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

    df["alpha_ma"] = np.nan
    df.loc[sub.index, "alpha_ma"] = out
    return df


# ----------------------------
# 偏離分布（不分時間）+ 過濾小機率/小樣本 bins
# ----------------------------
def deviation_bin_probability(
    df: pd.DataFrame,
    bin_step_pct: float = 0.02,
    min_prob: float = 0.001,   # 0.1%
    min_count: int = 50,
    renormalize: bool = False
) -> pd.DataFrame:
    """
    統計週末 (Fri17~Sun18) 每根 bar 的偏離%（相對 alpha_ma）的分箱出現機率。
    - min_prob: 低於此機率的 bin 會被去掉
    - min_count: 低於此樣本數的 bin 會被去掉
    - renormalize: 去掉後是否把剩餘 prob 重新正規化為總和=1
    """
    idx = pd.DatetimeIndex(df.index)
    entry_mask = mask_entry_weekend(idx)

    sub = df.loc[entry_mask, ["Close", "alpha_ma"]].dropna()
    if sub.empty:
        return pd.DataFrame(columns=["count", "prob"]).set_index(pd.Index([], name="dev_bin_pct"))

    close = sub["Close"].to_numpy(float)
    ma = sub["alpha_ma"].to_numpy(float)

    dev_pct = np.abs(close - ma) / close * 100.0
    dev_bin = bin_floor(dev_pct, bin_step_pct)

    s = pd.Series(dev_bin, name="dev_bin_pct")
    counts = s.value_counts().sort_index()
    prob = counts / counts.sum()

    out = pd.DataFrame({"count": counts.astype(int), "prob": prob.astype(float)})
    out.index.name = "dev_bin_pct"

    # ✅ 過濾小機率 / 小樣本
    out = out[(out["prob"] >= min_prob) & (out["count"] >= min_count)].copy()

    # ✅（可選）重新正規化
    if renormalize and len(out) > 0:
        out["prob"] = out["prob"] / out["prob"].sum()

    return out


def plot_prob_bar(prob_tbl: pd.DataFrame, bin_step_pct: float, title: str = "Deviation bin probability",
                  label_every: int = 1):
    """
    label_every=1 代表每個 bin 都標；2 代表每隔一個標一次（較不擠）
    rotate: x 軸標籤旋轉角度
    """
    if prob_tbl.empty:
        print("No bins to plot (after filtering).")
        return

    x = prob_tbl.index.to_numpy(float)
    y = prob_tbl["prob"].to_numpy(float) * 100.0  # %

    width = bin_step_pct * 0.9

    plt.figure(figsize=(14, 5))
    plt.bar(x, y, width=width, align="center")
    plt.xlabel("deviation bin (%)")
    plt.ylabel("probability (%)")
    plt.title(title)
    plt.grid(True, axis="y")

    ticks = x[::label_every]
    plt.xticks(ticks, [f"{v:.2f}" for v in ticks])

    plt.tight_layout()
    plt.show()



# ----------------------------
# main
# ----------------------------
if __name__ == "__main__":
    path = r"data/PAXG_1m_weekend.parquet"  # 改成你的檔案

    # alpha_ma 參數（要和你策略一致）
    window = 1000
    alpha = 0.0
    cutoff_m = 5

    bin_step_pct = 0.05

    min_prob = 0.1 / 100
    min_count = 50
    renormalize = True  # True 表示保留下來的 bins 重新加總=100%

    df = load_ohlcv_parquet(path)
    df = add_alpha_ma(df, window=window, alpha=alpha, cutoff_m=cutoff_m)

    prob_tbl = deviation_bin_probability(
        df,
        bin_step_pct=bin_step_pct,
        min_prob=min_prob,
        min_count=min_count,
        renormalize=renormalize
    )
    
    min_deviation = 0.20
    prob_tbl = prob_tbl[prob_tbl.index >= min_deviation].copy()

    if not prob_tbl.empty:
        prob_tbl["prob"] = prob_tbl["prob"] / prob_tbl["prob"].sum()



    print(prob_tbl.head(50))
    print("bins:", len(prob_tbl), "total samples(after filter):", int(prob_tbl["count"].sum()))
    if not prob_tbl.empty:
        print("sum(prob):", float(prob_tbl["prob"].sum()))

    plot_prob_bar(prob_tbl, bin_step_pct=bin_step_pct, label_every=1)

