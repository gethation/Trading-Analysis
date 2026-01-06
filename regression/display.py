from pathlib import Path
import numpy as np
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from tqdm.rich import tqdm


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

    # ✅ 保險：先排序 + 去除重複時間（避免後面 pandas 對齊爆 duplicate labels）
    df = df.sort_index()
    dup = df.index.duplicated(keep="last")
    if dup.any():
        df = df[~dup]

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
    end_minutes: int = 0,               # ✅ 新增：週日 18:00 + end_minutes
):
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha 必須在 [0, 1]")
    if end_minutes < 0:
        raise ValueError("end_minutes 不能是負數")

    df = df.copy()
    idx = pd.DatetimeIndex(df.index)

    dow = idx.dayofweek
    minutes = idx.hour * 60 + idx.minute

    # 週末結束分鐘（含 18:00 那根）
    end_minute_of_sun = 18 * 60 + int(end_minutes)

    in_weekend = (
        ((dow == 4) & (minutes >= session_start_hour * 60)) |
        (dow == 5) |
        ((dow == 6) & (minutes <= end_minute_of_sun))   # ✅ 改成 <= 並可延長
    )

    # 每一根 bar 都對應到它的「週末 session 起點」：Fri 17:00
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

    # --- MA' = alpha*MA + (1-alpha)*ref_open ---
    ma_blend = alpha * wk[f"MA{window}_padded"].to_numpy() + (1 - alpha) * ref_open.to_numpy()
    wk[f"MA{window}_blend_a{alpha:g}"] = ma_blend

    # ✅ 注意：wk.index 必須唯一（load 時已去重；仍保險一下）
    wk = wk[~wk.index.duplicated(keep="last")]

    out_ma.loc[wk.index] = wk[f"MA{window}_padded"]
    out_map.loc[wk.index] = wk[f"MA{window}_blend_a{alpha:g}"]

    df[f"MA{window}_padded"] = out_ma
    df[f"MA{window}_blend_a{alpha:g}"] = out_map

    return df.drop(columns=["_session_start"])


def plot_ohlcv_with_mas(df, ma_cols, title, save_path=None):
    apds = [mpf.make_addplot(df[c], panel=0) for c in ma_cols]

    fig, axes = mpf.plot(
        df,
        type="candle",
        volume=True,
        addplot=apds,
        style="yahoo",
        figsize=(20, 8),
        returnfig=True,
    )

    fig.suptitle(title, fontsize=14, y=0.98, fontname="Arial")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.4)
    plt.close(fig)

def save_all_weekend_charts(
    df: pd.DataFrame,
    out_dir: str | Path = "display_result",
    ma_cols: list[str] | None = None,
    title_prefix: str = "",
    session_start_hour: int = 17,
    end_minutes: int = 0,
    plot_last_hours: float = 2,   # ✅ 新增：只畫最後 N 小時
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    idx = pd.DatetimeIndex(df.index)
    dow = idx.dayofweek

    # 找所有週五 session_start（Fri 17:00）
    session_start = idx.normalize() + pd.Timedelta(hours=session_start_hour)
    is_fri_session = (dow == 4) & (idx >= session_start)
    session_starts = pd.DatetimeIndex(sorted(pd.unique(session_start[is_fri_session])))

    if len(session_starts) == 0:
        print("[INFO] 找不到任何週末 session（沒有週五 17:00 之後的資料）")
        return

    if ma_cols is None:
        ma_cols = [c for c in df.columns if c.startswith("MA")]

    # 週末結束：Fri 17:00 + 2天 + 1小時 + end_minutes
    span = pd.Timedelta(days=2, hours=1, minutes=int(end_minutes))

    last_span = pd.Timedelta(hours=float(plot_last_hours))

    for ss in tqdm(session_starts):
        ee = ss + span

        # 起點 = max(session start, ee - last_span)
        plot_start = max(ss, ee - last_span)

        seg = df[(df.index >= plot_start) & (df.index <= ee)].copy()
        if seg.empty:
            continue

        # 檔名：weekend_YYYYMMDD_1700_last2h.png
        fname = f"weekend_{ss:%Y%m%d_%H%M}_last{plot_last_hours:g}h.png"
        save_path = out_dir / fname

        title = (
            f"{title_prefix} last {plot_last_hours:g}h "
            f"({plot_start:%Y-%m-%d %H:%M} ~ {ee:%Y-%m-%d %H:%M})"
        )
        plot_ohlcv_with_mas(seg, ma_cols=ma_cols, title=title, save_path=save_path)

    print(f"[OK] 已輸出週末圖表到: {out_dir.resolve()}")



if __name__ == "__main__":
    # 你原本的輸入檔
    path = r"data/PAXG_1m_weekend.parquet"
    df = load_ohlcv_parquet(path)

    window = 500
    alpha = 0.5

    end_minutes = 0

    df = add_padded_and_blended_ma_weekend(df, window=window, alpha=alpha, end_minutes=end_minutes)

    ma_cols = [f"MA{window}_padded", f"MA{window}_blend_a{alpha:g}"]
    title_prefix = f"{Path(path).stem}"

    save_all_weekend_charts(
        df,
        out_dir="regression/display_result",
        ma_cols=ma_cols,
        title_prefix=title_prefix,
        end_minutes=end_minutes,
        plot_last_hours=2,
    )

