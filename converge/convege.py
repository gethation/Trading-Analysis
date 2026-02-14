# spread_paxg_xaut.py
from pathlib import Path
import ccxt
import pandas as pd
import sys
import warnings
from tqdm import TqdmExperimentalWarning

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccxt_bybit_fetcher import resolve_bybit_swap_symbol, download_ohlcv_ccxt


def compute_sym_spread_ratio(
    paxg_close: pd.Series,
    xaut_close: pd.Series,
) -> pd.Series:
    """
    (PAXG - XAUT) / (PAXG + XAUT) * 200
    """
    denom = (paxg_close + xaut_close)
    ratio = (paxg_close - xaut_close) / denom * 200.0
    ratio = ratio.replace([float("inf"), float("-inf")], pd.NA)
    return ratio


def main():
    timeframe = "1m"
    since = "2025-12-15T00:00:00Z"
    until = "2026-02-07T00:00:00Z"
    tz = "America/New_York"  # 你抓資料那邊如果有固定 tz，這裡也要一致
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Bybit 永續
    exchange = ccxt.bybit({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })

    paxg_symbol = resolve_bybit_swap_symbol(exchange, "PAXG", "USDT")
    xaut_symbol = resolve_bybit_swap_symbol(exchange, "XAUT", "USDT")
    print("Resolved:", paxg_symbol, xaut_symbol)

    # 抓兩個標的的 OHLCV
    paxg_df, _, _ = download_ohlcv_ccxt(
        symbol=paxg_symbol,
        timeframe=timeframe,
        since=since,
        until=until,
        exchange=exchange,
        tz=tz,
        save_csv=False,
        save_parquet=False,
        show_pbar=True,
    )

    xaut_df, _, _ = download_ohlcv_ccxt(
        symbol=xaut_symbol,
        timeframe=timeframe,
        since=since,
        until=until,
        exchange=exchange,
        tz=tz,
        save_csv=False,
        save_parquet=False,
        show_pbar=True,
    )

    # 對齊時間（用 close）
# convege.py 內，抓完 paxg_df / xaut_df 後，取代你原本「只用 close 對齊」那段

# 對齊時間（用 OHLC）
    merged = (
        pd.concat(
            [
                paxg_df[["open", "high", "low", "close"]].add_prefix("PAXG_"),
                xaut_df[["open", "high", "low", "close"]].add_prefix("XAUT_"),
            ],
            axis=1,
        )
        .dropna(how="any")
        .sort_index()
    )

    # 用各自 OHLC 計算每根 bar 的 ratio OHLC
    for k in ["open", "high", "low", "close"]:
        merged[f"ratio_{k}"] = compute_sym_spread_ratio(
            merged[f"PAXG_{k}"],
            merged[f"XAUT_{k}"],
        )

    # （可選）如果你只想存 ratio OHLC，不想把原始價也存進去：
    # merged = merged[[f"ratio_{k}" for k in ["open","high","low","close"]]]

    csv_path = out_dir / f"spread_ratio_PAXG_XAUT_{timeframe}.csv"
    pq_path = out_dir / f"spread_ratio_PAXG_XAUT_{timeframe}.parquet"
    merged.to_csv(csv_path, index=True)
    merged.to_parquet(pq_path, index=True)



if __name__ == "__main__":
    main()
