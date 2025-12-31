import math
from pathlib import Path
from typing import Optional, Union, Tuple

import ccxt
import pandas as pd
from tqdm.rich import tqdm


def download_ohlcv_binance_futures(
    symbol: str,
    timeframe: str = "1m",
    since: Union[str, int] = None,
    until: Optional[Union[str, int]] = None,
    *,
    exchange: Optional[ccxt.Exchange] = None,
    limit: int = 1000,
    tz: str = "America/New_York",
    save_dir: Union[str, Path] = "data",
    mark: str = "",
    save_csv: bool = True,
    save_parquet: bool = True,
    parquet_engine: str = "pyarrow",
    show_pbar: bool = True,
) -> Tuple[pd.DataFrame, Optional[Path], Optional[Path]]:
    """
    下載 Binance Futures OHLCV，回傳 DataFrame，並可選擇存成 CSV/Parquet。

    參數
    - symbol: e.g. 'PAXG/USDT'
    - timeframe: e.g. '1m'
    - since/until: 可傳 ISO8601 字串或毫秒 timestamp(int)
      - since: 必填(建議給)，until: 可選(不含 until 那根，與你原碼一致：candle[0] < until)
    - exchange: 可傳入已建好的 ccxt binance instance；不傳就自動建立 futures
    - limit: 每批抓多少根（Binance 常見上限 1000）
    - tz: 轉換到的時區（會轉成 naive datetime，與你原碼一致）
    - save_dir: 存檔資料夾
    - mark: 檔名後綴標記
    - save_csv/save_parquet: 是否輸出檔案
    - show_pbar: 是否顯示 tqdm 進度條

    回傳
    - df, csv_path, parquet_path（若未存檔則為 None）
    """
    if since is None:
        raise ValueError("since 不能是 None，請提供開始時間（ISO8601 或毫秒 timestamp）")

    # exchange：可重用外部傳入的（例如你想設定 proxy / api key 等）
    if exchange is None:
        exchange = ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })

    def to_ms(x: Union[str, int]) -> int:
        if isinstance(x, int):
            return x
        if isinstance(x, str):
            return exchange.parse8601(x)
        raise TypeError(f"since/until 只接受 str 或 int，現在是 {type(x)}")

    since_ms = to_ms(since)
    until_ms = to_ms(until) if until is not None else None

    ms_per_candle = exchange.parse_timeframe(timeframe) * 1000
    end_timestamp = until_ms if until_ms is not None else exchange.milliseconds()

    if since_ms >= end_timestamp:
        if show_pbar:
            print(f"start time {exchange.iso8601(since_ms)} after until time {exchange.iso8601(end_timestamp)}")
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        empty.index.name = "datetime"
        return empty, None, None

    total_candles = (end_timestamp - since_ms) // ms_per_candle + 1
    total_iters = math.ceil(total_candles / limit) if total_candles > 0 else 0

    pbar = None
    if show_pbar:
        pbar = tqdm(total=total_iters, desc=f"Fetching {symbol} {timeframe}", dynamic_ncols=True, unit="batch")

    all_ohlcv = []
    cursor = since_ms

    while True:
        current_limit = limit

        if until_ms is not None and cursor + limit * ms_per_candle > until_ms:
            remaining_candles = (until_ms - cursor) // ms_per_candle + 1
            current_limit = max(1, int(remaining_candles))

        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, cursor, current_limit)

        if not ohlcv:
            break

        # 與你原碼一致：until 不包含那根（< until）
        if until_ms is not None:
            ohlcv = [c for c in ohlcv if c[0] < until_ms]
            if not ohlcv:
                break

        all_ohlcv.extend(ohlcv)
        new_cursor = ohlcv[-1][0] + ms_per_candle

        if pbar is not None:
            pbar.update(1)

        if (until_ms is not None and new_cursor >= until_ms) or (len(ohlcv) < current_limit):
            break

        cursor = new_cursor

    if pbar is not None:
        pbar.close()

    if not all_ohlcv:
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        empty.index.name = "datetime"
        return empty, None, None

    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df.index = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.index = df.index.tz_convert(tz).tz_localize(None)
    df.drop(columns=["timestamp"], inplace=True)
    df.index.name = "datetime"
    df = df.sort_index()

    csv_path = None
    pq_path = None

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    df = df.sort_index()
    dup = df.index.duplicated(keep="last")
    if dup.any():
        df = df[~dup]

    base_name = f"{symbol.split('/')[0]}_{timeframe}"
    if mark:
        base_name = f"{base_name}({mark})"

    if save_csv:
        csv_path = save_dir / f"{base_name}.csv"
        df.to_csv(csv_path, index=True)

    if save_parquet:
        pq_path = save_dir / f"{base_name}.parquet"
        df.to_parquet(pq_path, engine=parquet_engine)

    return df, csv_path, pq_path


if __name__ == "__main__":
    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })

    df, csv_path, pq_path = download_ohlcv_binance_futures(
        symbol="PAXG/USDT",
        timeframe="5m",
        since="2025-12-20T00:00:00Z",
        until="2025-12-26T00:00:00Z",
        exchange=exchange,
        save_dir="data",
        mark="",
    )

    print(df.head())
    print("CSV:", csv_path)
    print("Parquet:", pq_path)
