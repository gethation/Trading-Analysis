import math
import ccxt
from tqdm.rich import tqdm
import pandas as pd
exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {"defaultType": "future"},
})

symbol    = 'PAXG/USDT'
timeframe = '5m'
mark      = ''
since     = exchange.parse8601('2025-12-20T00:00:00Z')
limit     = 1000
until     = None
until     = exchange.parse8601('2025-12-26T00:00:00Z')

ms_per_candle = exchange.parse_timeframe(timeframe) * 1000

end_timestamp = until if until is not None else exchange.milliseconds()
if since >= end_timestamp:
    total_candles = 0
    print(f"start time {exchange.iso8601(since)} after until time {exchange.iso8601(end_timestamp)}")
else:
    total_candles = (end_timestamp - since) // ms_per_candle + 1

total_iters   = math.ceil(total_candles / limit) if total_candles > 0 else 0


pbar = tqdm(total=total_iters, desc=f'Fetching {symbol} {timeframe}', dynamic_ncols=True, unit="batch")

all_ohlcv = []
if total_candles > 0:
    while True:
        current_limit = limit
        if until is not None:
            if since + limit * ms_per_candle > until:
                remaining_candles = (until - since) // ms_per_candle + 1
                current_limit = max(1, int(remaining_candles))

        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, current_limit)

        if not ohlcv:
            break

        if until is not None:
            ohlcv = [candle for candle in ohlcv if candle[0] < until]

        if not ohlcv:
            break

        all_ohlcv.extend(ohlcv)
        new_since = ohlcv[-1][0] + ms_per_candle

        pbar.update(1)
        if (until is not None and new_since >= until) or len(ohlcv) < current_limit:
            break
        since = new_since


pbar.close()

if not all_ohlcv:
    print(f"did not download {symbol} {timeframe} data")
else:
    df = pd.DataFrame(all_ohlcv, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume'
    ])
    df.index = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.index = df.index.tz_convert('America/New_York')
    df.index = df.index.tz_localize(None)
    df.drop(columns=['timestamp'], inplace=True)

    df.index.name = "datetime"
    df = df.sort_index()

    base = fr"data\{symbol.split('/')[0]}_{timeframe}"
    if mark != '':
        base = fr"{base}({mark})"

    csv_path = base + ".csv"
    pq_path  = base + ".parquet"

    df.to_csv(csv_path, index=True)
    df.to_parquet(pq_path, engine="pyarrow")

    print(f'{symbol} {timeframe} data save to:\nCSV     -> {csv_path}\nParquet -> {pq_path}')
