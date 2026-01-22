# run_live.py
from __future__ import annotations

import time
import ccxt
import pandas as pd

from core import DCACore, DCAState
from broker import CCXTBroker
from setting import load_settings


def fetch_closed_bars(ex, symbol: str, timeframe: str, limit: int, tz: str) -> pd.DataFrame:
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    dt = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert(tz).dt.tz_localize(None)
    df = df.drop(columns=["ts"])
    df.index = dt
    df = df.sort_index()
    if len(df) >= 2:
        df = df.iloc[:-1]  # 丟掉最後一根未收盤
    return df


def make_exchange(name: str, api_key: str, api_secret: str, market_type: str):
    if name.lower() != "binance":
        raise ValueError(f"Unsupported exchange: {name}")

    return ccxt.binance({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
        "options": {"defaultType": market_type},  # 'future' / 'spot'
    })


def run_live():
    st = load_settings("config.yaml")

    ex = make_exchange(st.exchange, st.api_key, st.api_secret, st.market_type)
    broker = CCXTBroker(ex, st.symbol, quote=st.quote)

    core = DCACore(
        window=st.window,
        alpha=st.alpha,
        cutoff_m=st.cutoff_m,
        min_dev=st.min_dev,
        interval_minutes=st.interval_minutes,
        open_time_proportion=st.open_time_proportion,
        tz=st.tz,
    )

    tranches = int(49 * 60 // core.interval_minutes)
    state = DCAState(tranches=tranches)

    df_all = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    while True:
        try:
            df = fetch_closed_bars(ex, st.symbol, st.timeframe, limit=500, tz=st.tz)
            if not df.empty:
                df_all = pd.concat([df_all, df]).sort_index()
                df_all = df_all[~df_all.index.duplicated(keep="last")]
                if len(df_all) > st.max_bars:
                    df_all = df_all.iloc[-st.max_bars:]

            cash = broker.fetch_cash()
            pos_qty = broker.fetch_position_qty()

            intents = core.on_bar(df=df_all, state=state, cash=cash, position_qty=pos_qty)

            # 簡單風控：全關 or 限制單筆 notional
            if st.disable_trading:
                intents = [it for it in intents if it.kind in ("cancel_all", "close")]

            for it in intents:
                if it.kind == "order" and st.max_notional_per_order > 0 and it.qty:
                    last_price = float(df_all["close"].iloc[-1])
                    if it.qty * last_price > st.max_notional_per_order:
                        it.qty = st.max_notional_per_order / last_price
                broker.place_intent(it)

        except Exception as e:
            print("ERR:", repr(e))

        time.sleep(st.poll_seconds)


if __name__ == "__main__":
    run_live()