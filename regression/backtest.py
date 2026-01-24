import pandas as pd
from backtesting import Backtest
from optimized_strategy import DCA_Strategy

import warnings

warnings.filterwarnings(
    "ignore",
    message=r"Data contains too many candlesticks to plot; downsampling to .*",
    category=UserWarning,
    module=r"backtesting\._plotting"
)


def load_data(path):
    df = pd.read_parquet(path)

    df = df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    })

    df = df[["Open", "High", "Low", "Close", "Volume"]].sort_index()
    df = df[~df.index.duplicated(keep="first")]

    return df


if __name__ == "__main__":
    df = load_data(r"data\PAXG_1m_weekend.parquet")
    bt = Backtest(df, DCA_Strategy, cash=1_000_000_000, commission=0.03/100)
    stats = bt.run(
        window=1000,
        alpha=0.5,
        cutoff_m=10,
        above_min_dev=0.15 / 100,
        below_min_dev=0.15 / 100,
        interval_minutes=10,
        open_time_proportion=0.5
    )
    print(stats)
    bt.plot(filename=r"regression\reports\Strategy.html")
