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
    bt = Backtest(df, DCA_Strategy, cash=1_000_000_000, commission=0.03/100, margin = 0.02)
    stats = bt.run(window = 1000, 
                alpha = 0.5, 
                cutoff_m = 5,
                min_dev = 0.10 / 100,
                interval_minutes = 10,
                open_time_proportion = 0.6)
    print(stats)
    # trades = stats['_trades']
    # print("num trades:", len(trades))
    # print(trades[['EntryTime','ExitTime','EntryPrice','ExitPrice','Size','PnL']].tail(20))
    bt.plot(filename=r"regression\reports\Strategy.html")