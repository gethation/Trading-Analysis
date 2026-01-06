import pandas as pd
from backtesting import Backtest, Strategy
from stratgey import DCA_Strategy

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
    bt = Backtest(df, DCA_Strategy, cash=100_000, commission=0.0)
    stats = bt.run(window = 500, 
                alpha = 0.5, 
                cutoff_m = 5,
                min_dev = 0.25 / 100,
                tranche = 100)
    print(stats)
    bt.plot(filename=r"regression\reports\Strategy.html")