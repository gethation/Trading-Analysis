from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import backtrader as bt
from mean_reversion_strat import load_merged_df, PxPandasData, PairsMeanReversion

def run_backtest(
    data_path: str = "data/spread_ratio_PAXG_XAUT_1m.parquet",
    cash: float = 100000.0,
    commission: float = 0.02/100,
):
    cerebro = bt.Cerebro(stdstats=False)

    df = load_merged_df(Path(data_path))

    paxg_df = df[["PAXG_open", "PAXG_high", "PAXG_low", "PAXG_close"]].rename(
        columns={"PAXG_open": "open", "PAXG_high": "high", "PAXG_low": "low", "PAXG_close": "close"}
    )
    xaut_df = df[["XAUT_open", "XAUT_high", "XAUT_low", "XAUT_close"]].rename(
        columns={"XAUT_open": "open", "XAUT_high": "high", "XAUT_low": "low", "XAUT_close": "close"}
    )

    data_paxg = PxPandasData(dataname=paxg_df, name="PAXG")
    data_xaut = PxPandasData(dataname=xaut_df, name="XAUT")

    cerebro.adddata(data_paxg)
    cerebro.adddata(data_xaut)

    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)

    cerebro.addstrategy(
        PairsMeanReversion,
        lookback=1000,
        entry_z=2.5,
        exit_z=-0.5,
        stop_z=100.0,
        leg_cash=45000.0,
        hedge_beta=1.0,
        allow_flip=True,
        printlog=False,
    )

    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio,
        _name="sharpe",
        timeframe=bt.TimeFrame.Days,
        annualize=True,
        riskfreerate=0.0,
    )
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addobserver(bt.observers.Value)
    cerebro.addobserver(bt.observers.BuySell)
    cerebro.addobserver(bt.observers.DrawDown)

    print("start")
    strat = cerebro.run(maxcpus=1)[0]

    start_value = cash
    end_value = cerebro.broker.getvalue()

    total_return = end_value / start_value - 1.0

    print(f"Start Value: {start_value:.2f}")
    print(f"Final Value: {end_value:.2f}")
    print(f"Total Return: {total_return * 100:.2f}%")

    print("Sharpe:", f"{strat.analyzers.sharpe.get_analysis().get('sharperatio', float('nan')):.2f}")
    cerebro.plot(iplot=False, volume=False)



if __name__ == "__main__":
    run_backtest()
