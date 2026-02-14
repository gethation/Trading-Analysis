from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import backtrader as bt

from mean_reversion_strat import load_merged_df, PxPandasData, PairsMeanReversion


def run_and_get_equity(
    data_path: str = "data/spread_ratio_PAXG_XAUT_1m.parquet",
    cash: float = 100000.0,
    commission: float = 0.02 / 100,

    # strategy params (照你 backtest.py 現在的預設)
    lookback: int = 1000,
    entry_z: float = 2.0,
    exit_buffer_z: float = 1.0,
    stop_z: float = 100.0,
    leg_cash: float = 45000.0,
    hedge_beta: float = 1.0,
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
        lookback=lookback,
        entry_z=entry_z,
        exit_buffer_z=exit_buffer_z,
        stop_z=stop_z,
        leg_cash=leg_cash,
        hedge_beta=hedge_beta,
        allow_flip=True,
        printlog=False,
    )

    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn")

    strat = cerebro.run(maxcpus=1)[0]

    # TimeReturn: dict[datetime -> return]
    timeret = strat.analyzers.timereturn.get_analysis()
    ret = pd.Series(timeret, dtype="float64")
    ret.index = pd.to_datetime(ret.index)
    ret = ret.sort_index()

    # equity（以 cash 為起點）
    equity = (1.0 + ret).cumprod() * float(cash)
    return equity


def plot_equity_curve(
    equity: pd.Series,
    title: str = "Equity Curve",
    save_csv: str | None = "equity_curve.csv",
    save_png: str | None = "equity_curve.png",
    resample_to_day: bool = False,
):
    if resample_to_day:
        equity = equity.resample("1D").last().dropna()

    if save_csv:
        equity.to_frame("equity").to_csv(save_csv)
        print(f"Saved: {save_csv}")

    plt.figure()
    plt.plot(equity.index, equity.values)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Equity")

    plt.tight_layout()

    if save_png:
        plt.savefig(save_png, dpi=150)
        print(f"Saved: {save_png}")

    plt.show()


if __name__ == "__main__":
    equity = run_and_get_equity(
        data_path="data/spread_ratio_PAXG_XAUT_1m.parquet",
        cash=100000.0,
        commission=0.01/100,

        lookback=1000,
        entry_z=2.0,
        exit_buffer_z=1.5,
        stop_z=100.0,
        leg_cash=45000.0,
        hedge_beta=1.0,
    )

    plot_equity_curve(
        equity,
        title="PAXG/XAUT Pair Trading - Equity Curve",
        save_csv="equity_curve.csv",
        save_png="equity_curve.png",
        resample_to_day=True,
    )
