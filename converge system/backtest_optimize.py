from __future__ import annotations

from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import backtrader as bt
from concurrent.futures import ProcessPoolExecutor, as_completed

from mean_reversion_strat import load_merged_df, PxPandasData, PairsMeanReversion


def run_backtest_once_worker(args: dict) -> dict:
    """
    ✅ 進程 worker：只吃純資料（dict/float/int/str），自己建立 Cerebro 跑一次。
    """
    data_path = args["data_path"]
    cash = args["cash"]
    commission = args["commission"]
    lookback = args["lookback"]
    entry_z = args["entry_z"]
    exit_buffer_z = args["exit_buffer_z"]
    stop_z = args["stop_z"]
    leg_cash = args["leg_cash"]
    hedge_beta = args["hedge_beta"]

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
        lookback=int(lookback),
        entry_z=float(entry_z),
        exit_buffer_z=float(exit_buffer_z),
        stop_z=float(stop_z),
        leg_cash=float(leg_cash),
        hedge_beta=float(hedge_beta),
        allow_flip=True,
        printlog=False,
    )

    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio,
        _name="sharpe",
        timeframe=bt.TimeFrame.Days,  # ✅ 日波動
        annualize=True,
        riskfreerate=0.0,
    )
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    strat = cerebro.run(maxcpus=1)[0]

    end_value = float(cerebro.broker.getvalue())
    total_return = end_value / float(cash) - 1.0
    sharpe = strat.analyzers.sharpe.get_analysis().get("sharperatio", np.nan)

    trades = strat.analyzers.trades.get_analysis()
    total_trades = np.nan
    try:
        total_trades = trades["total"]["total"]
    except Exception:
        pass

    return {
        "entry_z": float(entry_z),
        "exit_buffer_z": float(exit_buffer_z),
        "sharpe": float(sharpe) if sharpe is not None else np.nan,
        "return_pct": float(total_return * 100.0),
        "final_value": end_value,
        "trades": float(total_trades) if total_trades is not None else np.nan,
    }


def plot_heatmaps(res: pd.DataFrame):
    sharpe_mat = res.pivot(index="entry_z", columns="exit_buffer_z", values="sharpe").sort_index()
    ret_mat = res.pivot(index="entry_z", columns="exit_buffer_z", values="return_pct").sort_index()

    plt.figure()
    plt.imshow(sharpe_mat.values, aspect="auto", origin="lower")
    plt.title("Sharpe (daily vol, annualized)")
    plt.xlabel("exit_buffer_z")
    plt.ylabel("entry_z")
    plt.xticks(range(len(sharpe_mat.columns)), [f"{v:.2f}" for v in sharpe_mat.columns], rotation=45)
    plt.yticks(range(len(sharpe_mat.index)), [f"{v:.2f}" for v in sharpe_mat.index])
    plt.colorbar()

    plt.figure()
    plt.imshow(ret_mat.values, aspect="auto", origin="lower")
    plt.title("Total Return (%)")
    plt.xlabel("exit_buffer_z")
    plt.ylabel("entry_z")
    plt.xticks(range(len(ret_mat.columns)), [f"{v:.2f}" for v in ret_mat.columns], rotation=45)
    plt.yticks(range(len(ret_mat.index)), [f"{v:.2f}" for v in ret_mat.index])
    plt.colorbar()

    plt.tight_layout()
    plt.show()


def sweep_heatmap_mp(
    entry_grid,
    exit_grid,
    data_path="data/spread_ratio_PAXG_XAUT_1m.parquet",
    cash=100000.0,
    commission=0.02 / 100,
    lookback=1000,
    stop_z=100.0,
    leg_cash=45000.0,
    hedge_beta=1.0,
    max_workers=None,
    save_csv="grid_results.csv",
):
    jobs = []
    for ez, xb in itertools.product(entry_grid, exit_grid):
        jobs.append({
            "data_path": data_path,
            "cash": cash,
            "commission": commission,
            "lookback": lookback,
            "entry_z": float(ez),
            "exit_buffer_z": float(xb),
            "stop_z": stop_z,
            "leg_cash": leg_cash,
            "hedge_beta": hedge_beta,
        })

    rows = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(run_backtest_once_worker, j) for j in jobs]
        for fut in as_completed(futs):
            r = fut.result()
            rows.append(r)
            print(f"entry_z={r['entry_z']:.2f}, exit_buffer_z={r['exit_buffer_z']:.2f} -> Sharpe={r['sharpe']:.2f}, Return%={r['return_pct']:.2f}, Trades={r['trades']}")

    res = pd.DataFrame(rows)
    res = res.sort_values(["entry_z", "exit_buffer_z"]).reset_index(drop=True)

    if save_csv:
        res.to_csv(save_csv, index=False)
        print(f"Saved: {save_csv}")

    plot_heatmaps(res)
    return res


if __name__ == "__main__":
    entry_grid = np.arange(1.5, 3.0, 0.5)
    exit_grid = np.arange(0.0, 2.5, 0.5)

    sweep_heatmap_mp(
        entry_grid=entry_grid,
        exit_grid=exit_grid,
        data_path="data/spread_ratio_PAXG_XAUT_1m.parquet",
        cash=100000.0,
        commission=0.02/100,
        lookback=1000,
        stop_z=100.0,
        leg_cash=45000.0,
        hedge_beta=1.0,
        max_workers=16,
        save_csv="grid_results.csv",
    )
