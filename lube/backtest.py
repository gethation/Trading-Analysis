from strategy import LubeStrategy
import pandas as pd
import backtrader as bt


def run_backtest(parquet_path: str, cash=1000000, commission=0.0004):
    df = pd.read_parquet(parquet_path, engine="pyarrow")

    data = bt.feeds.PandasData(
        dataname=df,
        datetime=None,     # index as datetime
        open='open',
        high='high',
        low='low',
        close='close',
        volume='volume',
        openinterest=None
    )

    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(LubeStrategy)

    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)  # 很簡單的手續費（可改）

    # ---- analyzers（報告核心）----
    cerebro.addanalyzer(bt.analyzers.Returns,     _name='returns')      # rtot, ravg, rnorm, rnorm100
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown,    _name='dd')           # max drawdown
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')     # 交易統計
    cerebro.addanalyzer(bt.analyzers.SQN,         _name='sqn')          # 系統品質分數
    cerebro.addanalyzer(bt.analyzers.TimeReturn,  _name='timereturn')   # 每期報酬序列（拿去畫圖/做更完整報告）

    start_value = cerebro.broker.getvalue()
    results = cerebro.run()
    strat = results[0]
    end_value = cerebro.broker.getvalue()

    # ---- 取結果 ----
    rets = strat.analyzers.returns.get_analysis()
    sharpe = strat.analyzers.sharpe.get_analysis()
    dd = strat.analyzers.dd.get_analysis()
    trades = strat.analyzers.trades.get_analysis()
    sqn = strat.analyzers.sqn.get_analysis()

    # ---- 印回測報告 ----
    print("\n========== Backtest Report ==========")
    print(f"Start Value: {start_value:.2f}")
    print(f"End Value:   {end_value:.2f}")
    print(f"Net PnL:     {end_value - start_value:.2f}")
    print(f"Total Return: {((end_value / start_value) - 1) * 100:.2f}%")

    # Returns analyzer
    # rnorm100 = 年化報酬（百分比）
    if 'rnorm100' in rets:
        print(f"Annual Return (Returns analyzer): {rets['rnorm100']:.2f}%")

    # Sharpe
    # 有些資料/期間可能出現 None（例如報酬方差太小），這裡直接印
    print(f"Sharpe (daily, annualized): {sharpe.get('sharperatio')}")

    # Drawdown
    print(f"Max Drawdown: {dd['max']['drawdown']:.2f}%")
    print(f"Max Drawdown Len: {dd['max']['len']} bars")

    # Trades（這個 dict 很大，挑常用的印）
    total_closed = trades.get('total', {}).get('closed', 0)
    won_total = trades.get('won', {}).get('total', 0)
    lost_total = trades.get('lost', {}).get('total', 0)

    pnl_net_total = trades.get('pnl', {}).get('net', {}).get('total', None)
    pnl_net_avg = trades.get('pnl', {}).get('net', {}).get('average', None)

    print("\n--- Trades ---")
    print(f"Closed Trades: {total_closed}")
    print(f"Won / Lost:    {won_total} / {lost_total}")
    print(f"Net PnL (all trades): {pnl_net_total}")
    print(f"Avg Net PnL per trade: {pnl_net_avg}")

    print("\n--- SQN ---")
    print(f"SQN: {sqn.get('sqn')}")

    print("====================================\n")

    # 你如果想要畫圖（可選）
    tr = strat.analyzers.timereturn.get_analysis()   # {datetime: return}
    ret = pd.Series(tr).sort_index().fillna(0.0)

    equity = cash * (1.0 + ret).cumprod()
    equity.name = "equity"

    ax = equity.plot(title="Equity Curve")  # 這行會用 matplotlib
    fig = ax.get_figure()
    fig.savefig(r"Lube/backtest_result/equity_curve.png", dpi=200, bbox_inches="tight")

    equity.to_csv(r"Lube/backtest_result/equity_curve.csv")
    print("Saved: equity_curve.png, equity_curve.csv")

    cerebro.plot()

    return strat


if __name__ == "__main__":
    run_backtest(r"data\BTC_30m.parquet")
