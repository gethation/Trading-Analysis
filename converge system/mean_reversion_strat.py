# bt_pairs_paxg_xaut.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import backtrader as bt


class PxPandasData(bt.feeds.PandasData):
    """
    讓 PandasData 可以把欄位名稱（例如 PAXG_open）映射到 Backtrader 的 OHLC。
    """
    params = (
        ("datetime", None),  # index is datetime
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", -1),
        ("openinterest", -1),
    )


def load_merged_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, index_col=0)

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    need = [
        "PAXG_open", "PAXG_high", "PAXG_low", "PAXG_close",
        "XAUT_open", "XAUT_high", "XAUT_low", "XAUT_close",
    ]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Existing: {list(df.columns)}")

    df = df.dropna(subset=need, how="any")
    return df


class PairsMeanReversion(bt.Strategy):
    params = dict(
        lookback=1000,
        entry_z=2.0,

        exit_buffer_z=1.5,

        stop_z=5.0,

        leg_cash=1000.0,
        hedge_beta=1.0,

        allow_flip=True,
        printlog=False,
    )

    def log(self, txt: str):
        if self.p.printlog:
            dt = self.data0.datetime.datetime(0)
            print(f"{dt.isoformat()} {txt}")

    def __init__(self):
        # data0 = PAXG, data1 = XAUT
        self.paxg = self.data0
        self.xaut = self.data1

        # ratio_close = (PAXG - XAUT) / (PAXG + XAUT) * 200
        denom = (self.paxg.close + self.xaut.close)
        self.ratio = (self.paxg.close - self.xaut.close) / (denom + 1e-12) * 200.0

        self.mean = bt.ind.SMA(self.ratio, period=self.p.lookback)
        self.std = bt.ind.StdDev(self.ratio, period=self.p.lookback)
        self.z = (self.ratio - self.mean) / (self.std + 1e-12)

        self.cross = bt.ind.CrossOver(self.ratio, self.mean)

        self.ord_paxg = None
        self.ord_xaut = None

    def _orders_pending(self) -> bool:
        return (self.ord_paxg is not None) or (self.ord_xaut is not None)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            side = "BUY" if order.isbuy() else "SELL"
            self.log(
                f"{order.data._name} {side} EXECUTED "
                f"price={order.executed.price:.6f} size={order.executed.size:.6f}"
            )
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"{order.data._name} Order Canceled/Margin/Rejected")

        if order.data is self.paxg:
            self.ord_paxg = None
        elif order.data is self.xaut:
            self.ord_xaut = None

    def _target_sizes(self) -> tuple[float, float]:
        paxg_px = float(self.paxg.close[0])
        xaut_px = float(self.xaut.close[0])
        if paxg_px <= 0 or xaut_px <= 0:
            return 0.0, 0.0

        paxg_size = self.p.leg_cash / paxg_px
        xaut_size = (self.p.leg_cash / xaut_px) * float(self.p.hedge_beta)
        return paxg_size, xaut_size

    def _close_both(self):
        if self.getposition(self.paxg).size != 0:
            self.ord_paxg = self.close(data=self.paxg)
        if self.getposition(self.xaut).size != 0:
            self.ord_xaut = self.close(data=self.xaut)

    def next(self):
        if len(self.paxg) < self.p.lookback:
            return

        if self._orders_pending():
            return

        z = float(self.z[0])

        pos_p = self.getposition(self.paxg).size
        pos_x = self.getposition(self.xaut).size
        in_market = (pos_p != 0) or (pos_x != 0)

        # --- 停損：z 擴散太大 ---
        if in_market and abs(z) >= self.p.stop_z:
            self.log(f"STOP: |z|={z:.3f} >= {self.p.stop_z}, close both")
            self._close_both()
            return

        if in_market:
            is_long_spread = (pos_p > 0 and pos_x < 0)   # BUY PAXG, SELL XAUT
            is_short_spread = (pos_p < 0 and pos_x > 0)  # SELL PAXG, BUY XAUT
            buf = float(self.p.exit_buffer_z)

            if is_long_spread:
                # ratio 上穿均線（cross>0）且 z >= +buf 才出
                if self.cross[0] > 0 and z >= buf:
                    self.log(f"EXIT LONG (cross+buffer): z={z:.3f} >= {buf}")
                    self._close_both()
                    return

            if is_short_spread:
                # ratio 下穿均線（cross<0）且 z <= -buf 才出
                if self.cross[0] < 0 and z <= -buf:
                    self.log(f"EXIT SHORT (cross+buffer): z={z:.3f} <= {-buf}")
                    self._close_both()
                    return

        # --- 進場 ---
        if not in_market:
            paxg_size, xaut_size = self._target_sizes()
            if paxg_size == 0 or xaut_size == 0:
                return

            if z >= self.p.entry_z:
                self.log(f"ENTRY SHORT SPREAD: z={z:.3f} (SELL PAXG, BUY XAUT)")
                self.ord_paxg = self.sell(data=self.paxg, size=paxg_size)
                self.ord_xaut = self.buy(data=self.xaut, size=xaut_size)
                return

            if z <= -self.p.entry_z:
                self.log(f"ENTRY LONG SPREAD: z={z:.3f} (BUY PAXG, SELL XAUT)")
                self.ord_paxg = self.buy(data=self.paxg, size=paxg_size)
                self.ord_xaut = self.sell(data=self.xaut, size=xaut_size)
                return

            return

        # --- 反向翻倉（可選）---
        if self.p.allow_flip:
            if pos_p > 0 and pos_x < 0 and z >= self.p.entry_z:
                self.log(f"FLIP requested -> close (z={z:.3f})")
                self._close_both()
                return

            if pos_p < 0 and pos_x > 0 and z <= -self.p.entry_z:
                self.log(f"FLIP requested -> close (z={z:.3f})")
                self._close_both()
                return