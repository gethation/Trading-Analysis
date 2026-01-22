# core.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Deque, Literal, Optional
from collections import deque

import numpy as np
import pandas as pd


Side = Literal["buy", "sell"]
Kind = Literal["order", "cancel_all", "close"]


@dataclass
class Intent:
    kind: Kind
    side: Optional[Side] = None
    qty: Optional[float] = None
    type: Literal["market", "limit"] = "market"
    price: Optional[float] = None
    reduce_only: bool = False
    tag: str = ""


@dataclass
class DCAState:
    tranches: int
    session_bar: int = 0
    accumulation_cash: float = 0.0
    portion_cash_q: Deque[float] = field(default_factory=deque)
    last_bar_ts: Optional[pd.Timestamp] = None


class DCACore:
    """
    最小可用策略核心：
    - 輸入：已收盤K線 df、state、cash、position_qty
    - 輸出：Intent 列表
    """

    def __init__(
        self,
        *,
        window: int = 500,
        alpha: float = 0.5,
        cutoff_m: int = 5,
        min_dev: float = 0.25 / 100,
        interval_minutes: int = 10,
        open_time_proportion: float = 0.25,
        tz: str = "America/New_York",
    ):
        self.window = window
        self.alpha = alpha
        self.cutoff_m = cutoff_m
        self.min_dev = min_dev
        self.interval_minutes = interval_minutes
        self.open_time_proportion = open_time_proportion
        self.tz = tz

    # ----- 時間/Session 判斷（精簡版） -----
    def _in_session(self, t: pd.Timestamp) -> bool:
        dow = t.dayofweek
        mins = t.hour * 60 + t.minute
        cutoff = 18 * 60 + self.cutoff_m
        return (
            ((dow == 4) and (mins >= 17 * 60)) or
            (dow == 5) or
            ((dow == 6) and (mins < cutoff))
        )

    def _on_cutoff_minutes(self, t: pd.Timestamp) -> bool:
        mins = t.hour * 60 + t.minute
        cutoff_mins = 18 * 60 + self.cutoff_m - 2
        return t.dayofweek == 6 and mins == cutoff_mins

    def _on_start_minutes(self, t: pd.Timestamp) -> bool:
        mins = t.hour * 60 + t.minute
        return t.dayofweek == 4 and mins == 17 * 60 + 1  # 17:01

    def _minutes_since_fri_17(self, ts: pd.Timestamp) -> int:
        delta_days = ts.dayofweek - 4
        if delta_days < 0:
            delta_days = 0
        session_start = ts.normalize() + pd.Timedelta(hours=17) - pd.Timedelta(days=delta_days)
        return int((ts - session_start) / pd.Timedelta(minutes=1))

    def _time_factor(self, passed_mins: int) -> float:
        return 0.0 if (passed_mins / (49 * 60)) > self.open_time_proportion else 1.0

    def _dev_factor(self, deviation: float) -> float:
        return max(min(deviation * 100 - 0.1, 1.0), 0)

    def _compute_alpha_ma_last(self, df: pd.DataFrame) -> float:
        """
        最精簡：只算最後一根的 alpha_ma（行為未必100%等同你回測版，但架構是對的）。
        之後你可替換成你 optimized_strategy.py 的完整計算。
        """
        idx = df.index
        close = df["close"].to_numpy(dtype=float)
        open_ = df["open"].to_numpy(dtype=float)

        # 用 NY 時間判斷 session（df 可能是 naive 或 tz-aware）
        if idx[-1].tzinfo is not None:
            idx_ny = idx.tz_convert(self.tz).tz_localize(None)
        else:
            idx_ny = idx

        mask = np.array([self._in_session(x) for x in idx_ny], dtype=bool)
        if not mask.any():
            return float("nan")

        sidx = idx_ny[mask]
        sopen = open_[mask]
        sclose = close[mask]

        # ref_open：簡化成 session 第一根 open（你可改成嚴格對齊 Fri 17:00）
        ref_open = float(sopen[0])

        w = self.window
        n = len(sclose)
        tail = sclose[max(0, n - w):]
        eff_n = len(tail)

        ma_padded = (float(tail.sum()) + (w - eff_n) * ref_open) / w
        alpha_ma = self.alpha * ma_padded + (1 - self.alpha) * ref_open
        return float(alpha_ma)

    def on_bar(
        self,
        *,
        df: pd.DataFrame,        # 必須含最新「已收盤」bar
        state: DCAState,
        cash: float,             # 可用資金（quote）
        position_qty: float,     # 目前倉位（正多負空）
    ) -> list[Intent]:
        ts = df.index[-1]
        if state.last_bar_ts is not None and ts <= state.last_bar_ts:
            return []
        state.last_bar_ts = ts

        # 用 NY 時間判斷
        if ts.tzinfo is not None:
            ts_ny = ts.tz_convert(self.tz).tz_localize(None)
        else:
            ts_ny = ts

        if not self._in_session(ts_ny):
            return []

        if self._on_cutoff_minutes(ts_ny):
            intents = [Intent(kind="cancel_all", tag="cutoff_cancel")]
            if abs(position_qty) > 0:
                intents.append(Intent(kind="close", tag="cutoff_close"))
            return intents

        if self._on_start_minutes(ts_ny):
            state.session_bar = 0
            state.accumulation_cash = 0.0

            n_open = int(state.tranches * self.open_time_proportion)
            portion_cash = cash / float(max(1, n_open))
            state.portion_cash_q = deque([portion_cash] * (n_open + 1) + [0.0] * (state.tranches - n_open))

        a = self._compute_alpha_ma_last(df)
        if not np.isfinite(a):
            return []

        state.session_bar += 1
        if (state.session_bar - 1) % self.interval_minutes != 0:
            return []

        if state.portion_cash_q:
            state.accumulation_cash += float(state.portion_cash_q.popleft())

        price = float(df["close"].iloc[-1])
        deviation = abs((price - a) / a)
        if deviation < self.min_dev:
            return []

        passed_mins = self._minutes_since_fri_17(ts_ny)
        used_portion = self._dev_factor(deviation) * self._time_factor(passed_mins)

        notional = state.accumulation_cash * used_portion
        if notional <= 0:
            return []

        qty = notional / price
        if qty <= 0:
            return []

        state.accumulation_cash -= qty * price

        if price > a:
            return [Intent(kind="order", side="sell", qty=qty, type="market", reduce_only=False, tag="mean_rev_short")]
        else:
            return [Intent(kind="order", side="buy", qty=qty, type="market", reduce_only=False, tag="mean_rev_long")]
