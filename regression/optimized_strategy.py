import numpy as np
import pandas as pd
from collections import deque
from backtesting import Strategy


def compute_alpha_ma(index, open_, close, window: int = 500, alpha: float = 0.7, cutoff_m: int = 3):
    """
    完整復刻你專案 regression/deviation_bin_probability.py 的 add_alpha_ma 邏輯，
    只是改成回傳 numpy array，方便給 backtesting 的 self.I 使用。
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1]")

    idx = pd.DatetimeIndex(index)
    open_ = np.asarray(open_, dtype=float)
    close = np.asarray(close, dtype=float)

    dow = idx.dayofweek.to_numpy()
    minutes = (idx.hour.to_numpy() * 60 + idx.minute.to_numpy())
    cutoff_min = 18 * 60 + cutoff_m

    calc_mask = (
        ((dow == 4) & (minutes >= 17 * 60)) |   # Fri >= 17:00
        (dow == 5) |                             # Sat
        ((dow == 6) & (minutes < cutoff_min))    # Sun < 18:00 + cutoff_m
    )

    out = np.full(len(idx), np.nan, dtype=float)
    if not calc_mask.any():
        return out

    pos = np.flatnonzero(calc_mask)
    sidx = idx[pos]
    s_open = open_[pos]
    s_close = close[pos]

    # 每筆映射到該週末 session 的 Fri 17:00
    s_dow = sidx.dayofweek.to_numpy()
    delta_days = s_dow - 4
    delta_days = np.where(delta_days < 0, 0, delta_days)
    ss = (sidx.normalize() + pd.Timedelta(hours=17) - pd.to_timedelta(delta_days, unit="D")).to_numpy()

    # 逐 session 計算
    for ss_val in np.unique(ss):
        m = (ss == ss_val)
        gidx = sidx[m]
        gopen = s_open[m]
        gclose = s_close[m]

        # ref_open：Fri 17:00 那根 Open（缺就用第一根）
        is_fri17 = (gidx.dayofweek == 4) & (gidx.hour == 17) & (gidx.minute == 0)
        ref_open = float(gopen[np.argmax(is_fri17)]) if is_fri17.any() else float(gopen[0])

        # padded MA：rolling sum + padding（用 cumsum 加速）
        csum = np.cumsum(gclose)
        n = np.arange(1, len(gclose) + 1, dtype=int)
        roll_sum = np.empty_like(gclose)

        if len(gclose) <= window:
            roll_sum[:] = csum
            eff_n = n
        else:
            roll_sum[:window] = csum[:window]
            roll_sum[window:] = csum[window:] - csum[:-window]
            eff_n = np.minimum(n, window)

        ma_padded = (roll_sum + (window - eff_n) * ref_open) / window
        alpha_ma = alpha * ma_padded + (1 - alpha) * ref_open

        out[pos[m]] = alpha_ma

    return out


class DCA_Strategy(Strategy):
    window = 500
    alpha = 0.5
    cutoff_m = 5

    min_dev = 0.25 / 100
    interval_minutes = 10
    open_time_proportion = 0.25

    def init(self):
        idx = self.data.index

        def alpha_ma_indicator(open_, close, window, alpha, cutoff_m):
            return compute_alpha_ma(idx, open_, close, window, alpha, cutoff_m)

        self.alpha_ma = self.I(
            alpha_ma_indicator,
            self.data.Open,
            self.data.Close,
            self.window,
            self.alpha,
            self.cutoff_m
        )

        self.tranches = int(49 * 60 // self.interval_minutes)  # 49 小時，假設 1 分 bar
        self.session_bar = 0
        self.accumulation_cash = 0.0
        self.portion_cash_q = deque()

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
        cutoff_mins = 18 * 60 + self.cutoff_m - 2  # 保留你原本 -2 行為
        return t.dayofweek == 6 and mins == cutoff_mins

    def _on_start_minutes(self, t: pd.Timestamp) -> bool:
        mins = t.hour * 60 + t.minute
        start_mins = 17 * 60 + 1  # 保留你原本 17:01
        return t.dayofweek == 4 and mins == start_mins

    def _minutes_since_fri_17(self, ts: pd.Timestamp) -> int:
        # ts.dayofweek: Mon=0 ... Sun=6
        delta_days = ts.dayofweek - 4  # Fri=4
        if delta_days < 0:
            delta_days = 0

        session_start = ts.normalize() + pd.Timedelta(hours=17) - pd.Timedelta(days=delta_days)
        return int((ts - session_start) / pd.Timedelta(minutes=1))

    def _time_factor(self, passed_mins: int) -> float:
        if passed_mins / (49 * 60) > self.open_time_proportion:
            return 0.0
        else:
            return 1.0

    def _dev_factor(self, deviation: float) -> float:
        return max(min(deviation * 100 -0.1, 1.0), 0)

    def next(self):
        ts = self.data.index[-1]
        a = self.alpha_ma[-1]
        passed_mins = self._minutes_since_fri_17(ts)

        # --- out of session ---
        if not self._in_session(ts):
            return

        # --- cutoff: 撤單、關倉 ---
        if self._on_cutoff_minutes(ts):
            for tr in self.trades:
                tr.tp = None
                tr.sl = None

            for o in self.orders:
                o.cancel()

            if self.position:
                self.position.close()
            return

        # --- session start: 初始化 tranche 分配 ---
        if self._on_start_minutes(ts):
            self.session_bar = 0
            self.accumulation_cash = 0.0

            cash = float(self._broker._cash)

            n_open = int(self.tranches * self.open_time_proportion)
            portion_cash = cash / float(n_open)

            self.portion_cash_q = deque(
                [portion_cash] * (n_open + 1) + [0.0] * (self.tranches - n_open)
            )

        for tr in self.trades:
            tr.tp = float(a)

        self.session_bar += 1
        if (self.session_bar - 1) % self.interval_minutes != 0:
            return

        self.accumulation_cash += float(self.portion_cash_q.popleft())

        price = float(self.data.Close[-1])
        deviation = abs((price - a) / a)
        if deviation < self.min_dev:
            return

        used_potion = self._dev_factor(deviation) * self._time_factor(passed_mins)

        qty = int(self.accumulation_cash * used_potion / price)
        if qty == 0:
            return

        self.accumulation_cash -= qty * price

        if price > a:
            self.sell(size=qty, tp=float(a))
        else:
            self.buy(size=qty, tp=float(a))
