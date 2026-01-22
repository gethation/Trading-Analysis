import numpy as np
import pandas as pd
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

        self.portion_cash = 0.0
        self.tranches = int(49 * 60 // self.interval_minutes)
        self.session_bar = 0

        self.session_start_cash = {}   # {session_start_ts: cash}
        self.session_start_ts = None

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
        start_mins = 17 * 60 + 1
        return t.dayofweek == 4 and mins == start_mins

    def next(self):
        ts = self.data.index[-1]
        a = self.alpha_ma[-1]

        # --- out of session ---
        if not self._in_session(ts): return

        if self._on_cutoff_minutes(ts):
            for tr in self.trades:
                tr.tp = None
                tr.sl = None

            for o in self.orders:
                o.cancel()

            if self.position:
                self.position.close()
            return

        # --- session start ---
        if self._on_start_minutes(ts):
            self.session_bar = 0
            cash = self._broker._cash
            self.portion_cash = cash / float(self.tranches)

            self.session_start_ts = ts.normalize() + pd.Timedelta(hours=17, minutes=1)
            self.session_start_cash[self.session_start_ts] = float(cash)

        for t in self.trades:
            t.tp = float(a)

        self.session_bar += 1
        if (self.session_bar - 1) % self.interval_minutes != 0:
            return

        price = float(self.data.Close[-1])
        deviation = abs((price - a) / a)
        if deviation < self.min_dev:
            return

        qty = round(self.portion_cash / price)
        print()

        if price > a:
            self.sell(size=qty, tp=float(a))
            # print(ts, "sell", round(price, 2))
        elif price < a:
            self.buy(size=qty, tp=float(a))
            # print(ts, "buy ", round(price, 2))

        



