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
    tranche = 100

    def init(self):
        idx = self.data.index  # DatetimeIndex

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

        self.potion_cash = 0.0

    def next(self):
        a = self.alpha_ma[-1]
        a_prev = self.alpha_ma[-2] if len(self.alpha_ma) > 1 else np.nan

        # --- out of session ---
        if np.isnan(a):
            if not np.isnan(a_prev) and self.position:
                self.position.close()
            return

        # --- session start ---
        if np.isnan(a_prev):
            cash = self._broker._cash
            self.potion_cash = cash / float(self.tranche)
            self.tranche_left = self.tranche

        price = float(self.data.Close[-1])
        deviation = abs((price - a) / a)

        if deviation < self.min_dev:
            return
        
        qty = self.potion_cash / price

        if price > a:
            self.sell(size=qty)
            self.tranche_left -= 1
        elif price < a:
            self.buy(size=qty)
            self.tranche_left -= 1

        



