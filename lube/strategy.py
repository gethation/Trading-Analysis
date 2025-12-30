import backtrader as bt
from collections import deque
import pandas as pd


class LubeStrategy(bt.Strategy):
    params = dict(
        barsback=500,       # bars back to measure friction
        flevel=50,          # 0-100 friction level to stop trade
        tlevel=-10,         # pick lower than 0 ... to initiate trade (as in pine)
        range=100,          # bars back to measure lowest friction
        leverage=2.0,
        enableshort=True,
        src='close',        # 'close' / 'open' / 'high' / 'low' ...
    )

    def __init__(self):
        self.srcline = getattr(self.data, self.p.src)
        self.friction = 0.0

        self.friction_hist = deque(maxlen=self.p.range)
        self.midf_hist = deque(maxlen=6)     # 取 [5] 需要至少 6 個
        self.lowf2_hist = deque(maxlen=6)

        self.fir_hist = deque(maxlen=2)      # fir 與 fir[1]
        self.barcount = 0

    def next(self):
        self.barcount += 1

        # 最小必要的長度，避免 early bar 索引炸掉（這是唯一比較像防禦的部分）
        need = max(self.p.barsback, self.p.range, 6, 4, 21)
        if len(self.data) < need:
            return

        close0 = self.data.close[0]

        # === friction 累積（照 pine：每根 bar 都在舊 friction 上加本次增量） ===
        bb = self.p.barsback
        friction = 0.0
        for i in range(1, bb + 1):
            if self.data.high[-i] >= close0 and self.data.low[-i] <= close0:
                friction += (1.0 + bb) / (i + bb)

        self.friction = friction


        # === lowf/highf over range ===
        self.friction_hist.append(self.friction)
        lowf = min(self.friction_hist)
        highf = max(self.friction_hist)

        fl = self.p.flevel / 100.0
        tl = self.p.tlevel / 100.0

        midf = (lowf * (1.0 - fl) + highf * fl)
        lowf2 = (lowf * (1.0 - tl) + highf * tl)

        self.midf_hist.append(midf)
        self.lowf2_hist.append(lowf2)

        # 取 midf[5], lowf2[5]
        midf_5 = self.midf_hist[0]
        lowf2_5 = self.lowf2_hist[0]

        # === FIR filter on src ===
        src0 = self.srcline[0]
        src1 = self.srcline[-1]
        src2 = self.srcline[-2]
        src3 = self.srcline[-3]
        fir = (4.0 * src0 + 3.0 * src1 + 2.0 * src2 + 1.0 * src3) / 10.0

        prev_fir = self.fir_hist[-1] if self.fir_hist else fir
        self.fir_hist.append(fir)

        trend = 1 if fir > prev_fir else -1

        # === signals ===
        long_sig = (self.friction < lowf2_5) and (trend == 1)
        short_sig = (self.friction < lowf2_5) and (trend == -1)
        end_sig = (self.friction > midf_5)

        # === contracts size (照 pine) ===
        equity = self.broker.getvalue()
        contracts = (equity / close0) * self.p.leverage
        if contracts < 0.000001:
            contracts = 0.000001
        if contracts > 1_000_000_000:
            contracts = 1_000_000_000

        # === order logic（先平倉，再開倉，貼近 pine 的 close/entry 規則） ===
        if self.position.size > 0:
            if short_sig or end_sig:
                self.close()

        if self.position.size < 0:
            if long_sig or end_sig:
                self.close()

        if self.barcount > 20:
            if self.position.size == 0:
                if long_sig:
                    self.buy(size=contracts)
                elif self.p.enableshort and short_sig:
                    self.sell(size=contracts)
