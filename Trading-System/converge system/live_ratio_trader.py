# live_ratio_trader.py
from __future__ import annotations

import sys
import time
import math
import shutil
import datetime as dt
from dataclasses import dataclass
from collections import deque
from typing import Deque, Optional, Tuple, Dict, Literal

import pandas as pd
import ccxt

from tool import KCEXTool
from ccxt_bybit_fetcher import resolve_bybit_swap_symbol, download_ohlcv_ccxt

UTC = dt.timezone.utc
Pos = Literal["LONG_SPREAD", "SHORT_SPREAD"]


# ---------- console ----------
def overwrite_line(line: str) -> None:
    """同一行刷新顯示（避免 wrap 黏行）"""
    width = shutil.get_terminal_size((120, 20)).columns
    if width < 40:
        width = 120
    usable = max(0, width - 1)
    line = line[:usable]
    sys.stdout.write("\r" + line.ljust(usable))
    sys.stdout.flush()


def log_line(msg: str, tick_refresh: bool) -> None:
    """要印正常 log（signal/trade）前先換行，避免跟 tick 刷新行黏住"""
    if tick_refresh:
        sys.stdout.write("\n")
        sys.stdout.flush()
    print(msg, flush=True)


# ---------- math ----------
def ratio_from_prices(paxg: float, xaut: float) -> float:
    denom = paxg + xaut
    if denom <= 0:
        return float("nan")
    return (paxg - xaut) / (denom + 1e-12) * 200.0


def mid_price(last: float, bid: float, ask: float) -> float:
    if bid > 0 and ask > 0:
        return 0.5 * (bid + ask)
    if bid > 0 and last > 0:
        return 0.5 * (bid + last)
    if ask > 0 and last > 0:
        return 0.5 * (ask + last)
    return last


def safe_bid(bid: float, mid: float) -> float:
    return bid if bid > 0 else mid


def safe_ask(ask: float, mid: float) -> float:
    return ask if ask > 0 else mid


# ---------- rolling ----------
class RollingStats:
    def __init__(self, window: int):
        self.n = int(window)
        self.buf: Deque[float] = deque()
        self.sum = 0.0
        self.sumsq = 0.0

    def ready(self) -> bool:
        return len(self.buf) >= self.n

    def push(self, x: float) -> None:
        if not math.isfinite(x):
            return
        self.buf.append(x)
        self.sum += x
        self.sumsq += x * x
        if len(self.buf) > self.n:
            old = self.buf.popleft()
            self.sum -= old
            self.sumsq -= old * old

    def mean_std(self) -> Tuple[float, float]:
        if len(self.buf) == 0:
            return float("nan"), float("nan")
        m = self.sum / len(self.buf)
        v = self.sumsq / len(self.buf) - m * m
        v = max(v, 0.0)
        return m, math.sqrt(v)


class MinuteMidAggregator:
    """高頻 mid -> 每分鐘一筆 mid close（用該分鐘最後一次看到的 mid）"""

    def __init__(self):
        self.cur_min_key: Optional[int] = None
        self.last_p_mid: Optional[float] = None
        self.last_x_mid: Optional[float] = None

    def update(self, ts: float, p_mid: float, x_mid: float) -> Optional[Tuple[int, float, float]]:
        min_key = int(ts // 60)
        if self.cur_min_key is None:
            self.cur_min_key = min_key
            self.last_p_mid, self.last_x_mid = p_mid, x_mid
            return None

        if min_key == self.cur_min_key:
            self.last_p_mid, self.last_x_mid = p_mid, x_mid
            return None

        out_key = self.cur_min_key
        out_p, out_x = self.last_p_mid, self.last_x_mid

        self.cur_min_key = min_key
        self.last_p_mid, self.last_x_mid = p_mid, x_mid

        if out_p is None or out_x is None:
            return None
        return out_key, float(out_p), float(out_x)


# ---------- config ----------
@dataclass
class Config:
    # warmup
    lookback: int = 1000
    extra: int = 300
    timeframe: str = "1m"
    tz: str = "America/New_York"

    # strategy thresholds (tick z-score, exec)
    entry_z: float = 2.0
    exit_buffer_z: float = 1.5
    stop_z: float = 5.0

    # order
    leg_usdt: float = 30.0
    order_type: str = "market"
    margin_mode: str = "Cross"
    leverage: str = "20"
    take_screenshot: bool = True

    # live
    poll_sec: float = 1.0
    auth_path: str = "auth.json"
    headless: bool = False
    show_tick_refresh: bool = True

    # 防止連續觸發
    min_trade_interval_sec: float = 3.0


# ---------- warmup ----------
def warmup_bybit_ratio_close(cfg: Config) -> list[float]:
    now = dt.datetime.now(UTC).replace(second=0, microsecond=0)
    minutes = cfg.lookback + cfg.extra
    since_dt = now - dt.timedelta(minutes=minutes)

    since = since_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    until = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    exchange = ccxt.bybit({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })

    paxg_symbol = resolve_bybit_swap_symbol(exchange, "PAXG", "USDT")
    xaut_symbol = resolve_bybit_swap_symbol(exchange, "XAUT", "USDT")
    print(f"[warmup] Resolved: {paxg_symbol} / {xaut_symbol}")

    paxg_df, _, _ = download_ohlcv_ccxt(
        symbol=paxg_symbol,
        timeframe=cfg.timeframe,
        since=since,
        until=until,
        exchange=exchange,
        tz=cfg.tz,
        save_csv=False,
        save_parquet=False,
        show_pbar=True,
    )
    xaut_df, _, _ = download_ohlcv_ccxt(
        symbol=xaut_symbol,
        timeframe=cfg.timeframe,
        since=since,
        until=until,
        exchange=exchange,
        tz=cfg.tz,
        save_csv=False,
        save_parquet=False,
        show_pbar=True,
    )

    merged = (
        pd.concat(
            [
                paxg_df[["close"]].rename(columns={"close": "PAXG_close"}),
                xaut_df[["close"]].rename(columns={"close": "XAUT_close"}),
            ],
            axis=1,
        )
        .dropna(how="any")
        .sort_index()
    )

    ratio = (merged["PAXG_close"] - merged["XAUT_close"]) / (merged["PAXG_close"] + merged["XAUT_close"] + 1e-12) * 200.0
    ratio = ratio.replace([float("inf"), float("-inf")], pd.NA).dropna()

    vals = [float(x) for x in ratio.tolist() if math.isfinite(float(x))]
    print(f"[warmup] bars={len(vals)} (need >= lookback={cfg.lookback})")
    return vals


# ---------- trader ----------
class LiveMeanReversionTrader:
    def __init__(self, cfg: Config, k: KCEXTool):
        self.cfg = cfg
        self.k = k

        self.stats = RollingStats(cfg.lookback)
        self.agg = MinuteMidAggregator()

        self.pos: Optional[Pos] = None
        self.prev_diff_mid: Optional[float] = None
        self.last_trade_ts: float = 0.0

    def can_trade(self) -> bool:
        return (time.time() - self.last_trade_ts) >= self.cfg.min_trade_interval_sec

    def mark_traded(self) -> None:
        self.last_trade_ts = time.time()

    def _trade_open_long(self) -> None:
        res = self.k.two_legs_trade(
            strategy="long_paxg",
            action="open",
            order_type=self.cfg.order_type,
            amount_usdt=str(self.cfg.leg_usdt),
            margin_mode=self.cfg.margin_mode,
            leverage=self.cfg.leverage,
            take_screenshot=self.cfg.take_screenshot,
        )
        if not (res["PAXG"].ok and res["XAUT"].ok):
            raise RuntimeError(f"OPEN long_spread failed: {res}")
        self.pos = "LONG_SPREAD"
        self.mark_traded()

    def _trade_open_short(self) -> None:
        res = self.k.two_legs_trade(
            strategy="short_paxg",
            action="open",
            order_type=self.cfg.order_type,
            amount_usdt=str(self.cfg.leg_usdt),
            margin_mode=self.cfg.margin_mode,
            leverage=self.cfg.leverage,
            take_screenshot=self.cfg.take_screenshot,
        )
        if not (res["PAXG"].ok and res["XAUT"].ok):
            raise RuntimeError(f"OPEN short_spread failed: {res}")
        self.pos = "SHORT_SPREAD"
        self.mark_traded()

    def _trade_close_long(self) -> None:
        res = self.k.two_legs_trade(
            strategy="long_paxg",
            action="close",
            order_type=self.cfg.order_type,
            amount_usdt=str(self.cfg.leg_usdt),
            margin_mode=self.cfg.margin_mode,
            leverage=self.cfg.leverage,
            take_screenshot=self.cfg.take_screenshot,
        )
        if not (res["PAXG"].ok and res["XAUT"].ok):
            raise RuntimeError(f"CLOSE long_spread failed: {res}")
        self.pos = None
        self.mark_traded()

    def _trade_close_short(self) -> None:
        res = self.k.two_legs_trade(
            strategy="short_paxg",
            action="close",
            order_type=self.cfg.order_type,
            amount_usdt=str(self.cfg.leg_usdt),
            margin_mode=self.cfg.margin_mode,
            leverage=self.cfg.leverage,
            take_screenshot=self.cfg.take_screenshot,
        )
        if not (res["PAXG"].ok and res["XAUT"].ok):
            raise RuntimeError(f"CLOSE short_spread failed: {res}")
        self.pos = None
        self.mark_traded()

    def on_tick(self, ts: float, p_last: float, p_bid: float, p_ask: float, x_last: float, x_bid: float, x_ask: float) -> None:
        # --- mid (rolling) ---
        p_mid = mid_price(p_last, p_bid, p_ask)
        x_mid = mid_price(x_last, x_bid, x_ask)

        # --- update rolling per minute (mid close) ---
        out = self.agg.update(ts, p_mid, x_mid)
        if out is not None:
            min_key, p_close_mid, x_close_mid = out
            ratio_close_mid = ratio_from_prices(p_close_mid, x_close_mid)
            self.stats.push(ratio_close_mid)

            mean, std = self.stats.mean_std()
            z_close_mid = (ratio_close_mid - mean) / (std + 1e-12)

            bar_time_utc = dt.datetime.fromtimestamp(min_key * 60, tz=UTC).strftime("%Y-%m-%d %H:%M:%S")
            log_line(
                f"[1m close(mid)] {bar_time_utc}Z Pmid={p_close_mid:.2f} Xmid={x_close_mid:.2f} "
                f"rmid={ratio_close_mid:+.6f} m={mean:+.6f} s={std:.6f} Zmid={z_close_mid:+.3f}",
                tick_refresh=self.cfg.show_tick_refresh,
            )

        if not self.stats.ready():
            return

        mean, std = self.stats.mean_std()
        if not (math.isfinite(mean) and math.isfinite(std) and std > 1e-12):
            return

        # --- cross (use mid vs mean) ---
        ratio_mid = ratio_from_prices(p_mid, x_mid)
        diff_mid = ratio_mid - mean
        cross_up = False
        cross_down = False
        if self.prev_diff_mid is not None:
            cross_up = (self.prev_diff_mid <= 0.0 and diff_mid > 0.0)
            cross_down = (self.prev_diff_mid >= 0.0 and diff_mid < 0.0)
        self.prev_diff_mid = diff_mid

        # --- exec prices ---
        p_sell = safe_bid(p_bid, p_mid)  # sell paxg
        p_buy = safe_ask(p_ask, p_mid)   # buy paxg
        x_sell = safe_bid(x_bid, x_mid)  # sell xaut
        x_buy = safe_ask(x_ask, x_mid)   # buy xaut

        # L side: BUY PAXG@ask, SELL XAUT@bid  (open long / close short)
        ratio_exec_L = ratio_from_prices(p_buy, x_sell)
        z_exec_L = (ratio_exec_L - mean) / (std + 1e-12)

        # S side: SELL PAXG@bid, BUY XAUT@ask  (open short / close long)
        ratio_exec_S = ratio_from_prices(p_sell, x_buy)
        z_exec_S = (ratio_exec_S - mean) / (std + 1e-12)

        # --- tick display (concise) ---
        now_utc = dt.datetime.fromtimestamp(ts, tz=UTC).strftime("%H:%M:%S")
        tick_line = (
            f"[tick {now_utc}Z] pos={self.pos or '-'} "
            f"ZL={z_exec_L:+.3f} ZS={z_exec_S:+.3f} "
            f"m={mean:+.6f} s={std:.6f}"
        )
        if self.cfg.show_tick_refresh:
            overwrite_line(tick_line)
        else:
            print(tick_line, flush=True)

        # --- strategy core (use tick z-score, exec) ---
        if not self.can_trade():
            return

        # STOP: 用「關倉那一側」的 z
        if self.pos == "LONG_SPREAD" and abs(z_exec_S) >= self.cfg.stop_z:
            log_line(f"[signal] STOP long |ZS|={abs(z_exec_S):.3f} >= {self.cfg.stop_z} -> close", self.cfg.show_tick_refresh)
            self._trade_close_long()
            return

        if self.pos == "SHORT_SPREAD" and abs(z_exec_L) >= self.cfg.stop_z:
            log_line(f"[signal] STOP short |ZL|={abs(z_exec_L):.3f} >= {self.cfg.stop_z} -> close", self.cfg.show_tick_refresh)
            self._trade_close_short()
            return

        # EXIT: cross + buffer（同樣用關倉那側 z）
        if self.pos == "LONG_SPREAD":
            if cross_up and z_exec_S >= self.cfg.exit_buffer_z:
                log_line(f"[signal] EXIT long cross_up & ZS={z_exec_S:+.3f} >= {self.cfg.exit_buffer_z} -> close", self.cfg.show_tick_refresh)
                self._trade_close_long()
                return

        if self.pos == "SHORT_SPREAD":
            if cross_down and z_exec_L <= -self.cfg.exit_buffer_z:
                log_line(f"[signal] EXIT short cross_down & ZL={z_exec_L:+.3f} <= {-self.cfg.exit_buffer_z} -> close", self.cfg.show_tick_refresh)
                self._trade_close_short()
                return

        # ENTRY: 只在無倉位時進場
        if self.pos is None:
            if z_exec_S >= self.cfg.entry_z:
                log_line(f"[signal] ENTRY short ZS={z_exec_S:+.3f} >= {self.cfg.entry_z} -> open short_spread", self.cfg.show_tick_refresh)
                self._trade_open_short()
                return

            if z_exec_L <= -self.cfg.entry_z:
                log_line(f"[signal] ENTRY long ZL={z_exec_L:+.3f} <= {-self.cfg.entry_z} -> open long_spread", self.cfg.show_tick_refresh)
                self._trade_open_long()
                return


def main():
    cfg = Config(
        lookback=1000,
        extra=300,
        entry_z=2.0,
        exit_buffer_z=1.5,
        stop_z=5.0,
        leg_usdt=20.0,
        leverage="10",
        margin_mode="Cross",
        poll_sec=1.0,
        auth_path="auth.json",
        headless=False,
        show_tick_refresh=True,
        min_trade_interval_sec=3.0,
    )

    # warmup rolling with bybit
    stats_seed = warmup_bybit_ratio_close(cfg)
    k = KCEXTool(cfg.auth_path, headless=cfg.headless)

    trader = LiveMeanReversionTrader(cfg, k)
    for v in stats_seed:
        trader.stats.push(v)
    if not trader.stats.ready():
        raise RuntimeError(f"warmup not enough: have={len(trader.stats.buf)} need={cfg.lookback}")

    mean, std = trader.stats.mean_std()
    print(f"[warmup] ready. rolling_len={len(trader.stats.buf)} mean={mean:.6f} std={std:.6f}")

    # start KCEX workers
    k.start(["PAXG", "XAUT"])
    print("[live] start tick loop ... (Ctrl+C to stop)")

    try:
        while True:
            snaps: Dict[str, object] = k.get_multi_snapshot(["PAXG", "XAUT"])

            p = snaps["PAXG"]
            x = snaps["XAUT"]

            trader.on_tick(
                ts=float(p.ts),
                p_last=float(p.price),
                p_bid=float(p.bid_price),
                p_ask=float(p.ask_price),
                x_last=float(x.price),
                x_bid=float(x.bid_price),
                x_ask=float(x.ask_price),
            )

            time.sleep(cfg.poll_sec)
    finally:
        k.stop()
        log_line("[live] stopped.", tick_refresh=cfg.show_tick_refresh)


if __name__ == "__main__":
    main()
