# live_ratio_feed.py
from __future__ import annotations

import sys
import time
import math
import shutil
import datetime as dt
import json
import threading
import webbrowser
from dataclasses import dataclass
from collections import deque
from typing import Deque, Optional, Tuple, Dict, Any
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import pandas as pd
import ccxt

from tool import KCEXTool
from ccxt_bybit_fetcher import resolve_bybit_swap_symbol, download_ohlcv_ccxt

UTC = dt.timezone.utc


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
    """印 log 前先換行，避免跟 tick 刷新行黏住"""
    if tick_refresh:
        sys.stdout.write("\n")
        sys.stdout.flush()
    print(msg, flush=True)


# ---------- math ----------
def ratio_from_prices(paxg: float, xaut: float) -> float:
    """(PAXG - XAUT) / (PAXG + XAUT) * 200"""
    denom = paxg + xaut
    if denom <= 0:
        return float("nan")
    return (paxg - xaut) / (denom + 1e-12) * 200.0


def mid_price(last: float, bid: float, ask: float) -> float:
    """優先用 bid/ask 算 mid；缺資料就退回 last（或與 last 折衷）"""
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


def _ts_to_unix_seconds(ts: pd.Timestamp | dt.datetime) -> int:
    if isinstance(ts, pd.Timestamp):
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return int(ts.value // 10**9)
    # datetime
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return int(ts.timestamp())


# ---------- rolling ----------
class RollingStats:
    """固定長度 rolling mean/std（O(1) 更新），並支援一次性校正"""

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

    def calibrate_to_once(self, target_mean: float, target_std: float, eps: float = 1e-12) -> None:
        """
        一次性校正：把當前 window 的 mean/std 對齊到 target_mean/target_std
        x' = a x + b
        a = target_std / cur_std
        b = target_mean - a * cur_mean
        """
        if len(self.buf) == 0:
            return
        if not (math.isfinite(target_mean) and math.isfinite(target_std)) or target_std <= 0:
            return

        cur_mean, cur_std = self.mean_std()
        if not (math.isfinite(cur_mean) and math.isfinite(cur_std)) or cur_std <= eps:
            return

        a = target_std / cur_std
        b = target_mean - a * cur_mean

        new_buf = deque((a * x + b) for x in self.buf)
        self.buf = new_buf
        self.sum = sum(self.buf)
        self.sumsq = sum(x * x for x in self.buf)


# ---------- live candle builder ----------
class CandleBuilder:
    """每秒 ratio_mid -> 1m candle（time = minute_start_unix_seconds）"""

    def __init__(self):
        self.cur_min_key: Optional[int] = None  # minute key = int(ts//60)
        self.cur_candle: Optional[dict[str, float | int]] = None

    def update(self, ts: float, price: float) -> Tuple[Optional[dict[str, float | int]], dict[str, float | int]]:
        """
        Return (closed_candle_or_None, current_candle).
        - current candle is always returned (updated to this tick).
        """
        min_key = int(ts // 60)

        if self.cur_min_key is None:
            self.cur_min_key = min_key
            t0 = min_key * 60
            self.cur_candle = {"time": int(t0), "open": price, "high": price, "low": price, "close": price}
            return None, self.cur_candle

        if min_key == self.cur_min_key:
            c = self.cur_candle
            assert c is not None
            c["high"] = float(max(float(c["high"]), price))
            c["low"] = float(min(float(c["low"]), price))
            c["close"] = float(price)
            return None, c

        # minute rolled
        closed = self.cur_candle
        self.cur_min_key = min_key
        t0 = min_key * 60
        self.cur_candle = {"time": int(t0), "open": price, "high": price, "low": price, "close": price}
        return closed, self.cur_candle


# ---------- web ui ----------
@dataclass
class ChartState:
    lock: threading.Lock
    candles: Deque[dict]
    ma: Deque[dict]
    upper: Deque[dict]
    lower: Deque[dict]
    current_candle: Optional[dict]
    last_header: dict[str, Any]
    version: int = 0

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            candles = list(self.candles)
            if self.current_candle is not None:
                candles = candles + [self.current_candle]
            return {
                "version": self.version,
                "candles": candles,
                "ma": list(self.ma),
                "upper": list(self.upper),
                "lower": list(self.lower),
                "header": dict(self.last_header),
            }

    def poll(self, since: int) -> dict[str, Any]:
        with self.lock:
            if since >= self.version:
                return {"version": self.version}
            payload: dict[str, Any] = {
                "version": self.version,
                "header": dict(self.last_header),
            }
            if self.current_candle is not None:
                payload["candle"] = dict(self.current_candle)
            if len(self.ma) > 0:
                payload["ma"] = self.ma[-1]
            if len(self.upper) > 0:
                payload["upper"] = self.upper[-1]
            if len(self.lower) > 0:
                payload["lower"] = self.lower[-1]
            return payload


def _make_live_html(title: str, poll_ms: int = 1000) -> str:
    # NOTE: keep JS self-contained; fetch JSON from /snapshot and /poll
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>{title}</title>
  <style>
    html, body {{
      height: 100%;
      margin: 0;
      background: #0b0f14;
      color: #e6edf3;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, "Noto Sans", "Apple Color Emoji","Segoe UI Emoji";
    }}
    .wrap {{
      height: 100%;
      display: flex;
      flex-direction: column;
    }}
    .header {{
      padding: 10px 14px;
      font-size: 13px;
      opacity: 0.95;
      border-bottom: 1px solid rgba(255,255,255,0.08);
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      align-items: center;
    }}
    .kv {{
      display: inline-flex;
      gap: 6px;
      align-items: baseline;
      white-space: nowrap;
    }}
    .k {{ opacity: 0.7; }}
    #chart {{
      flex: 1;
      position: relative;
    }}
    a {{ color: inherit; }}
  </style>
</head>
<body>
<div class="wrap">
  <div class="header">
    <div style="font-weight:600;">{title}</div>
    <div class="kv"><span class="k">ratio</span><span id="h_ratio">--</span></div>
    <div class="kv"><span class="k">ma</span><span id="h_ma">--</span></div>
    <div class="kv"><span class="k">std</span><span id="h_std">--</span></div>
    <div class="kv"><span class="k">upper</span><span id="h_upper">--</span></div>
    <div class="kv"><span class="k">lower</span><span id="h_lower">--</span></div>
    <div class="kv"><span class="k">z</span><span id="h_z">--</span></div>
    <div class="kv"><span class="k">time</span><span id="h_time">--</span></div>
  </div>
  <div id="chart"></div>
</div>

<script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
<script>
  const container = document.getElementById('chart');

  const chart = LightweightCharts.createChart(container, {{
    layout: {{
      background: {{ type: 'solid', color: '#0b0f14' }},
      textColor: '#d0d7de',
    }},
    grid: {{
      vertLines: {{ color: 'rgba(255,255,255,0.06)' }},
      horzLines: {{ color: 'rgba(255,255,255,0.06)' }},
    }},
    rightPriceScale: {{
      borderColor: 'rgba(255,255,255,0.12)',
    }},
    timeScale: {{
      borderColor: 'rgba(255,255,255,0.12)',
      timeVisible: true,
      secondsVisible: true,
    }},
    crosshair: {{
      mode: LightweightCharts.CrosshairMode.Normal,
    }},
    handleScroll: true,
    handleScale: true,
  }});

  const candleSeries = chart.addCandlestickSeries({{
    priceFormat: {{
      type: 'custom',
      formatter: (price) => price.toFixed(6),
    }},
  }});

  const maSeries = chart.addLineSeries({{
    lineWidth: 2,
    title: 'MA',
    priceLineVisible: false,
    lastValueVisible: true,
  }});

  const upperSeries = chart.addLineSeries({{
    lineWidth: 1,
    title: 'Upper',
    priceLineVisible: false,
    lastValueVisible: false,
    lineStyle: LightweightCharts.LineStyle.Dashed,
  }});

  const lowerSeries = chart.addLineSeries({{
    lineWidth: 1,
    title: 'Lower',
    priceLineVisible: false,
    lastValueVisible: false,
    lineStyle: LightweightCharts.LineStyle.Dashed,
  }});

  function setHeader(h) {{
    if (!h) return;
    const fmt = (x) => (x === null || x === undefined || !isFinite(x)) ? '--' : Number(x).toFixed(6);
    document.getElementById('h_ratio').textContent = fmt(h.ratio);
    document.getElementById('h_ma').textContent = fmt(h.ma);
    document.getElementById('h_std').textContent = fmt(h.std);
    document.getElementById('h_upper').textContent = fmt(h.upper);
    document.getElementById('h_lower').textContent = fmt(h.lower);
    document.getElementById('h_z').textContent = (h.z === null || h.z === undefined || !isFinite(h.z)) ? '--' : Number(h.z).toFixed(3);
    document.getElementById('h_time').textContent = h.time_str || '--';
  }}

  let version = 0;
  async function init() {{
    const resp = await fetch('/snapshot');
    const data = await resp.json();
    version = data.version || 0;
    candleSeries.setData(data.candles || []);
    maSeries.setData(data.ma || []);
    upperSeries.setData(data.upper || []);
    lowerSeries.setData(data.lower || []);
    setHeader(data.header);
    chart.timeScale().fitContent();
  }}

  async function poll() {{
    try {{
      const resp = await fetch('/poll?since=' + version);
      const data = await resp.json();
      if (data.version !== undefined) {{
        if (data.version > version) {{
          version = data.version;
          if (data.candle) candleSeries.update(data.candle);
          if (data.ma) maSeries.update(data.ma);
          if (data.upper) upperSeries.update(data.upper);
          if (data.lower) lowerSeries.update(data.lower);
          if (data.header) setHeader(data.header);
        }}
      }}
    }} catch (e) {{
      // ignore transient errors
    }}
  }}

  init().then(() => {{
    setInterval(poll, {poll_ms});
  }});

  const ro = new ResizeObserver(entries => {{
    for (const entry of entries) {{
      const {{ width, height }} = entry.contentRect;
      chart.applyOptions({{ width, height }});
    }}
  }});
  ro.observe(container);
</script>
</body>
</html>
"""


class LiveChartHandler(BaseHTTPRequestHandler):
    """Serve the live chart HTML + JSON endpoints."""

    # injected at runtime
    state: ChartState
    html: str

    def _send(self, code: int, body: bytes, content_type: str = "text/plain; charset=utf-8") -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self._send(200, self.html.encode("utf-8"), "text/html; charset=utf-8")
            return

        if path == "/snapshot":
            payload = self.state.snapshot()
            self._send(200, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")
            return

        if path == "/poll":
            qs = parse_qs(parsed.query or "")
            since = 0
            try:
                since = int(qs.get("since", ["0"])[0])
            except Exception:
                since = 0
            payload = self.state.poll(since)
            self._send(200, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")
            return

        self._send(404, b"not found")

    def log_message(self, fmt: str, *args: Any) -> None:
        # silence default http.server logs
        return


class LiveChartServer:
    def __init__(self, state: ChartState, host: str = "127.0.0.1", port: int = 8765, title: str = "Live Ratio"):
        handler_cls = LiveChartHandler
        handler_cls.state = state
        handler_cls.html = _make_live_html(title=title, poll_ms=1000)
        self.httpd = ThreadingHTTPServer((host, port), handler_cls)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.host = host
        self.port = port

    def start(self, open_browser: bool = True) -> None:
        self.thread.start()
        if open_browser:
            webbrowser.open(f"http://{self.host}:{self.port}/", new=2)

    def stop(self) -> None:
        try:
            self.httpd.shutdown()
            self.httpd.server_close()
        except Exception:
            pass


# ---------- config ----------
@dataclass
class Config:
    # warmup
    lookback: int = 1000
    extra: int = 300
    timeframe: str = "1m"
    tz: str = "America/New_York"

    # live
    poll_sec: float = 1.0
    auth_path: str = "auth.json"
    headless: bool = False
    show_tick_refresh: bool = True  # True: 同一行刷新；False: 每秒 print 一行

    # band
    band_z: float = 2.0  # upper/lower = ma ± std * band_z

    # web ui
    enable_web: bool = True
    web_port: int = 8765
    open_browser: bool = True
    chart_candles_max: int = 2000
    chart_lines_max: int = 6000  # ~ last 100 minutes if 1s points

    # one-time calibration (from TradingView / KCEX)
    # 填 None 就不校正；填數字就初始化校正一次
    kcex_mean: Optional[float] = None
    kcex_std: Optional[float] = None


# ---------- warmup ----------
def warmup_bybit_ratio_seed(cfg: Config) -> Tuple[list[float], list[dict]]:
    """
    Warmup rolling stats with Bybit 1m close ratio,
    and return a lightweight "seed candles" list for initial chart context.

    Seed candles: [{time, open, high, low, close}] where OHLC are all ratio_close.
    """
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

    vals: list[float] = []
    candles: list[dict] = []

    for ts, v in ratio.items():
        fv = float(v)
        if not math.isfinite(fv):
            continue

        # ts 可能是 str / datetime / Timestamp / numpy datetime64
        t = pd.Timestamp(ts)

        # ✅ 修正點：單一 Timestamp 的 tz_localize() 不支援 ambiguous="infer"
        # 用 ambiguous="NaT" 遇到 DST ambiguous 就跳過，nonexistent 用 shift_forward 避免爆炸
        try:
            if t.tzinfo is None:
                t = t.tz_localize(cfg.tz, ambiguous="NaT", nonexistent="shift_forward")
                if pd.isna(t):
                    # ambiguous 的時間被標成 NaT，就略過
                    continue
            # 轉成 UTC
            t = t.tz_convert("UTC")
        except Exception:
            # 保守：任何怪時間直接略過，避免 warmup 中斷
            continue

        tsec = int(t.value // 10**9)

        vals.append(fv)
        candles.append({"time": tsec, "open": fv, "high": fv, "low": fv, "close": fv})

    print(f"[warmup] bars={len(vals)} (need >= lookback={cfg.lookback})")
    return vals, candles



# ---------- main ----------
def main():
    cfg = Config(
        lookback=1000,
        extra=300,
        poll_sec=1.0,
        auth_path="auth.json",
        headless=False,
        show_tick_refresh=True,
        band_z=2.0,
        enable_web=True,
        web_port=8765,
        open_browser=True,
        # one-time calibration example:
        kcex_mean=346/1000,
        kcex_std=25.3/1000,
    )

    # 1) warmup rolling（Bybit close ≈ mid）
    stats = RollingStats(cfg.lookback)
    warm_vals, seed_candles = warmup_bybit_ratio_seed(cfg)
    for v in warm_vals:
        stats.push(v)

    if not stats.ready():
        raise RuntimeError(f"warmup not enough: have={len(stats.buf)} need={cfg.lookback}")

    m0, s0 = stats.mean_std()
    print(f"[warmup] ready. rolling_len={len(stats.buf)} mean={m0:+.6f} std={s0:.6f}")

    # 2) one-time calibration to KCEX mean/std (optional)
    if cfg.kcex_mean is not None and cfg.kcex_std is not None:
        stats.calibrate_to_once(float(cfg.kcex_mean), float(cfg.kcex_std))
        m1, s1 = stats.mean_std()
        print(f"[calib ] applied once. target_mean={cfg.kcex_mean:+.6f} target_std={cfg.kcex_std:.6f}")
        print(f"[calib ] after: mean={m1:+.6f} std={s1:.6f}")

    # 3) web chart (optional)
    lock = threading.Lock()
    state = ChartState(
        lock=lock,
        candles=deque(seed_candles[-cfg.chart_candles_max :], maxlen=cfg.chart_candles_max),
        ma=deque(maxlen=cfg.chart_lines_max),
        upper=deque(maxlen=cfg.chart_lines_max),
        lower=deque(maxlen=cfg.chart_lines_max),
        current_candle=None,
        last_header={},
        version=0,
    )

    server: Optional[LiveChartServer] = None
    if cfg.enable_web:
        server = LiveChartServer(
            state=state,
            host="127.0.0.1",
            port=cfg.web_port,
            title="Live Ratio (PAXG vs XAUT) 1m Candle + MA/Bands",
        )
        server.start(open_browser=cfg.open_browser)
        print(f"[web  ] http://127.0.0.1:{cfg.web_port}/")

    # 4) live: rolling 用 mid 的 1m close 更新；tick 顯示 exec z-score（兩方向）
    candle_builder = CandleBuilder()
    k = KCEXTool(cfg.auth_path, headless=cfg.headless)
    k.start(["PAXG", "XAUT"])
    print("[live] polling KCEX snapshots ... (Ctrl+C to stop)")

    try:
        while True:
            snaps: Dict[str, object] = k.get_multi_snapshot(["PAXG", "XAUT"])

            p = snaps["PAXG"]
            x = snaps["XAUT"]

            p_last = float(p.price)
            p_bid = float(p.bid_price)
            p_ask = float(p.ask_price)

            x_last = float(x.price)
            x_bid = float(x.bid_price)
            x_ask = float(x.ask_price)

            ts = float(p.ts)

            # mid (rolling base)
            p_mid = mid_price(p_last, p_bid, p_ask)
            x_mid = mid_price(x_last, x_bid, x_ask)
            ratio_mid = ratio_from_prices(p_mid, x_mid)

            mean, std = stats.mean_std()
            upper = mean + std * float(cfg.band_z) if math.isfinite(mean) and math.isfinite(std) else float("nan")
            lower = mean - std * float(cfg.band_z) if math.isfinite(mean) and math.isfinite(std) else float("nan")

            # exec prices (both sides)
            p_sell = safe_bid(p_bid, p_mid)   # SELL PAXG
            p_buy  = safe_ask(p_ask, p_mid)   # BUY  PAXG
            x_sell = safe_bid(x_bid, x_mid)   # SELL XAUT
            x_buy  = safe_ask(x_ask, x_mid)   # BUY XAUT

            # L side: BUY PAXG@ask, SELL XAUT@bid (open long / close short)
            ratio_exec_L = ratio_from_prices(p_buy, x_sell)
            # S side: SELL PAXG@bid, BUY XAUT@ask (open short / close long)
            ratio_exec_S = ratio_from_prices(p_sell, x_buy)

            if math.isfinite(mean) and math.isfinite(std) and std > 1e-12:
                zL = (ratio_exec_L - mean) / (std + 1e-12)
                zS = (ratio_exec_S - mean) / (std + 1e-12)
                z_mid = (ratio_mid - mean) / (std + 1e-12)
            else:
                zL = float("nan")
                zS = float("nan")
                z_mid = float("nan")

            # tick display (console)
            now_utc = dt.datetime.fromtimestamp(ts, tz=UTC).strftime("%H:%M:%S")
            tick_line = (
                f"[tick {now_utc}Z] "
                f"rm={ratio_mid:+.6f} ma={mean:+.6f} s={std:.6f} "
                f"up={upper:+.6f} lo={lower:+.6f} "
                f"Zmid={z_mid:+.3f} ZL={zL:+.3f} ZS={zS:+.3f}"
            )
            if cfg.show_tick_refresh:
                overwrite_line(tick_line)
            else:
                print(tick_line, flush=True)

            # 1m candle building (always, used by both chart and rolling update)
            closed, cur = candle_builder.update(ts, ratio_mid)

            # update chart state (1s): candle + MA/bands as line points
            if cfg.enable_web:
                tsec = int(ts)
                ma_pt = {"time": tsec, "value": float(mean)} if math.isfinite(mean) else None
                up_pt = {"time": tsec, "value": float(upper)} if math.isfinite(upper) else None
                lo_pt = {"time": tsec, "value": float(lower)} if math.isfinite(lower) else None

                with lock:
                    if closed is not None:
                        state.candles.append(closed)
                    state.current_candle = cur

                    if ma_pt is not None:
                        state.ma.append(ma_pt)
                    if up_pt is not None:
                        state.upper.append(up_pt)
                    if lo_pt is not None:
                        state.lower.append(lo_pt)

                    state.last_header = {
                        "ratio": float(ratio_mid) if math.isfinite(ratio_mid) else None,
                        "ma": float(mean) if math.isfinite(mean) else None,
                        "std": float(std) if math.isfinite(std) else None,
                        "upper": float(upper) if math.isfinite(upper) else None,
                        "lower": float(lower) if math.isfinite(lower) else None,
                        "z": float(z_mid) if math.isfinite(z_mid) else None,
                        "time_str": dt.datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d %H:%M:%S") + "Z",
                    }
                    state.version += 1

            # minute close(mid) -> update rolling
            # we still keep the original logic: update stats only once per minute (mid close)
            # NOTE: candleBuilder already uses 1m candles for display but rolling uses close(mid) too.
            # Here we compute 1m close(mid) as "last seen mid of that minute" using CandleBuilder close on rollover.
            # So we simply reuse closed candle's close as the minute close(mid).
            if closed is not None:
                ratio_close_mid = float(closed["close"])
                stats.push(ratio_close_mid)

                mean2, std2 = stats.mean_std()
                z2 = (ratio_close_mid - mean2) / (std2 + 1e-12) if std2 > 1e-12 else float("nan")

                bar_time_utc = dt.datetime.fromtimestamp(int(closed["time"]), tz=UTC).strftime("%Y-%m-%d %H:%M:%S")
                log_line(
                    f"[1m close(mid)] {bar_time_utc}Z "
                    f"rm={ratio_close_mid:+.6f} ma={mean2:+.6f} s={std2:.6f} Z={z2:+.3f}",
                    tick_refresh=cfg.show_tick_refresh,
                )

            time.sleep(cfg.poll_sec)

    finally:
        try:
            k.stop()
        except Exception:
            pass
        if server is not None:
            server.stop()
        log_line("[live] stopped.", tick_refresh=cfg.show_tick_refresh)


if __name__ == "__main__":
    main()
