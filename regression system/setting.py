# settings.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict

import yaml
from dotenv import load_dotenv


@dataclass
class Settings:
    # exchange
    exchange: str
    market_type: str
    symbol: str
    timeframe: str
    tz: str

    quote: str
    poll_seconds: int
    max_bars: int

    # strategy
    window: int
    alpha: float
    cutoff_m: int
    min_dev: float
    interval_minutes: int
    open_time_proportion: float

    # risk
    max_notional_per_order: float
    disable_trading: bool

    # secrets
    api_key: str
    api_secret: str


def load_settings(config_path: str = "config.yaml") -> Settings:
    load_dotenv()  # 讀 .env 進環境變數

    with open(config_path, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    s = cfg.get("strategy", {})
    r = cfg.get("risk", {})

    # 你可以依交易所換成不同 env key 名稱
    api_key = os.environ.get("BINANCE_API_KEY", "")
    api_secret = os.environ.get("BINANCE_API_SECRET", "")

    return Settings(
        exchange=str(cfg.get("exchange", "binance")),
        market_type=str(cfg.get("market_type", "future")),
        symbol=str(cfg.get("symbol", "PAXG/USDT")),
        timeframe=str(cfg.get("timeframe", "1m")),
        tz=str(cfg.get("tz", "America/New_York")),
        quote=str(cfg.get("quote", "USDT")),
        poll_seconds=int(cfg.get("poll_seconds", 10)),
        max_bars=int(cfg.get("max_bars", 3000)),
        window=int(s.get("window", 1000)),
        alpha=float(s.get("alpha", 0.5)),
        cutoff_m=int(s.get("cutoff_m", 5)),
        min_dev=float(s.get("min_dev", 0.0015)),
        interval_minutes=int(s.get("interval_minutes", 10)),
        open_time_proportion=float(s.get("open_time_proportion", 0.5)),
        max_notional_per_order=float(r.get("max_notional_per_order", 0.0)),
        disable_trading=bool(r.get("disable_trading", False)),
        api_key=api_key,
        api_secret=api_secret,
    )
