from __future__ import annotations

import json
import webbrowser
from pathlib import Path
from typing import Literal, Optional

import pandas as pd


def load_ratio_df(
    path: Path,
    filetype: Optional[Literal["parquet", "csv"]] = None,
) -> pd.DataFrame:
    if filetype is None:
        if path.suffix.lower() == ".parquet":
            filetype = "parquet"
        elif path.suffix.lower() == ".csv":
            filetype = "csv"
        else:
            raise ValueError(f"Unknown file extension: {path.suffix}")

    if filetype == "parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, index_col=0)

    df.index = pd.to_datetime(df.index)
    return df


def _index_to_utc(idx: pd.DatetimeIndex, assume_tz: str) -> tuple[pd.DatetimeIndex, Optional[pd.Series]]:
    """
    Return (idx_utc, mask)
    - mask is only used when we had to drop ambiguous timestamps (NaT).
    """
    idx = pd.to_datetime(idx)

    if getattr(idx, "tz", None) is not None:
        return idx.tz_convert("UTC"), None

    # naive index: localize then to UTC
    try:
        idx_loc = idx.tz_localize(
            assume_tz,
            ambiguous="infer",
            nonexistent="shift_forward",
        )
        return idx_loc.tz_convert("UTC"), None
    except Exception:
        idx_loc = idx.tz_localize(
            assume_tz,
            ambiguous="NaT",
            nonexistent="shift_forward",
        )
        mask = ~pd.isna(idx_loc)
        idx_loc = idx_loc[mask]
        return idx_loc.tz_convert("UTC"), mask


def to_lwc_candles(
    df: pd.DataFrame,
    open_col: str = "ratio_open",
    high_col: str = "ratio_high",
    low_col: str = "ratio_low",
    close_col: str = "ratio_close",
    assume_tz: str = "UTC",
) -> list[dict]:
    """
    Lightweight Charts candlestick format:
    [{time: unix_seconds_utc, open: float, high: float, low: float, close: float}, ...]
    """
    need = [open_col, high_col, low_col, close_col]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Existing columns: {list(df.columns)}")

    ohlc = df[need].dropna(how="any").copy()
    ohlc = ohlc.sort_index()

    idx_utc, mask = _index_to_utc(ohlc.index, assume_tz)
    if mask is not None:
        ohlc = ohlc.loc[ohlc.index[mask]]

    times = (idx_utc.view("int64") // 10**9).astype(int)

    data: list[dict] = []
    for t, row in zip(times, ohlc.itertuples(index=False)):
        o, h, l, c = row
        data.append(
            {
                "time": int(t),
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c),
            }
        )
    return data


def add_ma_columns(
    df: pd.DataFrame,
    close_col: str = "ratio_close",
    windows: tuple[int, ...] = (20, 60),
    kind: Literal["sma", "ema"] = "sma",
) -> pd.DataFrame:
    out = df.copy()
    for w in windows:
        if kind == "sma":
            out[f"ma{w}"] = out[close_col].rolling(w, min_periods=w).mean()
        else:
            out[f"ma{w}"] = out[close_col].ewm(span=w, adjust=False, min_periods=w).mean()
    return out


def to_lwc_line(
    df: pd.DataFrame,
    col: str,
    assume_tz: str = "UTC",
) -> list[dict]:
    """
    Lightweight Charts line format:
    [{time: unix_seconds_utc, value: float}, ...]
    """
    s = df[[col]].dropna(how="any").sort_index()

    idx_utc, mask = _index_to_utc(s.index, assume_tz)
    if mask is not None:
        s = s.loc[s.index[mask]]

    times = (idx_utc.view("int64") // 10**9).astype(int)

    data: list[dict] = []
    for t, v in zip(times, s[col].to_numpy()):
        data.append({"time": int(t), "value": float(v)})
    return data


def write_html_candles(
    out_html: Path,
    candle_data: list[dict],
    ma_series: dict[str, list[dict]],
    title: str = "Ratio Candle: (PAXG - XAUT) / (PAXG + XAUT) * 200",
) -> None:
    data_json = json.dumps(candle_data, ensure_ascii=False)
    ma_json = json.dumps(ma_series, ensure_ascii=False)

    html = f"""<!doctype html>
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
      font-size: 14px;
      opacity: 0.9;
      border-bottom: 1px solid rgba(255,255,255,0.08);
    }}
    #chart {{
      flex: 1;
      position: relative;
    }}
  </style>
</head>
<body>
<div class="wrap">
  <div class="header">{title}</div>
  <div id="chart"></div>
</div>

<script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
<script>
  const candleData = {data_json};
  const maSeries = {ma_json};

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
      secondsVisible: false,
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
      formatter: (price) => price.toFixed(4) + '%',
    }},
  }});
  candleSeries.setData(candleData);

  // 0% reference line
  candleSeries.createPriceLine({{
    price: 0.208,
    color: 'rgba(255,255,255,0.9)',
    lineWidth: 1,
    lineStyle: LightweightCharts.LineStyle.Solid,
    axisLabelVisible: true,
    title: '0%',
  }});

  // MA overlays (line series)
  for (const [name, data] of Object.entries(maSeries)) {{
    const s = chart.addLineSeries({{
      lineWidth: 2,
      title: name,
      priceLineVisible: false,
      lastValueVisible: true,
    }});
    s.setData(data);
  }}

  chart.timeScale().fitContent();

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
    out_html.write_text(html, encoding="utf-8")


def main():
    # 改成你的檔案路徑
    infile = Path("data/spread_ratio_PAXG_XAUT_1m.parquet")
    df = load_ratio_df(infile)
    ma = 1000

    # 均線：用 ratio_close 算 SMA(20/60)
    df = add_ma_columns(df, close_col="ratio_close", windows=(ma,21), kind="sma")

    assume_tz = "America/New_York"  # 跟你抓資料的交易所時間一致即可（或直接用 UTC）

    candle_data = to_lwc_candles(
        df,
        open_col="ratio_open",
        high_col="ratio_high",
        low_col="ratio_low",
        close_col="ratio_close",
        assume_tz=assume_tz,
    )

    ma_series = {
        f"MA{ma}": to_lwc_line(df, f"ma{ma}", assume_tz=assume_tz),
        f"MA{9}": to_lwc_line(df, f"ma{21}", assume_tz=assume_tz),
    }

    out_html = Path("data/ratio_candle_chart.html")
    out_html.parent.mkdir(parents=True, exist_ok=True)
    write_html_candles(out_html, candle_data, ma_series)

    abs_path = out_html.resolve()
    print(f"OK: wrote {abs_path}")

    # Auto open in default browser
    webbrowser.open(abs_path.as_uri(), new=2)


if __name__ == "__main__":
    main()
