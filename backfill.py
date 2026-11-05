"""
One-off backfill: replays the screener rules over past price history and writes
reports/backtest_record.csv, then rebuilds docs/track.html.

Run it manually from GitHub Actions (backfill-workflow.yml). It is a backtest:
only today's S&P 500 members and no earnings filter, so results are flattering.
Live results (reports/track_record.csv) are kept separate.
"""

import os
import time

import numpy as np
import pandas as pd
import yfinance as yf

import screener as S

HISTORY_PERIOD = "5y"
SIGNAL_YEARS_BARS = 504        # only create signals for roughly the last 2 years
WIN_RATE_WINDOW = 345          # same amount of history the live screen uses (~500 calendar days)
MIN_BARS_BEFORE_SIGNAL = 260   # need 200-day average and enough history for the win rate


def backfill_ticker(ticker: str, hist: pd.DataFrame) -> list[dict]:
    close_s = S.series_of(hist, "Close").astype(float)
    high_s = S.series_of(hist, "High").astype(float)
    low_s = S.series_of(hist, "Low").astype(float)
    open_s = S.series_of(hist, "Open").astype(float)
    vol_s = S.series_of(hist, "Volume").astype(float)
    n = len(close_s)
    if n < MIN_BARS_BEFORE_SIGNAL + S.MAX_HOLD_DAYS + 5:
        return []

    close, high, low, opn = close_s.values, high_s.values, low_s.values, open_s.values
    sma20 = close_s.rolling(S.SMA_PERIOD).mean()
    std20 = close_s.rolling(S.BOLLINGER_PERIOD).std()
    lower = sma20 - S.BOLLINGER_STDDEV * std20
    upper = sma20 + S.BOLLINGER_STDDEV * std20
    rsi = S.compute_rsi(close_s)
    avg_vol = vol_s.rolling(20).mean()

    rows = []
    start = max(MIN_BARS_BEFORE_SIGNAL, n - SIGNAL_YEARS_BARS)
    i = start
    while i < n - 1:
        direction = S.get_direction(close[i], sma20.iloc[i], lower.iloc[i], upper.iloc[i], rsi.iloc[i])
        if direction is None or avg_vol.iloc[i] < S.MIN_AVG_VOLUME:
            i += 1
            continue
        entry = float(close[i])
        target = float(sma20.iloc[i])
        stop = entry * (1 - S.STOP_LOSS_PCT) if direction == "LONG" else entry * (1 + S.STOP_LOSS_PCT)
        risk = abs(entry - stop)
        reward = (target - entry) if direction == "LONG" else (entry - target)
        if risk <= 0 or reward <= 0 or reward / risk < S.MIN_RISK_REWARD:
            i += 1
            continue

        lo = max(0, i - WIN_RATE_WINDOW + 1)
        stats = S.backtest_hold_time(
            close_s.iloc[lo:i + 1], sma20.iloc[lo:i + 1], rsi.iloc[lo:i + 1], direction
        )
        if stats["win_rate"] is None or stats["win_rate"] < S.MIN_WIN_RATE:
            i += 1
            continue

        pen, reasons, _ = S.structure_penalty(close, high, low, i, direction)
        probability = int(max(5, min(95, stats["win_rate"] - pen)))

        bars = [(opn[j], high[j], low[j], close[j]) for j in range(i + 1, n)]
        result = S._walk_outcome(direction, entry, stop, target, bars)
        if result is None:
            i += 1   # not resolved yet; the live screen handles recent ones
            continue
        status, k, exit_price = result
        rows.append({
            "id": "", "date_flagged": pd.Timestamp(close_s.index[i]).date().isoformat(),
            "ticker": ticker, "direction": direction, "entry": str(round(entry, 2)),
            "stop": str(round(stop, 2)), "target": str(round(target, 2)),
            "probability": str(probability), "hist_win_rate": str(stats["win_rate"]),
            "penalty": str(pen), "reasons": "; ".join(reasons), "status": status,
            "date_resolved": pd.Timestamp(close_s.index[i + k]).date().isoformat(),
            "days_held": str(k), "exit_price": str(round(exit_price, 4)),
            "pnl_pct": str(S._pnl_pct(direction, entry, exit_price)),
        })
        i += k + 1   # no new signal on this ticker until the previous trade is over
    return rows


def main():
    limit = int(os.environ.get("BACKFILL_MAX_TICKERS", "0") or 0)
    records = S.get_sp500_tickers()
    if limit:
        records = records[:limit]
    print(f"Backfilling {len(records)} tickers...")
    all_rows = []
    for k, rec in enumerate(records):
        try:
            hist = yf.download(rec["ticker"], period=HISTORY_PERIOD, progress=False, auto_adjust=True)
        except Exception:
            continue
        if hist is None or hist.empty:
            continue
        all_rows.extend(backfill_ticker(rec["ticker"], hist))
        if k % 25 == 0:
            print(f"...{k}/{len(records)} tickers, {len(all_rows)} signals so far")
        time.sleep(S.REQUEST_PAUSE_SEC)

    df = pd.DataFrame(all_rows, columns=S.TRACK_RECORD_COLUMNS)
    if len(df):
        df = df.sort_values("date_flagged").reset_index(drop=True)
        df["id"] = [str(x + 1) for x in range(len(df))]
    os.makedirs("reports", exist_ok=True)
    df.to_csv(S.BACKTEST_PATH, index=False)
    print(f"Wrote {len(df)} backtest signals to {S.BACKTEST_PATH}")
    S.write_track_page()


if __name__ == "__main__":
    main()
