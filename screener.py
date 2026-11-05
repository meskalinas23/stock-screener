"""
S&P 500 Mean-Reversion Screener
Scans S&P 500 stocks for oversold mean-reversion setups, and writes a
dated markdown/HTML report with entry/stop/target and risk-to-reward for each hit.

This does NOT place any trades. It only researches and reports. You decide.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import datetime as dt
import time

import html as html_lib

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ---------------------------------------------------------------------------
# Config — tune these to change how strict/loose the screener is
# ---------------------------------------------------------------------------
RSI_PERIOD = 7
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
BOLLINGER_PERIOD = 20
BOLLINGER_STDDEV = 2
SMA_PERIOD = 20
MIN_PCT_BELOW_SMA = 0.06
MIN_PCT_ABOVE_SMA = 0.06
STOP_LOSS_PCT = 0.05
LOOKBACK_DAYS = 500
MAX_HOLD_DAYS = 20
MIN_AVG_VOLUME = 300_000
REQUEST_PAUSE_SEC = 0.3
EARNINGS_BLACKOUT_DAYS = 3
MIN_RISK_REWARD = 2.0
MIN_WIN_RATE = 50            # skip setups with historical win rate below 50%
SECTOR_WARNING_COUNT = 3
NEWS_ITEMS_PER_TICKER = 3
VOLUME_SPIKE_RATIO = 1.5
SWING_LOOKBACK = 60          # bars used to find the prior swing low/high
SWING_SKIP = 5               # ignore the most recent bars when finding it
PENALTY_SWING_BREAK = 15     # probability points removed if price broke the swing level
PENALTY_TREND_200 = 8        # ... if price is on the wrong side of the 200-day average
PENALTY_LH_LL = 7            # ... if highs and lows are both moving against the trade
CHART_BARS = 90              # candles shown per chart

SP500_LIST_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/"
    "master/data/constituents.csv"
)

# ---------------------------------------------------------------------------
# Macro event calendar — pulled live from a free public ForexFactory feed.
# No manual date entry needed. Shows today's events plus the full week's
# high-impact USD events (FOMC, CPI, PPI, NFP, GDP, etc.).
# ---------------------------------------------------------------------------
FF_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
MACRO_RELEVANT_COUNTRIES = {"USD"}
MACRO_RELEVANT_IMPACT = {"High"}


def fetch_macro_events() -> list[dict]:
    """Pull this week's economic calendar from the free ForexFactory feed. Best-effort — returns [] on failure."""
    try:
        resp = requests.get(FF_CALENDAR_URL, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []


def check_upcoming_macro_events() -> dict:
    """Returns this week's high-impact USD events, split into 'today' and 'this_week'."""
    events = fetch_macro_events()
    now = pd.Timestamp.now(tz="UTC")
    today_events = []
    week_events = []
    fetch_failed = len(events) == 0

    for e in events:
        if e.get("country") not in MACRO_RELEVANT_COUNTRIES:
            continue
        if e.get("impact") not in MACRO_RELEVANT_IMPACT:
            continue
        try:
            event_time = pd.to_datetime(e["date"], utc=True)
        except (ValueError, KeyError, TypeError):
            continue
        delta_days = (event_time.date() - now.date()).days
        title = e.get("title", "Economic event")
        entry = f"{title} on {event_time.date().isoformat()} at {event_time.strftime('%H:%M UTC')}"
        if delta_days == 0:
            today_events.append(entry)
        if 0 <= delta_days <= 7:
            week_events.append(entry)

    return {"today": today_events, "week": week_events, "fetch_failed": fetch_failed}


def get_sp500_tickers() -> list[dict]:
    df = pd.read_csv(SP500_LIST_URL)
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
    return df[["Symbol", "GICS Sector"]].rename(
        columns={"Symbol": "ticker", "GICS Sector": "sector"}
    ).to_dict("records")


def get_news_headlines(ticker: str, max_items: int = NEWS_ITEMS_PER_TICKER) -> list[dict]:
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception:
        return []
    headlines = []
    for item in raw[:max_items]:
        content = item.get("content", item)
        title = content.get("title") or item.get("title")
        link = (
            content.get("canonicalUrl", {}).get("url")
            if isinstance(content.get("canonicalUrl"), dict)
            else item.get("link")
        )
        if title:
            headlines.append({"title": title, "link": link or ""})
    return headlines


def days_until_next_earnings(ticker: str) -> int | None:
    try:
        edates = yf.Ticker(ticker).get_earnings_dates(limit=4)
        if edates is None or edates.empty:
            return None
        today = pd.Timestamp.now(tz=edates.index.tz)
        future = edates.index[edates.index >= today]
        if len(future) == 0:
            return None
        next_date = future.min()
        return (next_date.date() - today.date()).days
    except Exception:
        return None


def compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def backtest_hold_time(close: pd.Series, sma: pd.Series, rsi: pd.Series, direction: str) -> dict:
    closes = close.values
    smas = sma.values
    rsis = rsi.values
    n = len(closes)
    resolutions = []

    for i in range(SMA_PERIOD, n - 1 - MAX_HOLD_DAYS):
        if pd.isna(rsis[i]) or pd.isna(smas[i]):
            continue
        if direction == "LONG":
            fresh_signal = rsis[i] < RSI_OVERSOLD and not (
                not pd.isna(rsis[i - 1]) and rsis[i - 1] < RSI_OVERSOLD
            )
        else:
            fresh_signal = rsis[i] > RSI_OVERBOUGHT and not (
                not pd.isna(rsis[i - 1]) and rsis[i - 1] > RSI_OVERBOUGHT
            )
        if not fresh_signal:
            continue

        entry = closes[i]
        target = smas[i]
        stop = entry * (1 - STOP_LOSS_PCT) if direction == "LONG" else entry * (1 + STOP_LOSS_PCT)

        for j in range(1, MAX_HOLD_DAYS + 1):
            price = closes[i + j]
            if direction == "LONG":
                if price >= target:
                    resolutions.append((j, True)); break
                if price <= stop:
                    resolutions.append((j, False)); break
            else:
                if price <= target:
                    resolutions.append((j, True)); break
                if price >= stop:
                    resolutions.append((j, False)); break

    if not resolutions:
        return {"avg_hold_days": None, "win_rate": None, "sample_size": 0}
    wins = [r for r in resolutions if r[1]]
    win_rate = round(100 * len(wins) / len(resolutions), 0)
    avg_hold_days = round(sum(r[0] for r in wins) / len(wins), 1) if wins else None
    return {"avg_hold_days": avg_hold_days, "win_rate": win_rate, "sample_size": len(resolutions)}


def series_of(hist, name):
    s = hist[name]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s


def get_direction(price, sma, lower, upper, rsi_val):
    """Same signal rule the live screen uses, usable on any bar (live or backfill)."""
    if pd.isna(sma) or pd.isna(rsi_val) or pd.isna(lower) or pd.isna(upper):
        return None
    pct = (price - sma) / sma
    if rsi_val < RSI_OVERSOLD or price < lower or pct <= -MIN_PCT_BELOW_SMA:
        return "LONG"
    if rsi_val > RSI_OVERBOUGHT or price > upper or pct >= MIN_PCT_ABOVE_SMA:
        return "SHORT"
    return None


def structure_penalty(close, high, low, i, direction):
    """
    Lowers the probability when the chart structure is against a mean-reversion trade.
    close/high/low are numpy arrays, i is the bar index of the signal.
    Returns (penalty_points, [reasons], swing_level_for_chart_or_None).
    """
    penalty = 0
    reasons = []
    need = SWING_LOOKBACK + SWING_SKIP
    swing_level = None
    if i < need:
        return 0, reasons, None

    c = close[i]
    prior_lo = float(np.nanmin(low[i - need: i - SWING_SKIP + 1]))
    prior_hi = float(np.nanmax(high[i - need: i - SWING_SKIP + 1]))
    swing_level = prior_lo if direction == "LONG" else prior_hi

    if direction == "LONG" and c < prior_lo:
        penalty += PENALTY_SWING_BREAK
        reasons.append(f"broke below {SWING_LOOKBACK}d swing low (-{PENALTY_SWING_BREAK})")
    if direction == "SHORT" and c > prior_hi:
        penalty += PENALTY_SWING_BREAK
        reasons.append(f"broke above {SWING_LOOKBACK}d swing high (-{PENALTY_SWING_BREAK})")

    if i >= 199:
        sma200 = float(np.nanmean(close[i - 199: i + 1]))
        if direction == "LONG" and c < sma200:
            penalty += PENALTY_TREND_200
            reasons.append(f"below 200-day avg, downtrend (-{PENALTY_TREND_200})")
        if direction == "SHORT" and c > sma200:
            penalty += PENALTY_TREND_200
            reasons.append(f"above 200-day avg, uptrend (-{PENALTY_TREND_200})")

    if i >= 39:
        rh = float(np.nanmax(high[i - 19: i + 1]))
        rl = float(np.nanmin(low[i - 19: i + 1]))
        ph = float(np.nanmax(high[i - 39: i - 19]))
        pl = float(np.nanmin(low[i - 39: i - 19]))
        if direction == "LONG" and rh < ph and rl < pl:
            penalty += PENALTY_LH_LL
            reasons.append(f"lower highs and lower lows (-{PENALTY_LH_LL})")
        if direction == "SHORT" and rh > ph and rl > pl:
            penalty += PENALTY_LH_LL
            reasons.append(f"higher highs and higher lows (-{PENALTY_LH_LL})")

    return penalty, reasons, swing_level


def analyze_ticker(ticker: str, sector: str) -> dict | None:
    try:
        hist = yf.download(ticker, period=f"{LOOKBACK_DAYS}d", progress=False, auto_adjust=True)
    except Exception:
        return None
    if hist.empty or len(hist) < SMA_PERIOD + 5:
        return None

    close = hist["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    volume = hist["Volume"]
    if isinstance(volume, pd.DataFrame):
        volume = volume.iloc[:, 0]
    avg_volume = volume.tail(20).mean()
    if float(avg_volume) < MIN_AVG_VOLUME:
        return None
    last_volume = volume.iloc[-1]
    if isinstance(last_volume, pd.Series):
        last_volume = last_volume.iloc[0]
    volume_ratio = float(last_volume) / float(avg_volume) if avg_volume > 0 else 0.0
    volume_confirmed = volume_ratio >= VOLUME_SPIKE_RATIO

    sma20 = close.rolling(SMA_PERIOD).mean()
    std20 = close.rolling(BOLLINGER_PERIOD).std()
    lower_band = sma20 - BOLLINGER_STDDEV * std20
    upper_band = sma20 + BOLLINGER_STDDEV * std20
    rsi = compute_rsi(close)

    last_close = float(close.iloc[-1])
    last_sma = float(sma20.iloc[-1])
    last_lower_band = float(lower_band.iloc[-1])
    last_upper_band = float(upper_band.iloc[-1])
    last_rsi = float(rsi.iloc[-1])
    pct_vs_sma = (last_close - last_sma) / last_sma

    is_oversold_rsi = last_rsi < RSI_OVERSOLD
    is_below_band = last_close < last_lower_band
    is_far_below_sma = pct_vs_sma <= -MIN_PCT_BELOW_SMA
    is_overbought_rsi = last_rsi > RSI_OVERBOUGHT
    is_above_band = last_close > last_upper_band
    is_far_above_sma = pct_vs_sma >= MIN_PCT_ABOVE_SMA

    if is_oversold_rsi or is_below_band or is_far_below_sma:
        direction = "LONG"
    elif is_overbought_rsi or is_above_band or is_far_above_sma:
        direction = "SHORT"
    else:
        return None

    entry = last_close
    target = last_sma
    if direction == "LONG":
        stop_loss = entry * (1 - STOP_LOSS_PCT)
        risk = entry - stop_loss
        reward = target - entry
        touched_band = is_below_band
    else:
        stop_loss = entry * (1 + STOP_LOSS_PCT)
        risk = stop_loss - entry
        reward = entry - target
        touched_band = is_above_band

    if risk <= 0 or reward <= 0:
        return None

    days_to_earnings = days_until_next_earnings(ticker)
    if days_to_earnings is not None and 0 <= days_to_earnings <= EARNINGS_BLACKOUT_DAYS:
        return None

    risk_reward = reward / risk
    if risk_reward < MIN_RISK_REWARD:
        return None
    hold_stats = backtest_hold_time(close, sma20, rsi, direction)
    if hold_stats["win_rate"] is None or hold_stats["win_rate"] < MIN_WIN_RATE:
        return None
    news = get_news_headlines(ticker)

    high_s = series_of(hist, "High")
    low_s = series_of(hist, "Low")
    open_s = series_of(hist, "Open")
    pen, pen_reasons, swing_level = structure_penalty(
        close.values.astype(float), high_s.values.astype(float), low_s.values.astype(float),
        len(close) - 1, direction,
    )
    probability = int(max(5, min(95, hold_stats["win_rate"] - pen)))
    chart_df = pd.DataFrame(
        {"Open": open_s, "High": high_s, "Low": low_s, "Close": close, "SMA20": sma20}
    ).tail(CHART_BARS)

    return {
        "ticker": ticker, "sector": sector, "direction": direction,
        "price": round(entry, 2), "rsi": round(last_rsi, 1),
        "pct_vs_sma20": round(pct_vs_sma * 100, 1), "touched_band": touched_band,
        "entry": round(entry, 2), "stop_loss": round(stop_loss, 2), "target": round(target, 2),
        "risk_reward": round(risk_reward, 2), "avg_volume": int(avg_volume),
        "volume_ratio": round(volume_ratio, 2), "volume_confirmed": volume_confirmed,
        "avg_hold_days": hold_stats["avg_hold_days"], "hist_win_rate": hold_stats["win_rate"],
        "hist_sample_size": hold_stats["sample_size"], "days_to_earnings": days_to_earnings, "news": news,
        "probability": probability, "penalty": pen, "penalty_reasons": pen_reasons,
        "swing_level": swing_level, "chart_df": chart_df,
    }


def run_screen(ticker_records: list[dict]) -> list[dict]:
    hits = []
    for i, rec in enumerate(ticker_records):
        result = analyze_ticker(rec["ticker"], rec["sector"])
        if result:
            hits.append(result)
        if i % 25 == 0:
            print(f"...scanned {i}/{len(ticker_records)}")
        time.sleep(REQUEST_PAUSE_SEC)
    hits.sort(key=lambda h: h["risk_reward"], reverse=True)
    return hits


TRACK_RECORD_PATH = "reports/track_record.csv"
BACKTEST_PATH = "reports/backtest_record.csv"
TRACK_RECORD_COLUMNS = [
    "id", "date_flagged", "ticker", "direction", "entry", "stop", "target",
    "probability", "hist_win_rate", "penalty", "reasons",
    "status", "date_resolved", "days_held", "exit_price", "pnl_pct",
]
CLOSED_STATUSES = ["win", "loss", "expired"]


def load_track_record(path: str = TRACK_RECORD_PATH) -> pd.DataFrame:
    """Everything is read as text so old files (fewer columns) and new files mix safely."""
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except FileNotFoundError:
        df = pd.DataFrame(columns=TRACK_RECORD_COLUMNS)
    for c in TRACK_RECORD_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    return df[TRACK_RECORD_COLUMNS].copy()


def _walk_outcome(direction, entry, stop, target, bars):
    """
    bars = list of (open, high, low, close) for each trading day AFTER the signal day.
    First level touched decides. If stop and target are both inside one day's range,
    it counts as a loss (conservative). A gap through the stop exits at the open.
    Returns (status, bars_held, exit_price) or None if still open.
    """
    for k, (o, h, l, c) in enumerate(bars[:MAX_HOLD_DAYS], start=1):
        if direction == "LONG":
            if l <= stop:
                return "loss", k, (o if o <= stop else stop)
            if h >= target:
                return "win", k, target
        else:
            if h >= stop:
                return "loss", k, (o if o >= stop else stop)
            if l <= target:
                return "win", k, target
        if k >= MAX_HOLD_DAYS:
            return "expired", k, c
    return None


def _pnl_pct(direction, entry, exit_price):
    sign = 1 if direction == "LONG" else -1
    return round(sign * (exit_price - entry) / entry * 100, 2)


def _bars_after(hist: pd.DataFrame, signal_date: dt.date):
    o = series_of(hist, "Open")
    h = series_of(hist, "High")
    l = series_of(hist, "Low")
    c = series_of(hist, "Close")
    out = []
    for idx in range(len(c)):
        d = pd.Timestamp(c.index[idx]).date()
        if d > signal_date:
            out.append((d, float(o.iloc[idx]), float(h.iloc[idx]), float(l.iloc[idx]), float(c.iloc[idx])))
    return out


def resolve_open_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Walk every daily candle since each open signal. Stop/target touched on ANY day counts."""
    open_rows = df[df["status"] == "open"]
    for ticker, grp in open_rows.groupby("ticker"):
        first_date = min(dt.date.fromisoformat(d) for d in grp["date_flagged"])
        try:
            hist = yf.download(ticker, start=first_date.isoformat(), progress=False, auto_adjust=True)
        except Exception:
            continue
        if hist is None or hist.empty:
            continue
        for idx, row in grp.iterrows():
            signal_date = dt.date.fromisoformat(row["date_flagged"])
            bars = _bars_after(hist, signal_date)
            result = _walk_outcome(
                row["direction"], float(row["entry"]), float(row["stop"]), float(row["target"]),
                [(o, h, l, c) for (_, o, h, l, c) in bars],
            )
            if result is None:
                continue
            status, k, exit_price = result
            df.loc[idx, "status"] = status
            df.loc[idx, "date_resolved"] = bars[k - 1][0].isoformat()
            df.loc[idx, "days_held"] = str(k)
            df.loc[idx, "exit_price"] = str(round(exit_price, 4))
            df.loc[idx, "pnl_pct"] = str(_pnl_pct(row["direction"], float(row["entry"]), exit_price))
    return df


def append_new_hits(df: pd.DataFrame, hits: list[dict]) -> pd.DataFrame:
    today = dt.date.today().isoformat()
    ids = pd.to_numeric(df["id"], errors="coerce")
    next_id = int(ids.max()) + 1 if len(ids.dropna()) else 1
    open_keys = set(zip(df.loc[df["status"] == "open", "ticker"], df.loc[df["status"] == "open", "direction"]))
    new_rows = []
    for h in hits:
        if (h["ticker"], h["direction"]) in open_keys:
            continue  # already tracking this one, don't double-count the same setup
        if ((df["ticker"] == h["ticker"]) & (df["date_flagged"] == today)).any():
            continue  # re-run on the same day
        new_rows.append({
            "id": str(next_id), "date_flagged": today, "ticker": h["ticker"],
            "direction": h["direction"], "entry": str(h["entry"]), "stop": str(h["stop_loss"]),
            "target": str(h["target"]), "probability": str(h["probability"]),
            "hist_win_rate": str(h["hist_win_rate"]), "penalty": str(h["penalty"]),
            "reasons": "; ".join(h["penalty_reasons"]), "status": "open",
            "date_resolved": "", "days_held": "", "exit_price": "", "pnl_pct": "",
        })
        next_id += 1
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    return df


def _num(series):
    return pd.to_numeric(series, errors="coerce")


def summarize_track_record(df: pd.DataFrame) -> dict:
    closed = df[df["status"].isin(CLOSED_STATUSES)]
    decided = df[df["status"].isin(["win", "loss"])]
    open_n = int((df["status"] == "open").sum())
    if closed.empty:
        return {"n": 0, "win_rate": None, "avg_hold_days": None, "open": open_n, "pred": None, "actual": None, "n_cal": 0}
    wins = df[df["status"] == "win"]
    win_rate = round(100 * len(wins) / len(decided), 0) if len(decided) else None
    avg_hold = round(_num(wins["days_held"]).mean(), 1) if len(wins) else None
    cal = decided[_num(decided["probability"]).notna()]
    pred = round(_num(cal["probability"]).mean(), 1) if len(cal) else None
    actual = round(100 * (cal["status"] == "win").mean(), 1) if len(cal) else None
    return {"n": len(closed), "win_rate": win_rate, "avg_hold_days": avg_hold, "open": open_n,
            "pred": pred, "actual": actual, "n_cal": len(cal)}


def update_track_record(hits: list[dict]):
    import os
    df = load_track_record()
    df = resolve_open_positions(df)
    df = append_new_hits(df, hits)
    os.makedirs("reports", exist_ok=True)
    df.to_csv(TRACK_RECORD_PATH, index=False)
    return summarize_track_record(df), df


# ---------------------------------------------------------------------------
# Track record page (docs/track.html): results + predicted vs actual
# ---------------------------------------------------------------------------
BUCKETS = [("below 50%", 0, 50), ("50-59%", 50, 60), ("60-69%", 60, 70), ("70%+", 70, 101)]


def calibration_tables(df: pd.DataFrame) -> dict:
    decided = df[df["status"].isin(["win", "loss"])].copy()
    decided["p"] = _num(decided["probability"])
    decided = decided[decided["p"].notna()]
    decided["won"] = (decided["status"] == "win").astype(int)
    out = {"overall": None, "buckets": [], "months": []}
    if decided.empty:
        return out
    out["overall"] = {"n": len(decided), "pred": decided["p"].mean(), "actual": 100 * decided["won"].mean()}
    for label, lo, hi in BUCKETS:
        sub = decided[(decided["p"] >= lo) & (decided["p"] < hi)]
        if len(sub):
            out["buckets"].append({"label": label, "n": len(sub), "pred": sub["p"].mean(), "actual": 100 * sub["won"].mean()})
    decided["month"] = decided["date_flagged"].str[:7]
    for m, sub in decided.groupby("month"):
        out["months"].append({"label": m, "n": len(sub), "pred": sub["p"].mean(), "actual": 100 * sub["won"].mean()})
    return out


def _cal_rows_html(rows: list[dict]) -> str:
    if not rows:
        return "<tr><td colspan='5'>Not enough closed signals yet.</td></tr>"
    html = ""
    for r in rows:
        diff = r["actual"] - r["pred"]
        cls = "good" if diff >= -3 else "bad"
        flag = " (few trades, low confidence)" if r["n"] < 30 else ""
        html += (f"<tr><td>{r['label']}</td><td>{r['n']}{flag}</td><td>{r['pred']:.0f}%</td>"
                 f"<td>{r['actual']:.0f}%</td><td class='{cls}'>{diff:+.0f} pts</td></tr>")
    return html


def _fmt(x, d=1):
    return "n/a" if pd.isna(x) else f"{x:+.{d}f}%"


def _stats_block(df: pd.DataFrame, title: str, note: str) -> str:
    closed = df[df["status"].isin(CLOSED_STATUSES)]
    decided = df[df["status"].isin(["win", "loss"])]
    n_open = int((df["status"] == "open").sum())
    if closed.empty:
        return f"<h2>{title}</h2><p class='muted'>{note}</p><p>No closed signals yet. {n_open} open.</p>"
    pnl = _num(closed["pnl_pct"])
    wins_pnl = pnl[closed["status"] == "win"]
    loss_pnl = pnl[closed["status"] == "loss"]
    win_rate = 100 * (decided["status"] == "win").mean() if len(decided) else float("nan")
    expectancy = pnl.mean()
    cal = calibration_tables(df)
    ov = cal["overall"]
    if ov:
        gap = ov["actual"] - ov["pred"]
        verdict = ("Probabilities look honest." if abs(gap) <= 5
                   else ("Probabilities are too optimistic (reality is worse)." if gap < 0
                         else "Probabilities are too pessimistic (reality is better)."))
        headline = (f"<p class='big'>Average predicted probability <b>{ov['pred']:.0f}%</b> vs "
                    f"actual win rate <b>{ov['actual']:.0f}%</b> over {ov['n']} decided signals. {verdict}</p>")
    else:
        headline = "<p class='big'>No decided signals with a stored probability yet.</p>"
    return f"""
    <h2>{title}</h2>
    <p class="muted">{note}</p>
    {headline}
    <div class="cards">
      <div><span>Closed</span><b>{len(closed)}</b></div>
      <div><span>Open</span><b>{n_open}</b></div>
      <div><span>Win rate (target vs stop)</span><b>{win_rate:.0f}%</b></div>
      <div><span>Avg win</span><b>{_fmt(wins_pnl.mean())}</b></div>
      <div><span>Avg loss</span><b>{_fmt(loss_pnl.mean())}</b></div>
      <div><span>Avg result per trade</span><b>{_fmt(expectancy, 2)}</b></div>
      <div><span>Expired (no hit in {MAX_HOLD_DAYS}d)</span><b>{int((closed['status']=='expired').sum())}</b></div>
    </div>
    <h3>Predicted vs actual, by probability bucket</h3>
    <table><thead><tr><th>Predicted bucket</th><th>Signals</th><th>Avg predicted</th><th>Actual win rate</th><th>Gap</th></tr></thead>
    <tbody>{_cal_rows_html(cal['buckets'])}</tbody></table>
    <h3>Predicted vs actual, by month signaled</h3>
    <table><thead><tr><th>Month</th><th>Signals</th><th>Avg predicted</th><th>Actual win rate</th><th>Gap</th></tr></thead>
    <tbody>{_cal_rows_html(cal['months'])}</tbody></table>
    """


def _signals_table(df: pd.DataFrame, limit: int = 150) -> str:
    if df.empty:
        return "<p>No signals logged yet.</p>"
    icon = {"win": "✅ win", "loss": "❌ stop hit", "expired": "⌛ expired", "open": "⏳ open"}
    order = {"open": 0, "win": 1, "loss": 1, "expired": 1}
    d = df.copy()
    d["o"] = d["status"].map(order).fillna(1)
    d = d.sort_values(["o", "date_flagged"], ascending=[True, False]).head(limit)
    rows = ""
    for _, r in d.iterrows():
        pnl = f"{float(r['pnl_pct']):+.1f}%" if r["pnl_pct"] not in ("", None) else ""
        prob = f"{float(r['probability']):.0f}%" if r["probability"] not in ("", None) else "n/a"
        cls = "long" if r["direction"] == "LONG" else "short"
        rows += (f"<tr><td>{html_lib.escape(r['date_flagged'])}</td><td>{html_lib.escape(r['ticker'])}</td>"
                 f"<td class='{cls}'>{r['direction']}</td><td>${r['entry']}</td><td>${r['stop']}</td><td>${r['target']}</td>"
                 f"<td>{prob}</td><td>{icon.get(r['status'], r['status'])}</td><td>{r['days_held']}</td><td>{pnl}</td>"
                 f"<td class='muted'>{html_lib.escape(str(r['reasons']))}</td></tr>")
    return (f"<table><thead><tr><th>Signaled</th><th>Ticker</th><th>Dir</th><th>Entry</th><th>Stop</th><th>Target</th>"
            f"<th>Prob.</th><th>Result</th><th>Days</th><th>P/L</th><th>Structure flags</th></tr></thead><tbody>{rows}</tbody></table>")


def write_track_page():
    import os
    live = load_track_record(TRACK_RECORD_PATH)
    bt = load_track_record(BACKTEST_PATH) if os.path.exists(BACKTEST_PATH) else None
    parts = [_stats_block(live, "Live results",
                          "Real signals the screener showed, checked against every daily candle after the signal. "
                          "Stop and target touched on any day count; if both in one day it counts as a loss.")]
    parts.append("<h3>All live signals</h3>" + _signals_table(live))
    if bt is not None and len(bt):
        parts.append(_stats_block(bt, "Backtest (rules replayed on past data)",
                                  "Same rules run over past price history. Flattering: only today's S&P 500 members, "
                                  "no earnings filter, so treat as a rough guide, not proof."))
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Screener track record</title>
<style>
 body {{ font-family: -apple-system, Arial, sans-serif; max-width: 1100px; margin: 20px auto; padding: 0 15px; }}
 table {{ border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 0.9em; }}
 th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }} th {{ background: #f5f5f5; }}
 .long {{ color: #0a7d2c; font-weight: bold; }} .short {{ color: #c0392b; font-weight: bold; }}
 .good {{ color: #0a7d2c; }} .bad {{ color: #c0392b; font-weight: bold; }}
 .muted {{ color: #666; font-size: 0.9em; }} .big {{ font-size: 1.1em; }}
 .cards {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 10px 0; }}
 .cards div {{ background: #f5f5f5; border-radius: 6px; padding: 8px 14px; min-width: 120px; }}
 .cards span {{ display: block; font-size: 0.8em; color: #666; }} .cards b {{ font-size: 1.2em; }}
</style></head><body>
<p><a href="index.html">&larr; Back to today's setups</a></p>
<h1>Screener track record</h1>
<p class="muted">Updated {dt.date.today().isoformat()}. Win = target hit before stop. Probability = historical win rate minus structure penalties.</p>
{''.join(parts)}
</body></html>"""
    os.makedirs("docs", exist_ok=True)
    with open("docs/track.html", "w") as f:
        f.write(html)
    print("Wrote docs/track.html")


def build_report(hits: list[dict], track_summary: dict) -> str:
    today = dt.date.today().isoformat()
    lines = [f"# Mean-Reversion Screener — {today}", ""]

    macro = check_upcoming_macro_events()
    if macro["fetch_failed"]:
        lines.append("*Macro calendar: couldn't reach the feed this run — check manually if needed.*")
        lines.append("")
    else:
        if macro["today"]:
            lines.append("**📅 Today's high-impact events:**")
            for w in macro["today"]:
                lines.append(f"- {w}")
            lines.append("")
        if macro["week"]:
            lines.append("**This week's high-impact events:**")
            for w in macro["week"]:
                lines.append(f"- {w}")
            lines.append("")
        if not macro["today"] and not macro["week"]:
            lines.append("*No high-impact USD events found this week.*")
            lines.append("")

    if track_summary["n"] > 0:
        lines.append(
            f"**Live track record:** {track_summary['n']} resolved picks, "
            f"{(track_summary['win_rate'] or 0):.0f}% win rate, "
            f"avg {track_summary['avg_hold_days']} days held on winners. "
            f"(Small samples early on aren't statistically meaningful yet.)"
        )
        lines.append("")

    if not hits:
        lines.append("No candidates flagged today.")
        return "\n".join(lines)

    sector_counts: dict[str, int] = {}
    for h in hits:
        sector_counts[h["sector"]] = sector_counts.get(h["sector"], 0) + 1
    crowded = {s: c for s, c in sector_counts.items() if c >= SECTOR_WARNING_COUNT}
    if crowded:
        lines.append("**Sector concentration warning:**")
        for s, c in crowded.items():
            lines.append(f"- {c} of today's picks are in {s} — likely one shared sector move, not {c} independent opportunities")
        lines.append("")

    lines.append(f"{len(hits)} candidate(s) flagged. Highest risk/reward first.")
    lines.append("")
    lines.append(
        "| Ticker | Sector | Direction | Price | RSI(7) | % vs 20D Avg | "
        "Volume vs Avg | Entry | Stop | Target | Risk:Reward | Avg Hold (days) | "
        "Historical Win Rate | Sample | Next Earnings |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for h in hits:
        hold = h["avg_hold_days"] if h["avg_hold_days"] is not None else "n/a"
        win_rate = f"{h['hist_win_rate']:.0f}%" if h["hist_win_rate"] is not None else "n/a"
        earnings = f"{h['days_to_earnings']}d" if h["days_to_earnings"] is not None else "unknown"
        vol = f"{h['volume_ratio']}x" + (" 🔥" if h["volume_confirmed"] else "")
        lines.append(
            f"| {h['ticker']} | {h['sector']} | {h['direction']} | ${h['price']} | "
            f"{h['rsi']} | {h['pct_vs_sma20']}% | {vol} | ${h['entry']} | ${h['stop_loss']} | "
            f"${h['target']} | {h['risk_reward']}:1 | {hold} | {win_rate} | "
            f"{h['hist_sample_size']} | {earnings} |"
        )

    lines.append("")
    lines.append(
        "Notes: LONG = oversold, expected to bounce back up toward the "
        "20-day average. SHORT = overbought, expected to pull back down "
        "toward the 20-day average. Target = reversion to the 20-day "
        "moving average. Stop-loss = 5% against the position. "
        "'Avg Hold (days)' and 'Historical Win Rate' come from backtesting "
        "this same setup on this ticker's own past ~2 years — 'Sample' is "
        "how many past instances that's based on; under ~5 isn't reliable. "
        "Tickers with earnings due within "
        f"{EARNINGS_BLACKOUT_DAYS} days are excluded entirely (gap risk). "
        f"'Volume vs Avg' 🔥 means today's volume was {VOLUME_SPIKE_RATIO}x+ "
        "the 20-day average. This is a mechanical screen only — check the "
        "news below before acting. Not financial advice."
    )

    lines.append("")
    lines.append("## News context")
    for h in hits:
        lines.append(f"\n**{h['ticker']}**")
        if h["news"]:
            for n in h["news"]:
                if n["link"]:
                    lines.append(f"- [{n['title']}]({n['link']})")
                else:
                    lines.append(f"- {n['title']}")
        else:
            lines.append("- No recent headlines found — check manually before acting.")

    return "\n".join(lines)


def build_risk_reward_chart(h: dict, out_dir="docs/charts") -> str:
    """Candlestick chart (last ~90 days) with 20-day average, entry, stop, target and swing level."""
    import os
    os.makedirs(out_dir, exist_ok=True)

    d = h["chart_df"].dropna(subset=["Open", "High", "Low", "Close"])
    n = len(d)
    entry, stop, target = h["entry"], h["stop_loss"], h["target"]
    direction = h["direction"]

    fig, ax = plt.subplots(figsize=(8.5, 3.9))
    up_color, down_color = "#26a69a", "#ef5350"
    for i, (_, r) in enumerate(d.iterrows()):
        color = up_color if r["Close"] >= r["Open"] else down_color
        ax.vlines(i, r["Low"], r["High"], color=color, linewidth=1)
        body_lo, body_hi = min(r["Open"], r["Close"]), max(r["Open"], r["Close"])
        ax.bar(i, max(body_hi - body_lo, entry * 0.0003), bottom=body_lo, width=0.65, color=color)

    ax.plot(range(n), d["SMA20"].values, color="#1f77b4", linewidth=1.2, label="20-day avg (target)")

    # risk / reward zones to the right of the last candle
    x0, x1 = n - 1, n + 9
    if direction == "LONG":
        ax.fill_between([x0, x1], stop, entry, color="#e74c3c", alpha=0.22)
        ax.fill_between([x0, x1], entry, target, color="#27ae60", alpha=0.22)
    else:
        ax.fill_between([x0, x1], entry, stop, color="#e74c3c", alpha=0.22)
        ax.fill_between([x0, x1], target, entry, color="#27ae60", alpha=0.22)

    ax.axhline(entry, color="#333", linewidth=1.0)
    ax.axhline(stop, color="#c0392b", linewidth=1, linestyle="--")
    ax.axhline(target, color="#0a7d2c", linewidth=1, linestyle="--")
    label = dict(va="center", ha="left", fontsize=8.5, bbox=dict(facecolor="white", edgecolor="none", pad=1.2))
    ax.text(n + 0.3, entry, f"Entry ${entry}", color="#333", **label)
    ax.text(n + 0.3, stop, f"Stop ${stop}", color="#c0392b", **label)
    ax.text(n + 0.3, target, f"Target ${target}", color="#0a7d2c", **label)

    swing = h.get("swing_level")
    lows = [d["Low"].min(), stop, target, entry]
    highs = [d["High"].max(), stop, target, entry]
    if swing is not None:
        ax.axhline(swing, color="#888", linewidth=1, linestyle=":")
        name = "60d swing low" if direction == "LONG" else "60d swing high"
        ax.text(0.5, swing, f"{name} ${swing:.2f}", color="#666", fontsize=8, va="bottom")
        lows.append(swing)
        highs.append(swing)

    pad = (max(highs) - min(lows)) * 0.06
    ax.set_ylim(min(lows) - pad, max(highs) + pad)
    ax.set_xlim(-1, n + 14)
    ticks = list(range(0, n, 15))
    ax.set_xticks(ticks)
    ax.set_xticklabels([pd.Timestamp(d.index[t]).strftime("%b %d") for t in ticks], fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.set_title(f"{h['ticker']} - {direction} - probability {h['probability']}% - R:R {h['risk_reward']}:1", fontsize=10)
    fig.tight_layout()

    today = dt.date.today().isoformat()
    filename = f"{h['ticker']}_{today}.png"
    fig.savefig(os.path.join(out_dir, filename), dpi=110)
    plt.close(fig)
    return f"charts/{filename}"


def build_html_report(hits: list[dict], track_summary: dict, history_df=None) -> str:
    today = dt.date.today().isoformat()

    macro = check_upcoming_macro_events()
    if macro["fetch_failed"]:
        macro_html = "<div class='warning'><em>Macro calendar: couldn't reach the feed this run — check manually if needed.</em></div>"
    elif not macro["today"] and not macro["week"]:
        macro_html = "<div class='warning'><em>No high-impact USD events found this week.</em></div>"
    else:
        today_html = ""
        if macro["today"]:
            items = "".join(f"<li>{w}</li>" for w in macro["today"])
            today_html = f"<strong>📅 Today's high-impact events:</strong><ul>{items}</ul>"
        week_html = ""
        if macro["week"]:
            items = "".join(f"<li>{w}</li>" for w in macro["week"])
            week_html = f"<strong>This week's high-impact events:</strong><ul>{items}</ul>"
        macro_html = f"<div class='warning'>{today_html}{week_html}</div>"

    track_html = ""
    if track_summary["n"] > 0 and track_summary.get("pred") is not None:
        track_html = f"""
        <p class="track-record"><strong>Live track record:</strong>
        {track_summary['n']} closed picks, {(track_summary['win_rate'] or 0):.0f}% win rate.
        Average predicted probability {track_summary['pred']:.0f}% vs actual {track_summary['actual']:.0f}%
        ({track_summary['n_cal']} decided). {track_summary['open']} still open.
        <a href="track.html">Full track record &rarr;</a></p>
        """
    elif track_summary["n"] > 0:
        track_html = f"""
        <p class="track-record"><strong>Live track record:</strong>
        {track_summary['n']} closed picks, {(track_summary['win_rate'] or 0):.0f}% win rate.
        <a href="track.html">Full track record &rarr;</a></p>
        """
    else:
        track_html = '<p class="track-record">Live track record: nothing closed yet. <a href="track.html">Track record page &rarr;</a></p>'

    sector_html = ""
    if hits:
        sector_counts: dict = {}
        for h in hits:
            sector_counts[h["sector"]] = sector_counts.get(h["sector"], 0) + 1
        crowded = {s: c for s, c in sector_counts.items() if c >= SECTOR_WARNING_COUNT}
        if crowded:
            items = "".join(
                f"<li>{c} of today's picks are in {s} &mdash; likely one shared sector move, not {c} independent opportunities</li>"
                for s, c in crowded.items()
            )
            sector_html = f"<div class='warning'><strong>Sector concentration warning:</strong><ul>{items}</ul></div>"

    if not hits:
        table_html = "<p>No candidates flagged today.</p>"
        news_html = ""
    else:
        rows = ""
        for h in hits:
            hold = h["avg_hold_days"] if h["avg_hold_days"] is not None else "n/a"
            win_rate = f"{h['hist_win_rate']:.0f}%" if h["hist_win_rate"] is not None else "n/a"
            earnings = f"{h['days_to_earnings']}d" if h["days_to_earnings"] is not None else "unknown"
            vol = f"{h['volume_ratio']}x" + (" &#128293;" if h["volume_confirmed"] else "")
            direction_class = "long" if h["direction"] == "LONG" else "short"
            rows += f"""
            <tr>
                <td>{h['ticker']}</td>
                <td>{h['sector']}</td>
                <td class="{direction_class}">{h['direction']}</td>
                <td>${h['price']}</td>
                <td>{h['rsi']}</td>
                <td>{h['pct_vs_sma20']}%</td>
                <td>{vol}</td>
                <td>${h['entry']}</td>
                <td>${h['stop_loss']}</td>
                <td>${h['target']}</td>
                <td>{h['risk_reward']}:1</td>
                <td>{hold}</td>
                <td title="{html_lib.escape('; '.join(h['penalty_reasons']) or 'no structure penalty')}"><b>{h['probability']}%</b></td>
                <td>{win_rate}</td>
                <td>{h['hist_sample_size']}</td>
                <td>{earnings}</td>
            </tr>
            """
        table_html = f"""
        <table>
            <thead>
                <tr>
                    <th>Ticker</th><th>Sector</th><th>Dir</th><th>Price</th><th>RSI(7)</th>
                    <th>% vs 20D</th><th>Volume</th><th>Entry</th><th>Stop</th><th>Target</th>
                    <th>R:R</th><th>Avg Hold</th><th>Probability</th><th>Hist. win rate</th><th>Sample</th><th>Earnings</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
        """

        news_sections = ""
        for h in hits:
            chart_path = build_risk_reward_chart(h)
            if h["news"]:
                items = "".join(
                    f"<li><a href='{html_lib.escape(str(n['link']))}'>{html_lib.escape(str(n['title']))}</a></li>" if n["link"]
                    else f"<li>{html_lib.escape(str(n['title']))}</li>"
                    for n in h["news"]
                )
            else:
                items = "<li>No recent headlines found &mdash; check manually before acting.</li>"
            if h["penalty_reasons"]:
                flags = (f"<p class='flags'>Probability {h['hist_win_rate']:.0f}% &rarr; <b>{h['probability']}%</b>: "
                         + html_lib.escape("; ".join(h["penalty_reasons"])) + "</p>")
            else:
                flags = f"<p class='flags ok'>Probability {h['probability']}%: no structure warnings.</p>"
            prior_html = ""
            if history_df is not None and len(history_df):
                prev = history_df[(history_df["ticker"] == h["ticker"]) & (history_df["date_flagged"] != dt.date.today().isoformat())]
                chips = ""
                for _, r in prev.sort_values("date_flagged", ascending=False).head(3).iterrows():
                    label = {"win": "&#9989; hit target", "loss": "&#10060; hit stop", "expired": "&#8987; expired", "open": "&#9203; still open"}.get(r["status"], r["status"])
                    pnl = f" {float(r['pnl_pct']):+.1f}%" if r["pnl_pct"] not in ("", None) else ""
                    chips += f"<span class='chip'>{r['date_flagged']}: {label}{pnl}</span> "
                if chips:
                    prior_html = f"<p>Earlier signals on {html_lib.escape(h['ticker'])}: {chips}</p>"
            direction_class = "long" if h["direction"] == "LONG" else "short"
            news_sections += (f"<h3>{html_lib.escape(h['ticker'])} <span class='{direction_class}'>{h['direction']}</span> "
                              f"&mdash; {h['probability']}%</h3>{flags}{prior_html}"
                              f"<img src='{chart_path}' alt='{html_lib.escape(h['ticker'])} chart' style='max-width:100%;'><ul>{items}</ul>")
        news_html = f"<h2>Charts &amp; news</h2>{news_sections}"

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Mean-Reversion Screener</title>
    <style>
        body {{ font-family: -apple-system, Arial, sans-serif; max-width: 1100px; margin: 20px auto; padding: 0 15px; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 15px; font-size: 0.9em; }}
        th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
        th {{ background: #f5f5f5; }}
        .long {{ color: #0a7d2c; font-weight: bold; }}
        .short {{ color: #c0392b; font-weight: bold; }}
        .updated {{ color: #666; font-size: 0.9em; }}
        .warning {{ background: #fff8e1; border: 1px solid #f0d060; padding: 10px 15px; border-radius: 4px; margin: 15px 0; }}
        .track-record {{ background: #eef7ee; border: 1px solid #b7d9b7; padding: 10px 15px; border-radius: 4px; }}
        .notes {{ color: #555; font-size: 0.85em; margin-top: 15px; }}
        .archive {{ columns: 3; column-gap: 20px; list-style: none; padding: 0; }}
        .archive li {{ margin-bottom: 4px; }}
        .flags {{ color: #a05a00; font-size: 0.9em; margin: 4px 0; }}
        .flags.ok {{ color: #0a7d2c; }}
        .chip {{ background: #f0f0f0; border-radius: 10px; padding: 2px 8px; font-size: 0.85em; }}
    </style>
</head>
<body>
    <h1>Mean-Reversion Screener</h1>
    <p class="updated">Last updated: {today}</p>
    {macro_html}
    {track_html}
    {sector_html}
    <h2>Setups ({len(hits)})</h2>
    {table_html}
    <p class="notes">
        LONG = oversold, expected to bounce back up toward the 20-day average.
        SHORT = overbought, expected to pull back down toward the 20-day average.
        Stop-loss = 5% against the position. Probability = this ticker's historical win rate minus penalties when the chart structure is broken (hover for reasons). &#128293; means volume was {VOLUME_SPIKE_RATIO}x+ the 20-day average.
        Tickers with earnings due within {EARNINGS_BLACKOUT_DAYS} days are excluded.
        Macro events pulled live from a public economic calendar feed (high-impact USD events only).
        Mechanical screen only &mdash; check news before acting. Not financial advice.
    </p>
    {news_html}
</body>
</html>
"""
    return html


def build_archive_list_html() -> str:
    import os
    archive_dir = "docs/reports"
    if not os.path.isdir(archive_dir):
        return ""
    dates = sorted((f[:-5] for f in os.listdir(archive_dir) if f.endswith(".html")), reverse=True)
    if not dates:
        return ""
    items = "".join(f"<li><a href='reports/{d}.html'>{d}</a></li>" for d in dates)
    return f"<h2>Past reports</h2><ul class='archive'>{items}</ul>"


def write_html_report(hits: list[dict], track_summary: dict, history_df=None):
    import os
    today = dt.date.today().isoformat()
    os.makedirs("docs/reports", exist_ok=True)

    main_html = build_html_report(hits, track_summary, history_df)

    # the dated archive copy lives one folder deeper (docs/reports/) than the
    # chart images (docs/charts/), so its image paths need "../" prepended
    dated_html = main_html.replace("src='charts/", "src='../charts/")
    dated_html = dated_html.replace('href="track.html"', 'href="../track.html"')
    with open(f"docs/reports/{today}.html", "w") as f:
        f.write(dated_html)

    archive_html = build_archive_list_html()
    main_html_with_archive = main_html.replace("</body>", f"{archive_html}</body>")
    with open("docs/index.html", "w") as f:
        f.write(main_html_with_archive)

    print(f"Wrote docs/index.html and docs/reports/{today}.html")


def main():
    print("Fetching S&P 500 ticker list...")
    ticker_records = get_sp500_tickers()
    print(f"Scanning {len(ticker_records)} tickers...")
    hits = run_screen(ticker_records)

    print("Updating track record...")
    track_summary, track_df = update_track_record(hits)
    write_track_page()

    report = build_report(hits, track_summary)

    today = dt.date.today().isoformat()
    out_path = f"reports/{today}.md"
    import os
    os.makedirs("reports", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report)
    with open("reports/latest.md", "w") as f:
        f.write(report)

    write_html_report(hits, track_summary, track_df)

    print(f"Report written to {out_path}")
    print(report)


if __name__ == "__main__":
    main()
