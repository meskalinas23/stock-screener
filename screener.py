"""
S&P 500 Mean-Reversion Screener
Scans S&P 500 stocks for oversold mean-reversion setups, and writes a
dated markdown report with entry/stop/target and risk-to-reward for each hit.

This does NOT place any trades. It only researches and reports. You decide.
"""

import datetime as dt
import time

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Config — tune these to change how strict/loose the screener is
# ---------------------------------------------------------------------------
RSI_PERIOD = 7
RSI_OVERSOLD = 30          # flag LONG candidates if RSI drops below this
RSI_OVERBOUGHT = 70        # flag SHORT candidates if RSI rises above this
BOLLINGER_PERIOD = 20
BOLLINGER_STDDEV = 2
SMA_PERIOD = 20
MIN_PCT_BELOW_SMA = 0.06   # flag LONG if price is 6%+ below its 20-day SMA
MIN_PCT_ABOVE_SMA = 0.06   # flag SHORT if price is 6%+ above its 20-day SMA
STOP_LOSS_PCT = 0.05       # stop-loss 5% away from entry, either direction
LOOKBACK_DAYS = 500        # ~2 years — needed so there's history to backtest hold time on
MAX_HOLD_DAYS = 20         # cap how many days forward we look when timing a historical trade
MIN_AVG_VOLUME = 300_000   # skip illiquid names
REQUEST_PAUSE_SEC = 0.3    # pause between tickers to avoid rate-limiting
EARNINGS_BLACKOUT_DAYS = 3 # skip a hit if earnings fall within this many days
SECTOR_WARNING_COUNT = 3   # warn if this many+ hits share one sector
NEWS_ITEMS_PER_TICKER = 3  # headlines to pull for each flagged ticker
VOLUME_SPIKE_RATIO = 1.5   # today's volume vs 20-day avg — above this = "confirmed" by volume

SP500_LIST_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/"
    "master/data/constituents.csv"
)

# ---------------------------------------------------------------------------
# Macro event calendar — fill this in yourself from official sources:
#   FOMC meeting dates: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
#   CPI/PPI release schedule: https://www.bls.gov/schedule/news_release/2026_sched.htm
# Add one entry per event as ("YYYY-MM-DD", "Event name"). The screener will
# warn you in the report if a listed event falls within MACRO_WARNING_DAYS.
# ---------------------------------------------------------------------------
MACRO_EVENTS: list[tuple[str, str]] = [
    # ("2026-09-16", "FOMC Meeting"),
    # ("2026-09-11", "CPI Release"),
    # ("2026-09-10", "PPI Release"),
]
MACRO_WARNING_DAYS = 3


def get_sp500_tickers() -> list[dict]:
    df = pd.read_csv(SP500_LIST_URL)
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
    return df[["Symbol", "GICS Sector"]].rename(
        columns={"Symbol": "ticker", "GICS Sector": "sector"}
    ).to_dict("records")


def get_news_headlines(ticker: str, max_items: int = NEWS_ITEMS_PER_TICKER) -> list[dict]:
    """Pull recent headlines for a ticker. Best-effort — returns [] on any failure."""
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception:
        return []

    headlines = []
    for item in raw[:max_items]:
        # yfinance news items are nested dicts; be defensive about the shape
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
    """Trading days until next earnings report. None if unknown/unavailable."""
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


def check_upcoming_macro_events(days_ahead: int = MACRO_WARNING_DAYS) -> list[str]:
    today = dt.date.today()
    warnings = []
    for date_str, name in MACRO_EVENTS:
        try:
            event_date = dt.date.fromisoformat(date_str)
        except ValueError:
            continue
        delta = (event_date - today).days
        if 0 <= delta <= days_ahead:
            when = "today" if delta == 0 else f"in {delta} day(s)"
            warnings.append(f"{name} on {date_str} ({when})")
    return warnings


def compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def backtest_hold_time(
    close: pd.Series, sma: pd.Series, rsi: pd.Series, direction: str
) -> dict:
    """
    Look back through this ticker's own history for prior instances of the
    same kind of setup (RSI crossing into oversold/overbought), then measure
    how many trading days it took, on average, for price to reach the
    20-day-average target vs. hit the stop-loss first.
    """
    closes = close.values
    smas = sma.values
    rsis = rsi.values
    n = len(closes)

    resolutions = []  # list of (days_to_resolve, hit_target: bool)

    # walk through history, skip the most recent bar (that's today's live signal)
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
        if direction == "LONG":
            stop = entry * (1 - STOP_LOSS_PCT)
        else:
            stop = entry * (1 + STOP_LOSS_PCT)

        for j in range(1, MAX_HOLD_DAYS + 1):
            price = closes[i + j]
            if direction == "LONG":
                if price >= target:
                    resolutions.append((j, True))
                    break
                if price <= stop:
                    resolutions.append((j, False))
                    break
            else:
                if price <= target:
                    resolutions.append((j, True))
                    break
                if price >= stop:
                    resolutions.append((j, False))
                    break
        # if neither hit within MAX_HOLD_DAYS, the instance is dropped (inconclusive)

    if not resolutions:
        return {"avg_hold_days": None, "win_rate": None, "sample_size": 0}

    wins = [r for r in resolutions if r[1]]
    win_rate = round(100 * len(wins) / len(resolutions), 0)
    avg_hold_days = round(sum(r[0] for r in wins) / len(wins), 1) if wins else None

    return {
        "avg_hold_days": avg_hold_days,
        "win_rate": win_rate,
        "sample_size": len(resolutions),
    }


def analyze_ticker(ticker: str, sector: str) -> dict | None:
    try:
        hist = yf.download(
            ticker,
            period=f"{LOOKBACK_DAYS}d",
            progress=False,
            auto_adjust=True,
        )
    except Exception:
        return None

    if hist.empty or len(hist) < SMA_PERIOD + 5:
        return None

    close = hist["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    avg_volume = hist["Volume"].tail(20).mean()
    if float(avg_volume) < MIN_AVG_VOLUME:
        return None

    last_volume = hist["Volume"].iloc[-1]
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

    pct_vs_sma = (last_close - last_sma) / last_sma  # positive = above, negative = below

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
    target = last_sma  # reversion target = back to the 20-day average

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
        return None  # skip — earnings gap risk would swamp the stop-loss

    risk_reward = reward / risk
    hold_stats = backtest_hold_time(close, sma20, rsi, direction)
    news = get_news_headlines(ticker)

    return {
        "ticker": ticker,
        "sector": sector,
        "direction": direction,
        "price": round(entry, 2),
        "rsi": round(last_rsi, 1),
        "pct_vs_sma20": round(pct_vs_sma * 100, 1),
        "touched_band": touched_band,
        "entry": round(entry, 2),
        "stop_loss": round(stop_loss, 2),
        "target": round(target, 2),
        "risk_reward": round(risk_reward, 2),
        "avg_volume": int(avg_volume),
        "volume_ratio": round(volume_ratio, 2),
        "volume_confirmed": volume_confirmed,
        "avg_hold_days": hold_stats["avg_hold_days"],
        "hist_win_rate": hold_stats["win_rate"],
        "hist_sample_size": hold_stats["sample_size"],
        "days_to_earnings": days_to_earnings,
        "news": news,
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
TRACK_RECORD_COLUMNS = [
    "id", "date_flagged", "ticker", "direction", "entry", "stop", "target",
    "status", "date_resolved", "days_held", "exit_price",
]


def load_track_record() -> pd.DataFrame:
    try:
        return pd.read_csv(TRACK_RECORD_PATH)
    except FileNotFoundError:
        return pd.DataFrame(columns=TRACK_RECORD_COLUMNS)


def resolve_open_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Check every still-open logged position against current price and
    mark it win/loss/expired if it's been resolved since it was logged."""
    open_rows = df[df["status"] == "open"]
    for idx, row in open_rows.iterrows():
        try:
            recent = yf.download(
                row["ticker"], period="5d", progress=False, auto_adjust=True
            )
            if recent.empty:
                continue
            current_price = float(recent["Close"].iloc[-1])
        except Exception:
            continue

        date_flagged = dt.date.fromisoformat(row["date_flagged"])
        days_held = (dt.date.today() - date_flagged).days

        hit_target = (
            current_price >= row["target"] if row["direction"] == "LONG"
            else current_price <= row["target"]
        )
        hit_stop = (
            current_price <= row["stop"] if row["direction"] == "LONG"
            else current_price >= row["stop"]
        )

        if hit_target:
            df.loc[idx, ["status", "date_resolved", "days_held", "exit_price"]] = [
                "win", dt.date.today().isoformat(), days_held, current_price
            ]
        elif hit_stop:
            df.loc[idx, ["status", "date_resolved", "days_held", "exit_price"]] = [
                "loss", dt.date.today().isoformat(), days_held, current_price
            ]
        elif days_held > MAX_HOLD_DAYS:
            df.loc[idx, ["status", "date_resolved", "days_held", "exit_price"]] = [
                "expired", dt.date.today().isoformat(), days_held, current_price
            ]
    return df


def append_new_hits(df: pd.DataFrame, hits: list[dict]) -> pd.DataFrame:
    today = dt.date.today().isoformat()
    start_id = (df["id"].max() + 1) if len(df) else 1
    new_rows = []
    for i, h in enumerate(hits):
        new_rows.append({
            "id": start_id + i,
            "date_flagged": today,
            "ticker": h["ticker"],
            "direction": h["direction"],
            "entry": h["entry"],
            "stop": h["stop_loss"],
            "target": h["target"],
            "status": "open",
            "date_resolved": "",
            "days_held": "",
            "exit_price": "",
        })
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    return df


def summarize_track_record(df: pd.DataFrame) -> dict:
    resolved = df[df["status"].isin(["win", "loss", "expired"])]
    if resolved.empty:
        return {"n": 0, "win_rate": None, "avg_hold_days": None}
    wins = resolved[resolved["status"] == "win"]
    win_rate = round(100 * len(wins) / len(resolved), 0)
    avg_hold = round(wins["days_held"].astype(float).mean(), 1) if len(wins) else None
    return {"n": len(resolved), "win_rate": win_rate, "avg_hold_days": avg_hold}


def update_track_record(hits: list[dict]) -> dict:
    df = load_track_record()
    df = resolve_open_positions(df)
    df = append_new_hits(df, hits)
    import os
    os.makedirs("reports", exist_ok=True)
    df.to_csv(TRACK_RECORD_PATH, index=False)
    return summarize_track_record(df)


def build_report(hits: list[dict], track_summary: dict) -> str:
    today = dt.date.today().isoformat()
    lines = [f"# Mean-Reversion Screener — {today}", ""]

    macro_warnings = check_upcoming_macro_events()
    if macro_warnings:
        lines.append("**Macro event warning:**")
        for w in macro_warnings:
            lines.append(f"- {w} — expect elevated volatility, size down accordingly")
        lines.append("")

    if track_summary["n"] > 0:
        lines.append(
            f"**Live track record:** {track_summary['n']} resolved picks, "
            f"{track_summary['win_rate']:.0f}% win rate, "
            f"avg {track_summary['avg_hold_days']} days held on winners. "
            f"(Small samples early on aren't statistically meaningful yet.)"
        )
        lines.append("")

    if not hits:
        lines.append("No candidates flagged today.")
        return "\n".join(lines)

    # sector concentration check
    sector_counts: dict[str, int] = {}
    for h in hits:
        sector_counts[h["sector"]] = sector_counts.get(h["sector"], 0) + 1
    crowded = {s: c for s, c in sector_counts.items() if c >= SECTOR_WARNING_COUNT}
    if crowded:
        lines.append("**Sector concentration warning:**")
        for s, c in crowded.items():
            lines.append(
                f"- {c} of today's picks are in {s} — likely one shared "
                f"sector move, not {c} independent opportunities"
            )
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
        earnings = (
            f"{h['days_to_earnings']}d" if h["days_to_earnings"] is not None else "unknown"
        )
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
        "the 20-day average — high volume on the signal day usually means "
        "real capitulation/blow-off rather than a quiet drift that could "
        "just as easily reverse again. "
        "This is a mechanical screen only — check the news below before "
        "acting. Not financial advice."
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


def main():
    print("Fetching S&P 500 ticker list...")
    ticker_records = get_sp500_tickers()
    print(f"Scanning {len(ticker_records)} tickers...")
    hits = run_screen(ticker_records)

    print("Updating track record...")
    track_summary = update_track_record(hits)

    report = build_report(hits, track_summary)

    today = dt.date.today().isoformat()
    out_path = f"reports/{today}.md"
    import os
    os.makedirs("reports", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report)

    # Also always update a "latest.md" for easy viewing
    with open("reports/latest.md", "w") as f:
        f.write(report)

    print(f"Report written to {out_path}")
    print(report)


if __name__ == "__main__":
    main()
