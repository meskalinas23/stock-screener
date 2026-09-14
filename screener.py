"""
S&P 500 Mean-Reversion Screener
Scans S&P 500 stocks for oversold mean-reversion setups, and writes a
dated markdown/HTML report with entry/stop/target and risk-to-reward for each hit.

This does NOT place any trades. It only researches and reports. You decide.
"""

import datetime as dt
import time

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

SP500_LIST_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/"
    "master/data/constituents.csv"
)

# ---------------------------------------------------------------------------
# Macro event calendar — pulled live from a free public ForexFactory feed.
# No manual date entry needed. Filters to high-impact USD events only
# (FOMC, CPI, PPI, NFP, GDP, etc.) within MACRO_WARNING_DAYS.
# ---------------------------------------------------------------------------
FF_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
MACRO_WARNING_DAYS = 3
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


def check_upcoming_macro_events(days_ahead: int = MACRO_WARNING_DAYS) -> list[str]:
    events = fetch_macro_events()
    now = pd.Timestamp.now(tz="UTC")
    warnings = []
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
        if 0 <= delta_days <= days_ahead:
            when = "today" if delta_days == 0 else f"in {delta_days} day(s)"
            title = e.get("title", "Economic event")
            warnings.append(f"{title} ({e.get('country', '')}) on {event_time.date().isoformat()} ({when})")
    return warnings


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

    return {
        "ticker": ticker, "sector": sector, "direction": direction,
        "price": round(entry, 2), "rsi": round(last_rsi, 1),
        "pct_vs_sma20": round(pct_vs_sma * 100, 1), "touched_band": touched_band,
        "entry": round(entry, 2), "stop_loss": round(stop_loss, 2), "target": round(target, 2),
        "risk_reward": round(risk_reward, 2), "avg_volume": int(avg_volume),
        "volume_ratio": round(volume_ratio, 2), "volume_confirmed": volume_confirmed,
        "avg_hold_days": hold_stats["avg_hold_days"], "hist_win_rate": hold_stats["win_rate"],
        "hist_sample_size": hold_stats["sample_size"], "days_to_earnings": days_to_earnings, "news": news,
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
    open_rows = df[df["status"] == "open"]
    for idx, row in open_rows.iterrows():
        try:
            recent = yf.download(row["ticker"], period="5d", progress=False, auto_adjust=True)
            if recent.empty:
                continue
            current_price = float(recent["Close"].iloc[-1])
        except Exception:
            continue
        date_flagged = dt.date.fromisoformat(row["date_flagged"])
        days_held = (dt.date.today() - date_flagged).days
        hit_target = current_price >= row["target"] if row["direction"] == "LONG" else current_price <= row["target"]
        hit_stop = current_price <= row["stop"] if row["direction"] == "LONG" else current_price >= row["stop"]
        if hit_target:
            df.loc[idx, ["status", "date_resolved", "days_held", "exit_price"]] = ["win", dt.date.today().isoformat(), days_held, current_price]
        elif hit_stop:
            df.loc[idx, ["status", "date_resolved", "days_held", "exit_price"]] = ["loss", dt.date.today().isoformat(), days_held, current_price]
        elif days_held > MAX_HOLD_DAYS:
            df.loc[idx, ["status", "date_resolved", "days_held", "exit_price"]] = ["expired", dt.date.today().isoformat(), days_held, current_price]
    return df


def append_new_hits(df: pd.DataFrame, hits: list[dict]) -> pd.DataFrame:
    today = dt.date.today().isoformat()
    start_id = (df["id"].max() + 1) if len(df) else 1
    new_rows = []
    for i, h in enumerate(hits):
        new_rows.append({
            "id": start_id + i, "date_flagged": today, "ticker": h["ticker"], "direction": h["direction"],
            "entry": h["entry"], "stop": h["stop_loss"], "target": h["target"], "status": "open",
            "date_resolved": "", "days_held": "", "exit_price": "",
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


def build_html_report(hits: list[dict], track_summary: dict) -> str:
    today = dt.date.today().isoformat()

    macro_warnings = check_upcoming_macro_events()
    macro_html = ""
    if macro_warnings:
        items = "".join(f"<li>{w} &mdash; expect elevated volatility, size down accordingly</li>" for w in macro_warnings)
        macro_html = f"<div class='warning'><strong>Macro event warning:</strong><ul>{items}</ul></div>"

    track_html = ""
    if track_summary["n"] > 0:
        track_html = f"""
        <p class="track-record"><strong>Live track record:</strong>
        {track_summary['n']} resolved picks, {track_summary['win_rate']:.0f}% win rate,
        avg {track_summary['avg_hold_days']} days held on winners.
        (Small samples early on aren't statistically meaningful yet.)</p>
        """

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
                    <th>R:R</th><th>Avg Hold</th><th>Win Rate</th><th>Sample</th><th>Earnings</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
        """

        news_sections = ""
        for h in hits:
            if h["news"]:
                items = "".join(
                    f"<li><a href='{n['link']}'>{n['title']}</a></li>" if n["link"]
                    else f"<li>{n['title']}</li>"
                    for n in h["news"]
                )
            else:
                items = "<li>No recent headlines found &mdash; check manually before acting.</li>"
            news_sections += f"<h3>{h['ticker']}</h3><ul>{items}</ul>"
        news_html = f"<h2>News context</h2>{news_sections}"

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
        Stop-loss = 5% against the position. &#128293; means volume was {VOLUME_SPIKE_RATIO}x+ the 20-day average.
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


def write_html_report(hits: list[dict], track_summary: dict):
    import os
    today = dt.date.today().isoformat()
    os.makedirs("docs/reports", exist_ok=True)

    dated_html = build_html_report(hits, track_summary)
    with open(f"docs/reports/{today}.html", "w") as f:
        f.write(dated_html)

    archive_html = build_archive_list_html()
    main_html = dated_html.replace("</body>", f"{archive_html}</body>")
    with open("docs/index.html", "w") as f:
        f.write(main_html)

    print(f"Wrote docs/index.html and docs/reports/{today}.html")


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
    with open("reports/latest.md", "w") as f:
        f.write(report)

    write_html_report(hits, track_summary)

    print(f"Report written to {out_path}")
    print(report)


if __name__ == "__main__":
    main()
