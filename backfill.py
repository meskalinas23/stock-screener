"""
Backfill + strategy lab.

Replays the screener rules over past price history (baseline), then tests ideas on the
same signals: reversal confirmation, volatility-based stops, market-wide selling, and how
the stock fell. Writes:
  reports/backtest_record.csv   (baseline signals, same format as the live track record)
  reports/backtest_lab.csv      (every signal with its features and each variant's result)
  docs/track.html               (track record page)
  docs/lab.html                 (strategy lab: what separates winners from losers)

Run it manually from GitHub Actions. It is a backtest: only today's S&P 500 members and
no earnings filter, so results are flattering. Live results are kept separate.
"""

import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import yfinance as yf

import screener as S

HISTORY_PERIOD = "5y"
SIGNAL_YEARS_BARS = 504        # only create signals for roughly the last 2 years
WIN_RATE_WINDOW = 345          # same amount of history the live screen uses (~500 calendar days)
MIN_BARS_BEFORE_SIGNAL = 260   # need 200-day average and enough history for the win rate
ATR_PERIOD = 14
ATR_MULT = 2.0                 # volatility stop = 2 x average daily range
CONFIRM_WINDOW = 3             # bars allowed to wait for a bounce
TEST_START = "2026-01-01"      # signals from here on are the "unseen" test period

LAB_COLUMNS = [
    "ticker", "date", "direction", "probability", "penalty_reasons",
    "same_day_frac", "cons_move", "vol_ratio", "trend_dist", "move5", "atr_pct",
    "base_status", "base_pnl", "atr_status", "atr_pnl",
    "conf_status", "conf_pnl", "confatr_status", "confatr_pnl",
]


def _raw_signal_flags(close_s, sma20, lower, upper, rsi):
    pct = (close_s - sma20) / sma20
    valid = sma20.notna() & rsi.notna() & lower.notna() & upper.notna()
    long_raw = valid & ((rsi < S.RSI_OVERSOLD) | (close_s < lower) | (pct <= -S.MIN_PCT_BELOW_SMA))
    short_raw = valid & ~long_raw & ((rsi > S.RSI_OVERBOUGHT) | (close_s > upper) | (pct >= S.MIN_PCT_ABOVE_SMA))
    return long_raw, short_raw


def _variant(direction, i, n, close, high, low, opn, entry_stop_base, target, atr_i, use_confirm, use_atr):
    """Returns (status, pnl_pct). status 'skipped' if the variant never entered."""
    entry_idx = i
    if use_confirm:
        entry_idx = None
        for j in range(i + 1, min(i + CONFIRM_WINDOW, n - 1) + 1):
            if direction == "LONG" and close[j] > high[j - 1]:
                entry_idx = j
                break
            if direction == "SHORT" and close[j] < low[j - 1]:
                entry_idx = j
                break
        if entry_idx is None:
            return "skipped", np.nan
    entry = float(close[entry_idx])
    if direction == "LONG":
        stop = entry - ATR_MULT * atr_i if use_atr else entry * (1 - S.STOP_LOSS_PCT)
        if target <= entry or stop <= 0:
            return "skipped", np.nan
    else:
        stop = entry + ATR_MULT * atr_i if use_atr else entry * (1 + S.STOP_LOSS_PCT)
        if target >= entry:
            return "skipped", np.nan
    bars = [(opn[j], high[j], low[j], close[j]) for j in range(entry_idx + 1, n)]
    result = S._walk_outcome(direction, entry, stop, target, bars)
    if result is None:
        return "skipped", np.nan
    status, k, exit_price = result
    return status, S._pnl_pct(direction, entry, exit_price)


def backfill_ticker(ticker: str, hist: pd.DataFrame, market_counts=None) -> list[dict]:
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
    prev_close = close_s.shift(1)
    tr = pd.concat([high_s - low_s, (high_s - prev_close).abs(), (low_s - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(ATR_PERIOD).mean()
    sma200 = close_s.rolling(200).mean()

    if market_counts is not None:
        long_raw, short_raw = _raw_signal_flags(close_s, sma20, lower, upper, rsi)
        for d, lg, sh in zip(close_s.index, long_raw.values, short_raw.values):
            c = market_counts[d]
            c[0] += int(lg)
            c[1] += int(sh)
            c[2] += 1

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
        pnl = S._pnl_pct(direction, entry, exit_price)
        date = pd.Timestamp(close_s.index[i])

        # ---- features at signal time
        sign = 1 if direction == "LONG" else -1
        cons = 0
        j = i
        while j > 0 and ((close[j] < close[j - 1]) if direction == "LONG" else (close[j] > close[j - 1])):
            cons += 1
            j -= 1
        atr_i = float(atr.iloc[i]) if not pd.isna(atr.iloc[i]) else np.nan
        feats = {
            "cons_move": cons,
            "vol_ratio": float(vol_s.iloc[i] / avg_vol.iloc[i]) if avg_vol.iloc[i] > 0 else np.nan,
            "trend_dist": float(sign * (close[i] - sma200.iloc[i]) / sma200.iloc[i] * 100) if not pd.isna(sma200.iloc[i]) else np.nan,
            "move5": float(-sign * (close[i] / close[i - 5] - 1) * 100),
            "atr_pct": float(atr_i / close[i] * 100) if not np.isnan(atr_i) else np.nan,
        }
        variants = {}
        if not np.isnan(atr_i) and atr_i > 0:
            variants["atr"] = _variant(direction, i, n, close, high, low, opn, None, target, atr_i, False, True)
            variants["conf"] = _variant(direction, i, n, close, high, low, opn, None, target, atr_i, True, False)
            variants["confatr"] = _variant(direction, i, n, close, high, low, opn, None, target, atr_i, True, True)
        else:
            variants = {"atr": ("skipped", np.nan), "conf": ("skipped", np.nan), "confatr": ("skipped", np.nan)}

        rows.append({
            "id": "", "date_flagged": date.date().isoformat(),
            "ticker": ticker, "direction": direction, "entry": str(round(entry, 2)),
            "stop": str(round(stop, 2)), "target": str(round(target, 2)),
            "probability": str(probability), "hist_win_rate": str(stats["win_rate"]),
            "penalty": str(pen), "reasons": "; ".join(reasons), "status": status,
            "date_resolved": pd.Timestamp(close_s.index[i + k]).date().isoformat(),
            "days_held": str(k), "exit_price": str(round(exit_price, 4)),
            "pnl_pct": str(pnl),
            "_date": date, "_feats": feats, "_variants": variants, "_pnl": pnl,
        })
        i += k + 1   # no new signal on this ticker until the previous trade is over
    return rows


# ---------------------------------------------------------------------------
# Lab page
# ---------------------------------------------------------------------------
def _stats(sub: pd.DataFrame, status_col: str, pnl_col: str) -> dict:
    d = sub[sub[status_col].isin(["win", "loss", "expired"])]
    n = len(d)
    if n == 0:
        return {"n": 0, "win": np.nan, "avg": np.nan, "ci": np.nan, "avg_win": np.nan, "avg_loss": np.nan}
    decided = d[d[status_col].isin(["win", "loss"])]
    pnl = d[pnl_col].astype(float)
    return {
        "n": n,
        "win": 100 * (decided[status_col] == "win").mean() if len(decided) else np.nan,
        "avg": pnl.mean(),
        "ci": 1.96 * pnl.std() / np.sqrt(n) if n > 1 else np.nan,
        "avg_win": d.loc[d[status_col] == "win", pnl_col].astype(float).mean(),
        "avg_loss": d.loc[d[status_col] == "loss", pnl_col].astype(float).mean(),
    }


def _f(x, d=1, pct=True, plus=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:+.{d}f}%" if plus else (f"{x:.{d}f}%" if pct else f"{x:.{d}f}")


def _cls(avg, ci, n):
    if n < 40 or np.isnan(avg):
        return "dim"
    return "good" if avg - ci > 0 else ("bad" if avg + ci < 0 else "")


def _feature_table(df, col, order, title, hint):
    rows = ""
    for label in order:
        sub = df[df[col] == label]
        a = _stats(sub, "base_status", "base_pnl")
        tr = _stats(sub[sub["date"] < TEST_START], "base_status", "base_pnl")
        te = _stats(sub[sub["date"] >= TEST_START], "base_status", "base_pnl")
        rows += (
            f"<tr><td>{label}</td><td>{a['n']}</td><td>{_f(a['win'], 0)}</td>"
            f"<td class='{_cls(a['avg'], a['ci'], a['n'])}'>{_f(a['avg'], 2, plus=True)} &plusmn;{_f(a['ci'], 2)}</td>"
            f"<td>{tr['n']}</td><td class='{_cls(tr['avg'], tr['ci'], tr['n'])}'>{_f(tr['avg'], 2, plus=True)}</td>"
            f"<td>{te['n']}</td><td class='{_cls(te['avg'], te['ci'], te['n'])}'>{_f(te['avg'], 2, plus=True)}</td></tr>"
        )
    return (f"<h3>{title}</h3><p class='muted'>{hint}</p><table><thead><tr><th>Bucket</th><th>Trades</th><th>Win rate</th>"
            f"<th>Avg P/L per trade (&plusmn; margin)</th><th>Train n</th><th>Train avg</th><th>Test n</th><th>Test avg</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>")


def build_lab_page(lab: pd.DataFrame):
    lab = lab.copy()
    lab["same_day"] = pd.cut(lab["same_day_frac"], [-0.001, 0.02, 0.05, 0.10, 1.0],
                             labels=["under 2% of stocks", "2-5%", "5-10%", "over 10%"]).astype(str)
    lab["spy_trend"] = np.where(lab["spy_above_200"] == 1, "S&P 500 above 200d avg", "S&P 500 below 200d avg")
    lab["spy_20d"] = pd.cut(lab["spy_ret20"], [-1, -0.05, 0, 0.05, 1],
                            labels=["S&P down over 5% (20d)", "S&P down 0-5%", "S&P up 0-5%", "S&P up over 5%"]).astype(str)
    lab["cons"] = pd.cut(lab["cons_move"], [-1, 1, 2, 3, 100], labels=["0-1 days", "2 days", "3 days", "4+ days"]).astype(str)
    lab["vol"] = pd.cut(lab["vol_ratio"], [-1, 1, 1.5, 2.5, 1000], labels=["under 1x", "1-1.5x", "1.5-2.5x", "over 2.5x"]).astype(str)
    lab["trend"] = pd.cut(lab["trend_dist"], [-1000, -10, 0, 10, 1000],
                          labels=["against trend, over 10% away", "against trend, 0-10%", "with trend, 0-10%", "with trend, over 10%"]).astype(str)
    lab["m5"] = pd.cut(lab["move5"], [-1000, 3, 6, 10, 1000], labels=["under 3%", "3-6%", "6-10%", "over 10%"]).astype(str)
    lab["atr"] = pd.cut(lab["atr_pct"], [-1, 1.5, 2.5, 4, 100], labels=["calm (under 1.5%)", "1.5-2.5%", "2.5-4%", "wild (over 4%)"]).astype(str)
    lab["swing"] = np.where(lab["penalty_reasons"].fillna("").str.contains("swing"), "broke swing level", "no break")
    lab["t200"] = np.where(lab["penalty_reasons"].fillna("").str.contains("200-day"), "wrong side of 200d", "ok")
    lab["lhll"] = np.where(lab["penalty_reasons"].fillna("").str.contains("lower highs|higher highs"), "highs/lows against trade", "ok")
    lab["pbucket"] = pd.cut(lab["probability"], [-1, 49, 59, 69, 100], labels=["below 50%", "50-59%", "60-69%", "70%+"]).astype(str)

    # --- variants
    var_rows = ""
    for name, pre, text in [("Baseline (today's rules)", "base", "Buy at signal close, 5% stop, target = 20-day avg"),
                            ("Volatility stop", "atr", f"Same entry, stop = {ATR_MULT:g} x average daily range"),
                            ("Wait for bounce", "conf", f"Enter only after a close above the previous day's high (within {CONFIRM_WINDOW} days), 5% stop"),
                            ("Bounce + volatility stop", "confatr", "Both of the above")]:
        a = _stats(lab, f"{pre}_status", f"{pre}_pnl")
        te = _stats(lab[lab["date"] >= TEST_START], f"{pre}_status", f"{pre}_pnl")
        skipped = int((lab[f"{pre}_status"] == "skipped").sum())
        var_rows += (
            f"<tr><td><b>{name}</b><br><span class='muted'>{text}</span></td><td>{a['n']}</td><td>{skipped}</td><td>{_f(a['win'], 0)}</td>"
            f"<td>{_f(a['avg_win'], 1, plus=True)}</td><td>{_f(a['avg_loss'], 1, plus=True)}</td>"
            f"<td class='{_cls(a['avg'], a['ci'], a['n'])}'>{_f(a['avg'], 2, plus=True)} &plusmn;{_f(a['ci'], 2)}</td>"
            f"<td>{te['n']}</td><td class='{_cls(te['avg'], te['ci'], te['n'])}'>{_f(te['avg'], 2, plus=True)}</td></tr>"
        )
    variants = ("<h2>1. Which entry and stop rules work best?</h2>"
                "<p class='muted'>Same signals, different rules. Compare <b>average P/L per trade</b>, not win rate: a wider stop wins more often "
                "but loses more when it loses. Green = clearly positive, red = clearly negative, grey = too few trades to say.</p>"
                "<table><thead><tr><th>Rule set</th><th>Trades</th><th>Skipped</th><th>Win rate</th><th>Avg win</th><th>Avg loss</th>"
                "<th>Avg P/L per trade (&plusmn; margin)</th><th>Test n</th><th>Test avg P/L</th></tr></thead>"
                f"<tbody>{var_rows}</tbody></table>")

    parts = [variants, "<h2>2. What separates good setups from bad ones? (baseline rules)</h2>",
             f"<p class='muted'>Train = signals before {TEST_START}, Test = signals from then on. A feature is only believable if train and test agree. "
             "The margin is a rough 95% range: if it includes zero, the result may be luck.</p>"]
    parts.append(_feature_table(lab, "spy_trend", ["S&P 500 above 200d avg", "S&P 500 below 200d avg"], "Market trend", "Is the whole market in an uptrend or downtrend?"))
    parts.append(_feature_table(lab, "spy_20d", ["S&P down over 5% (20d)", "S&P down 0-5%", "S&P up 0-5%", "S&P up over 5%"], "Market momentum (last 20 days)", "Recent S&P 500 move."))
    parts.append(_feature_table(lab, "same_day", ["under 2% of stocks", "2-5%", "5-10%", "over 10%"], "Market-wide selling", "Share of all S&P 500 stocks giving a signal in the same direction that day. High = market-driven move."))
    parts.append(_feature_table(lab, "direction", ["LONG", "SHORT"], "Direction", "Buying oversold vs shorting overbought."))
    parts.append(_feature_table(lab, "cons", ["0-1 days", "2 days", "3 days", "4+ days"], "Consecutive days moving against the trade", "Slow grind vs sudden flush."))
    parts.append(_feature_table(lab, "m5", ["under 3%", "3-6%", "6-10%", "over 10%"], "Size of the 5-day move against the trade", "How violent the move was."))
    parts.append(_feature_table(lab, "vol", ["under 1x", "1-1.5x", "1.5-2.5x", "over 2.5x"], "Volume on signal day vs average", "High volume can mean capitulation."))
    parts.append(_feature_table(lab, "atr", ["calm (under 1.5%)", "1.5-2.5%", "2.5-4%", "wild (over 4%)"], "Stock volatility (average daily range)", "Calm vs wild stocks."))
    parts.append(_feature_table(lab, "trend", ["against trend, over 10% away", "against trend, 0-10%", "with trend, 0-10%", "with trend, over 10%"], "Distance from 200-day average", "'With trend' = a long above the 200d or a short below it."))
    parts.append("<h2>3. Do the structure penalties help?</h2><p class='muted'>If a flag really hurts, the 'flag' row should be worse than the 'no flag' row.</p>")
    parts.append(_feature_table(lab, "swing", ["broke swing level", "no break"], "Broke the 60-day swing level", "-15 points today."))
    parts.append(_feature_table(lab, "t200", ["wrong side of 200d", "ok"], "Wrong side of the 200-day average", "-8 points today."))
    parts.append(_feature_table(lab, "lhll", ["highs/lows against trade", "ok"], "Highs and lows moving against the trade", "-7 points today."))
    parts.append(_feature_table(lab, "pbucket", ["below 50%", "50-59%", "60-69%", "70%+"], "Current probability number", "If this worked, results would improve going down the table."))

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Screener strategy lab</title>
<style>
 body {{ font-family: -apple-system, Arial, sans-serif; max-width: 1150px; margin: 20px auto; padding: 0 15px; }}
 table {{ border-collapse: collapse; width: 100%; margin: 8px 0 18px; font-size: 0.9em; }}
 th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }} th {{ background: #f5f5f5; }}
 .good {{ color: #0a7d2c; font-weight: bold; }} .bad {{ color: #c0392b; font-weight: bold; }} .dim {{ color: #999; }}
 .muted {{ color: #666; font-size: 0.9em; }} h3 {{ margin-bottom: 2px; }}
</style></head><body>
<p><a href="track.html">&larr; Track record</a> &middot; <a href="index.html">Today's setups</a></p>
<h1>Strategy lab</h1>
<p class="muted">{len(lab)} backtest signals, {lab['date'].min()} to {lab['date'].max()}. Backtest only: today's S&amp;P 500 members, no earnings filter, no trading costs.
Signals on the same day are related, so the real certainty is lower than the margins suggest.</p>
{''.join(parts)}
</body></html>"""
    os.makedirs("docs", exist_ok=True)
    with open("docs/lab.html", "w") as f:
        f.write(html)
    print("Wrote docs/lab.html")


def main():
    limit = int(os.environ.get("BACKFILL_MAX_TICKERS", "0") or 0)
    records = S.get_sp500_tickers()
    if limit:
        records = records[:limit]
    print(f"Backfilling {len(records)} tickers...")

    market_counts = defaultdict(lambda: [0, 0, 0])   # date -> [long signals, short signals, stocks with data]
    all_rows = []
    for k, rec in enumerate(records):
        try:
            hist = yf.download(rec["ticker"], period=HISTORY_PERIOD, progress=False, auto_adjust=True)
        except Exception:
            continue
        if hist is None or hist.empty:
            continue
        all_rows.extend(backfill_ticker(rec["ticker"], hist, market_counts))
        if k % 25 == 0:
            print(f"...{k}/{len(records)} tickers, {len(all_rows)} signals so far")
        time.sleep(S.REQUEST_PAUSE_SEC)

    # ---- baseline record (same format as live)
    df = pd.DataFrame([{c: r[c] for c in S.TRACK_RECORD_COLUMNS} for r in all_rows], columns=S.TRACK_RECORD_COLUMNS)
    if len(df):
        order = sorted(range(len(all_rows)), key=lambda x: all_rows[x]["date_flagged"])
        all_rows = [all_rows[x] for x in order]
        df = pd.DataFrame([{c: r[c] for c in S.TRACK_RECORD_COLUMNS} for r in all_rows], columns=S.TRACK_RECORD_COLUMNS)
        df["id"] = [str(x + 1) for x in range(len(df))]
    os.makedirs("reports", exist_ok=True)
    df.to_csv(S.BACKTEST_PATH, index=False)
    print(f"Wrote {len(df)} backtest signals to {S.BACKTEST_PATH}")
    S.write_track_page()

    if not all_rows:
        return

    # ---- market features (S&P 500 via SPY)
    spy = yf.download("SPY", period=HISTORY_PERIOD, progress=False, auto_adjust=True)
    spy_close = S.series_of(spy, "Close").astype(float)
    spy_feat = pd.DataFrame({
        "spy_above_200": (spy_close > spy_close.rolling(200).mean()).astype(float),
        "spy_ret20": spy_close / spy_close.shift(20) - 1,
    })
    spy_feat.index = pd.to_datetime(spy_feat.index)

    lab_rows = []
    for r in all_rows:
        d = r["_date"]
        c = market_counts[d]
        frac_den = c[2] if c[2] else 1
        frac = (c[0] if r["direction"] == "LONG" else c[1]) / frac_den
        v = r["_variants"]
        lab_rows.append({
            "ticker": r["ticker"], "date": r["date_flagged"], "direction": r["direction"],
            "probability": int(r["probability"]), "penalty_reasons": r["reasons"],
            "same_day_frac": frac, **r["_feats"],
            "base_status": r["status"], "base_pnl": r["_pnl"],
            "atr_status": v["atr"][0], "atr_pnl": v["atr"][1],
            "conf_status": v["conf"][0], "conf_pnl": v["conf"][1],
            "confatr_status": v["confatr"][0], "confatr_pnl": v["confatr"][1],
            "_ts": d,
        })
    lab = pd.DataFrame(lab_rows)
    sf = spy_feat.reindex(pd.DatetimeIndex(lab["_ts"]), method="ffill")
    lab["spy_above_200"] = sf["spy_above_200"].values
    lab["spy_ret20"] = sf["spy_ret20"].values
    lab = lab.drop(columns=["_ts"])
    lab.to_csv("reports/backtest_lab.csv", index=False)
    print(f"Wrote {len(lab)} rows to reports/backtest_lab.csv")
    build_lab_page(lab)


if __name__ == "__main__":
    main()
