"""The few indicators the scanner reads, computed on OHLC frames.

Kept small and explicit rather than pulled from a library so every number in a
setup's reasoning can be traced back to one function here.
"""

from __future__ import annotations

import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's average true range."""
    prev_close = frame["close"].shift(1)
    true_range = pd.concat([
        frame["high"] - frame["low"],
        (frame["high"] - prev_close).abs(),
        (frame["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's relative strength index, 0–100."""
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, float("nan"))
    return (100 - 100 / (1 + rs)).fillna(100.0)


def swing_lows(frame: pd.DataFrame, wing: int = 3, intact: bool = True) -> list[float]:
    """Lows lower than the ``wing`` bars on each side, oldest first.

    The last ``wing`` bars cannot be confirmed yet and are never swings. With
    ``intact``, a low that a later bar closed below is dropped: broken support
    is no longer support.
    """
    lows = frame["low"].to_numpy()
    closes = frame["close"].to_numpy()
    out = []
    for i in range(wing, len(lows) - wing):
        window = lows[i - wing:i + wing + 1]
        if lows[i] == window.min() and (window == lows[i]).sum() == 1:
            if intact and (closes[i + 1:] < lows[i]).any():
                continue
            out.append(float(lows[i]))
    return out


def swing_highs(frame: pd.DataFrame, wing: int = 3, intact: bool = True) -> list[float]:
    """Highs higher than the ``wing`` bars on each side, oldest first.

    With ``intact``, a high that a later bar closed above is dropped.
    """
    highs = frame["high"].to_numpy()
    closes = frame["close"].to_numpy()
    out = []
    for i in range(wing, len(highs) - wing):
        window = highs[i - wing:i + wing + 1]
        if highs[i] == window.max() and (window == highs[i]).sum() == 1:
            if intact and (closes[i + 1:] > highs[i]).any():
                continue
            out.append(float(highs[i]))
    return out


def touches(frame: pd.DataFrame, level: float, tolerance: float) -> int:
    """How many separate visits price made to within ``tolerance`` of ``level``.

    A level price has come back to many times is one the market respects; this
    is the scanner's measure of a support or resistance zone's quality. A run of
    consecutive bars sitting on the level counts as one visit, so a slow drift
    along a line does not look like repeated tests of it.
    """
    near = (frame["low"] <= level + tolerance) & (frame["high"] >= level - tolerance)
    starts = near & ~near.shift(1, fill_value=False)
    return int(starts.sum())
