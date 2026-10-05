"""The scanner's default instruments and the price conventions for each.

A pip is the unit traders quote stops and spreads in: 0.0001 for most pairs,
0.01 for yen pairs. Metals have no formal pip; the values here follow the
common retail convention (gold 0.1, silver 0.01) so a spread or a stop reads
in the units a trader would expect.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InstrumentSpec:
    symbol: str          # EURUSD, XAUUSD
    pip: float           # size of one pip in price units
    decimals: int        # price precision for display and order prices

    @property
    def base(self) -> str:
        return self.symbol[:3]

    @property
    def quote(self) -> str:
        return self.symbol[3:]

    @property
    def is_metal(self) -> bool:
        return self.base in {"XAU", "XAG", "XPT", "XPD"}

    def pips(self, distance: float) -> float:
        return distance / self.pip

    def round_price(self, price: float) -> float:
        return round(price, self.decimals)


MAJORS = ("EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD")
CROSSES = ("EURJPY", "GBPJPY", "EURGBP", "AUDJPY", "EURAUD")
METALS = ("XAUUSD", "XAGUSD")

DEFAULT_UNIVERSE = MAJORS + CROSSES + METALS

_METAL_SPECS = {
    "XAU": (0.1, 2),
    "XAG": (0.01, 3),
    "XPT": (0.1, 2),
    "XPD": (0.1, 2),
}


def spec_for(symbol: str) -> InstrumentSpec:
    """Pip size and precision for a six-letter pair such as ``GBPJPY``."""
    s = symbol.strip().upper().rstrip("+").replace("/", "").replace("_", "")
    if len(s) != 6 or not s.isalpha():
        raise ValueError(f"Not a forex or metal pair: {symbol!r}")
    if s[:3] in _METAL_SPECS:
        pip, decimals = _METAL_SPECS[s[:3]]
    elif s[3:] == "JPY":
        pip, decimals = 0.01, 3
    else:
        pip, decimals = 0.0001, 5
    return InstrumentSpec(symbol=s, pip=pip, decimals=decimals)
