"""
Currency conversion for displaying costs in the user's preferred currency.
Uses Frankfurter (frankfurter.dev), a free/keyless API backed by real
European Central Bank reference rates - no fabricated conversion factors.
Rates are cached in memory for an hour since they only update once a day;
if the API is ever unreachable, we fall back to the last cached rates, and
if we've never successfully fetched, fall back to showing USD unconverted
rather than guessing a rate.
"""
import time
import httpx

FRANKFURTER_URL = "https://api.frankfurter.dev/v1/latest"

SUPPORTED_CURRENCIES = {
    "USD": {"name": "US Dollar", "symbol": "$"},
    "INR": {"name": "Indian Rupee", "symbol": "₹"},
    "SGD": {"name": "Singapore Dollar", "symbol": "S$"},
    "MYR": {"name": "Malaysian Ringgit", "symbol": "RM"},
    "EUR": {"name": "Euro", "symbol": "€"},
    "GBP": {"name": "British Pound", "symbol": "£"},
}

_CACHE_TTL_SECONDS = 3600
# What prices are SHOWN in by default. Amounts are held in USD internally -
# the flight dataset and the hotel ADR tiers are dollar-denominated - so this
# changes presentation, never the arithmetic or the budget optimiser.
DEFAULT_CURRENCY = "INR"

_rates_cache = {"rates": {"USD": 1.0}, "fetched_at": 0.0}


def _refresh_rates():
    targets = ",".join(c for c in SUPPORTED_CURRENCIES if c != "USD")
    try:
        r = httpx.get(FRANKFURTER_URL, params={"from": "USD", "to": targets}, timeout=8)
        r.raise_for_status()
        data = r.json()
        rates = {"USD": 1.0, **data.get("rates", {})}
        _rates_cache["rates"] = rates
        _rates_cache["fetched_at"] = time.monotonic()
    except Exception as e:
        print(f"[currency] Exchange rate refresh failed, using cached/USD-only rates: {e}")


def get_rates() -> dict:
    if time.monotonic() - _rates_cache["fetched_at"] > _CACHE_TTL_SECONDS:
        _refresh_rates()
    return _rates_cache["rates"]


def convert(usd_amount: float, currency: str) -> float:
    if usd_amount is None:
        return usd_amount
    rate = get_rates().get(currency, 1.0)
    return round(usd_amount * rate, 2)


def to_usd(amount: float, currency: str) -> float:
    """Convert an amount the user TYPED in `currency` back to USD.

    Everything downstream - the budget optimiser, the hotel ADR tiers, the
    training data - is dollar-denominated, so a figure entered in a field
    labelled with the display currency has to come back across before it is
    used as a constraint. Without this, typing 2000 in a box marked INR was
    read as $2,000 and planned a trip worth about 1.9 lakh.
    """
    if not currency or currency == "USD":
        return float(amount)
    rate = get_rates().get(currency)
    if not rate:
        return float(amount)
    return round(float(amount) / rate, 2)


def format_money(usd_amount: float, currency: str) -> str:
    if usd_amount is None:
        return ""
    currency = currency if currency in SUPPORTED_CURRENCIES else DEFAULT_CURRENCY
    symbol = SUPPORTED_CURRENCIES[currency]["symbol"]
    converted = convert(usd_amount, currency)
    return f"{symbol}{converted:,.2f}"
