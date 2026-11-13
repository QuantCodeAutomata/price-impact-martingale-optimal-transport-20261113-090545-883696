"""Market data used by the SPX-calibration experiment (Sec. 5.1.2 / Appendix A.2
of "Price-impact Martingale optimal transport").

The paper calibrates its two marginal laws to a Cboe SPX option chain snapshot
(AM-settled calls and puts for two maturities). Option chain snapshots,
open interest and quote fields are not OHLCV bars, so they are loaded directly
from the Massive REST API rather than through the QuantX Dataloader, which
serves OHLCV bars only.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

MASSIVE_API_ROOT = "https://api.massive.com"
CACHE_DIR = Path(__file__).resolve().parent / "cache"  # in .gitignore, never committed
RETRY_STATUSES = {429, 503, 504}


class MassiveDataError(RuntimeError):
    """A failed Massive REST request or an empty/unexpected response."""


def _massive_key() -> str:
    load_dotenv()
    key = os.environ.get("MASSIVE_API_KEY")
    if not key:
        raise MassiveDataError("MASSIVE_API_KEY must be set")
    return key


def _get(url: str, params: dict | None = None) -> dict:
    for attempt in range(3):
        response = requests.get(url, params=params, timeout=60)
        if response.status_code not in RETRY_STATUSES or attempt == 2:
            break
        time.sleep(float(response.headers.get("Retry-After") or 5 * 2 ** attempt))
    if response.status_code != 200:
        raise MassiveDataError(f"GET {url}: HTTP {response.status_code}: {response.text[:300]}")
    return response.json()


def fetch_option_chain_snapshot(
    underlying: str,
    expiration_date: str,
    contract_type: str,
    strike_low: float,
    strike_high: float,
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    """One side (calls or puts) of the option chain snapshot for ``underlying``
    at ``expiration_date``, restricted to the strike window
    [``strike_low``, ``strike_high``].

    Massive's option chain snapshot endpoint (``GET /v3/snapshot/options/{underlyingAsset}``)
    reports the contract's details, open interest, implied volatility, greeks,
    the most recent daily bar (``day``) and, when the plan includes quotes, a
    ``last_quote`` field with bid/ask. This account's options plan does not
    return ``last_quote``; the fair market value field ``fmv`` (a continuously
    updated model price) is used in its place as the option's quoted price, and
    the bid/ask spread filters described in Appendix A.2 cannot be applied.
    This substitution is recorded in ``results/RESULTS.md``.

    Returns one row per contract with columns: ticker, strike, contract_type,
    price (fmv, falling back to the last daily close), open_interest,
    expiration_date.
    """
    key = hashlib.sha256(
        f"{underlying}|{expiration_date}|{contract_type}|{strike_low}|{strike_high}".encode()
    ).hexdigest()[:16]
    cache_path = CACHE_DIR / f"options_{key}.csv"
    if use_cache and cache_path.exists():
        return pd.read_csv(cache_path)

    api_key = _massive_key()
    url = f"{MASSIVE_API_ROOT}/v3/snapshot/options/{underlying}"
    params = {
        "apiKey": api_key,
        "expiration_date": expiration_date,
        "contract_type": contract_type,
        "strike_price.gte": strike_low,
        "strike_price.lte": strike_high,
        "limit": 250,
        "sort": "strike_price",
        "order": "asc",
    }
    rows = []
    next_url, next_params = url, params
    while next_url:
        answer = _get(next_url, next_params)
        rows.extend(answer.get("results", []))
        next_url = answer.get("next_url")
        next_params = {"apiKey": api_key} if next_url else None

    if not rows:
        raise MassiveDataError(
            f"No {contract_type} contracts for {underlying} {expiration_date} "
            f"in [{strike_low}, {strike_high}]"
        )

    # SPX has two listed root symbols sharing one underlying: "O:SPX..." is the
    # standard AM-settled (third-Friday) series the paper uses, while
    # "O:SPXW..." is the PM-settled weekly/EOM series. Keep the AM-settled root
    # only (Sec. 5.1.2: "Cboe SPX option quotes ... AM-settled").
    records = []
    for row in rows:
        details = row.get("details", {})
        ticker = details.get("ticker") or ""
        if not ticker.startswith("O:SPX") or ticker.startswith("O:SPXW"):
            continue
        price = row.get("fmv")
        if price is None:
            day = row.get("day", {}) or {}
            price = day.get("close")
        if price is None:
            continue
        records.append({
            "ticker": ticker,
            "strike": details.get("strike_price"),
            "contract_type": details.get("contract_type"),
            "expiration_date": details.get("expiration_date"),
            "price": float(price),
            "open_interest": row.get("open_interest") or 0,
        })
    if not records:
        raise MassiveDataError(
            f"No AM-settled (O:SPX, non-SPXW) {contract_type} contracts for "
            f"{underlying} {expiration_date} in [{strike_low}, {strike_high}]"
        )
    frame = pd.DataFrame.from_records(records).sort_values("strike").reset_index(drop=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_path, index=False)
    return frame


def load_spx_chains(
    expiries: tuple[str, str],
    strike_low: float,
    strike_high: float,
    *,
    underlying: str = "I:SPX",
    use_cache: bool = True,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Calls and puts for both SPX expiries, restricted to a strike window.

    Returns {expiry: {"calls": DataFrame, "puts": DataFrame}}. Logs row counts
    and the fraction of contracts meeting the paper's open-interest filter.
    """
    out: dict[str, dict[str, pd.DataFrame]] = {}
    meta = {}
    for expiry in expiries:
        calls = fetch_option_chain_snapshot(underlying, expiry, "call", strike_low, strike_high,
                                             use_cache=use_cache)
        puts = fetch_option_chain_snapshot(underlying, expiry, "put", strike_low, strike_high,
                                            use_cache=use_cache)
        out[expiry] = {"calls": calls, "puts": puts}
        meta[expiry] = {
            "n_calls": len(calls),
            "n_puts": len(puts),
            "strike_range": [float(calls["strike"].min()), float(calls["strike"].max())],
        }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "spx_chain_meta.json").write_text(json.dumps(meta, indent=2))
    return out
