import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Optional, Tuple

import httpx
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import starlette.status as status

logger = logging.getLogger("sats_converter")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BLOCKCHAIR_URL = "https://api.blockchair.com/bitcoin/stats"
COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price"
REQUEST_TIMEOUT = httpx.Timeout(5.0, connect=3.0)
CACHE_TTL_SECONDS = 20  # keyless CoinGecko API is rate-limited; cache briefly

FIAT_LIST = [
    "USD", "EUR", "JPY", "CAD", "AUD", "GBP", "PLN",
    "CHF", "HKD", "CNY", "SGD", "TWD", "THB", "KRW",
    "BRL", "RUB", "TRY",
]
FIAT_SET = {c.lower() for c in FIAT_LIST}

SATS_PER_BTC = 100_000_000


# ---------------------------------------------------------------------------
# Tiny in-memory TTL cache (per-process).
# ---------------------------------------------------------------------------

_cache: Dict[str, Tuple[float, object]] = {}


def _cache_get(key: str):
    entry = _cache.get(key)
    if entry is None:
        return None
    ts, value = entry
    if time.monotonic() - ts > CACHE_TTL_SECONDS:
        _cache.pop(key, None)
        return None
    return value


def _cache_set(key: str, value) -> None:
    _cache[key] = (time.monotonic(), value)


# ---------------------------------------------------------------------------
# External API calls
# ---------------------------------------------------------------------------

class UpstreamError(Exception):
    """Raised when an upstream API call fails or returns unexpected data."""


async def get_block_height(client: httpx.AsyncClient) -> Optional[int]:
    """Non-critical: page should still render if this fails."""
    cached = _cache_get("block_height")
    if cached is not None:
        return cached
    try:
        res = await client.get(BLOCKCHAIR_URL, timeout=REQUEST_TIMEOUT)
        res.raise_for_status()
        height = res.json()["data"]["best_block_height"]
        _cache_set("block_height", height)
        return height
    except (httpx.HTTPError, KeyError, TypeError) as e:
        logger.warning("get_block_height failed: %s", e)
        return None


async def coingecko_btc_fiat(client: httpx.AsyncClient, currency: str) -> Tuple[int, float]:
    """
    Return (unix_timestamp, btc_price) for BTC priced in `currency`
    (a 3-letter fiat code, case-insensitive, e.g. 'usd').
    """
    currency = currency.lower()
    if currency not in FIAT_SET:
        raise UpstreamError(f"Unsupported currency: {currency}")

    cache_key = f"btc_{currency}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    params = {
        "ids": "bitcoin",
        "vs_currencies": currency,
        "include_last_updated_at": "true",
    }
    try:
        response = await client.get(COINGECKO_URL, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        rate = data["bitcoin"][currency]
        timestamp = data["bitcoin"]["last_updated_at"]
    except (httpx.HTTPError, KeyError, TypeError) as e:
        raise UpstreamError(f"Failed to fetch BTC/{currency.upper()} rate") from e

    result = (timestamp, rate)
    _cache_set(cache_key, result)
    return result


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http_client = httpx.AsyncClient()
    yield
    await app.state.http_client.aclose()


app = FastAPI(
    title="sats converter",
    description="simple web app to convert fiat to btc",
    version="0.0.1 alpha",
    contact={"name": "bitkarrot", "url": "http://github.com/bitkarrot"},
    license_info={"name": "MIT License", "url": "https://mit-license.org/"},
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5000",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates/")


def _format_fiat(amount: float) -> str:
    return "{:,.2f}".format(float(amount))


async def _render_index(request: Request, currency: str):
    client: httpx.AsyncClient = app.state.http_client
    currency = currency.upper()

    (timestamp, rate), height = await asyncio.gather(
        coingecko_btc_fiat(client, currency),
        get_block_height(client),
    )

    return templates.TemplateResponse(
        "index.html",
        context={
            "request": request,
            "title": "Sats Converter",
            "fiat": _format_fiat(rate),
            "fiattype": currency,
            "fiatlist": FIAT_LIST,
            "satsamt": SATS_PER_BTC,
            "moscow": int(SATS_PER_BTC / rate),
            "blockheight": height,
            "lastupdated": timestamp,
        },
    )


async def _render_index_or_502(request: Request, currency: str):
    try:
        return await _render_index(request, currency)
    except UpstreamError as e:
        logger.error(e)
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/")
async def initial_page(request: Request):
    return await _render_index_or_502(request, "USD")


@app.get("/btc")
async def redirectpage(request: Request):
    return RedirectResponse("/", status_code=status.HTTP_302_FOUND)


@app.post("/btc")
async def submit_form(request: Request, selected: str = Form(...)):
    selected = (selected or "").upper()
    if selected not in FIAT_LIST:
        raise HTTPException(status_code=400, detail=f"Unsupported currency: {selected}")
    return await _render_index_or_502(request, selected)


@app.get("/rate")
async def get_rate(pair: str):
    """
    Retrieve the exchange rate between BTC/SAT and any supported fiat currency.

    Parameters:
        - **pair** (str): 6-character pair, e.g. btcusd, usdbtc, sathkd, eursat

    Returns:
        dict: {"rate": "1200.35"}
    """
    pair = pair.lower().strip()
    if len(pair) != 6:
        return {"error": "Currency pair must be 6 characters, e.g. 'btcusd'"}

    left, right = pair[:3], pair[3:]

    if left == "btc":
        currency, inverse, sat = right, False, False
    elif right == "btc":
        currency, inverse, sat = left, True, False
    elif left == "sat":
        currency, inverse, sat = right, False, True
    elif right == "sat":
        currency, inverse, sat = left, True, True
    else:
        return {"error": "Pair must include 'btc' or 'sat', e.g. 'btcusd' or 'sathkd'"}

    if currency not in FIAT_SET:
        return {"error": f"Currency not found: {currency}"}

    try:
        client: httpx.AsyncClient = app.state.http_client
        # NOTE: fetch BTC-<currency> rate.
        _, rate = await coingecko_btc_fiat(client, currency)

        if not inverse and not sat:
            value = "%.2f" % rate
        elif inverse and not sat:
            value = format(1 / rate, ".8f")
        elif not inverse and sat:
            value = format(rate / SATS_PER_BTC, ".8f")
        else:
            value = format(SATS_PER_BTC / rate, ".2f")

        return {"rate": value}

    except UpstreamError as e:
        logger.error(e)
        return {"error": "Failed to fetch exchange rate"}