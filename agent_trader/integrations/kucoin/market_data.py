from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Variavel de ambiente obrigatoria ausente: {name}")
    return value


def normalize_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper().replace("-", "/")
    if "/" not in normalized:
        raise ValueError(
            f"symbol invalido para KuCoin: {symbol}. Use pares como BTC/USDT ou BTC-USDT."
        )
    return normalized


def split_symbol(symbol: str) -> tuple[str, str]:
    normalized = normalize_symbol(symbol)
    base_asset, quote_asset = normalized.split("/", 1)
    return base_asset, quote_asset


def build_kucoin_exchange(
    api_key: str = "",
    secret_key: str = "",
    passphrase: str = "",
    enable_rate_limit: bool = True,
):
    import ccxt

    params = {
        "enableRateLimit": enable_rate_limit,
        "options": {
            "adjustForTimeDifference": True,
        },
    }
    if api_key and secret_key and passphrase:
        params.update(
            {
                "apiKey": api_key,
                "secret": secret_key,
                "password": passphrase,
            }
        )
    return ccxt.kucoin(params)


def _to_ccxt_timeframe(
    timeframe_amount: int,
    timeframe_unit: str,
    supported_timeframes: dict[str, object],
) -> str:
    unit_normalized = timeframe_unit.strip().lower()
    suffix_map = {
        "minute": "m",
        "min": "m",
        "hour": "h",
        "day": "d",
        "week": "w",
        "month": "M",
    }
    suffix = suffix_map.get(unit_normalized)
    if suffix is None:
        raise ValueError(f"timeframe_unit invalido para KuCoin: {timeframe_unit}")

    ccxt_timeframe = f"{timeframe_amount}{suffix}"
    if ccxt_timeframe not in supported_timeframes:
        supported = ", ".join(sorted(supported_timeframes.keys()))
        raise ValueError(
            f"timeframe {ccxt_timeframe} nao suportado pela KuCoin no ccxt. Suportados: {supported}"
        )
    return ccxt_timeframe


@dataclass
class KuCoinMarketDataConfig:
    api_key: str = ""
    secret_key: str = ""
    passphrase: str = ""
    enable_rate_limit: bool = True

    @classmethod
    def from_env(cls) -> "KuCoinMarketDataConfig":
        return cls(
            api_key=os.getenv("KUCOIN_API_KEY", ""),
            secret_key=os.getenv("KUCOIN_SECRET_KEY", ""),
            passphrase=os.getenv("KUCOIN_PASSPHRASE", ""),
            enable_rate_limit=os.getenv("KUCOIN_ENABLE_RATE_LIMIT", "true").lower() == "true",
        )


class KuCoinHistoricalDataClient:
    def __init__(self, config: KuCoinMarketDataConfig):
        self.config = config
        self.client = build_kucoin_exchange(
            api_key=config.api_key,
            secret_key=config.secret_key,
            passphrase=config.passphrase,
            enable_rate_limit=config.enable_rate_limit,
        )

    def load_recent_bars(
        self,
        symbol: str,
        timeframe_amount: int,
        timeframe_unit: str,
        lookback_bars: int,
    ) -> pd.DataFrame:
        normalized_symbol = normalize_symbol(symbol)
        timeframe = _to_ccxt_timeframe(
            timeframe_amount=timeframe_amount,
            timeframe_unit=timeframe_unit,
            supported_timeframes=self.client.timeframes,
        )
        rows = self.client.fetch_ohlcv(
            normalized_symbol,
            timeframe=timeframe,
            limit=lookback_bars,
        )
        if not rows:
            raise RuntimeError(f"Nenhum candle retornado para {normalized_symbol}")

        bars = pd.DataFrame(
            rows,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        bars["date"] = pd.to_datetime(bars["timestamp"], unit="ms", utc=True)
        bars = bars.drop(columns=["timestamp"]).set_index("date").sort_index()
        return bars[["open", "high", "low", "close", "volume"]].astype(float)

    def fetch_last_price(self, symbol: str) -> float:
        ticker = self.client.fetch_ticker(normalize_symbol(symbol))
        price = ticker.get("last") or ticker.get("close")
        if price is None:
            raise RuntimeError(f"Nao foi possivel obter o ultimo preco para {symbol}")
        return float(price)
