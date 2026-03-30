from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Variavel de ambiente obrigatoria ausente: {name}")
    return value


@dataclass
class AlpacaMarketDataConfig:
    api_key: str
    secret_key: str
    feed: str = "iex"

    @classmethod
    def from_env(cls) -> "AlpacaMarketDataConfig":
        return cls(
            api_key=require_env("ALPACA_API_KEY"),
            secret_key=require_env("ALPACA_SECRET_KEY"),
            feed=os.getenv("ALPACA_DATA_FEED", "iex"),
        )


class AlpacaHistoricalDataClient:
    def __init__(self, config: AlpacaMarketDataConfig):
        from alpaca.data.historical.stock import StockHistoricalDataClient

        self.config = config
        self.client = StockHistoricalDataClient(
            api_key=config.api_key,
            secret_key=config.secret_key,
        )

    def load_recent_bars(
        self,
        symbol: str,
        timeframe_amount: int,
        timeframe_unit: str,
        lookback_bars: int,
    ) -> pd.DataFrame:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        unit_normalized = timeframe_unit.strip().lower()
        mapping = {
            "minute": TimeFrameUnit.Minute,
            "min": TimeFrameUnit.Minute,
            "hour": TimeFrameUnit.Hour,
            "day": TimeFrameUnit.Day,
        }
        if unit_normalized not in mapping:
            raise ValueError(f"timeframe_unit invalido: {timeframe_unit}")

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(timeframe_amount, mapping[unit_normalized]),
            limit=lookback_bars,
            adjustment="raw",
            feed=self.config.feed,
        )
        bars = self.client.get_stock_bars(request).df
        if bars.empty:
            raise RuntimeError(f"Nenhum candle retornado para {symbol}")

        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.xs(symbol, level="symbol")

        bars = bars.reset_index().rename(columns={"timestamp": "date"})
        bars = bars.sort_values("date").drop_duplicates(subset=["date"]).set_index("date")
        return bars[["open", "high", "low", "close", "volume"]].astype(float)
