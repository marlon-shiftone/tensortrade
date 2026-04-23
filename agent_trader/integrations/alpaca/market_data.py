from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

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
    request_retries: int = 3
    retry_delay_seconds: float = 1.0

    @classmethod
    def from_env(cls) -> "AlpacaMarketDataConfig":
        return cls(
            api_key=require_env("ALPACA_API_KEY"),
            secret_key=require_env("ALPACA_SECRET_KEY"),
            feed=os.getenv("ALPACA_DATA_FEED", "iex"),
            request_retries=int(os.getenv("ALPACA_DATA_RETRIES", "3")),
            retry_delay_seconds=float(os.getenv("ALPACA_DATA_RETRY_DELAY_SECONDS", "1.0")),
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
        from requests.exceptions import RequestException
        from alpaca.common.enums import Sort
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

        end = datetime.now(timezone.utc)
        if unit_normalized in {"minute", "min"}:
            trading_days = max(1, int((lookback_bars / max(1, timeframe_amount)) / 390) + 1)
            calendar_days = max(7, trading_days * 7)
        elif unit_normalized == "hour":
            trading_days = max(1, int((lookback_bars / max(1, timeframe_amount)) / 7) + 1)
            calendar_days = max(10, trading_days * 7)
        else:
            calendar_days = max(lookback_bars * 3, 30)
        start = end - timedelta(days=calendar_days)

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(timeframe_amount, mapping[unit_normalized]),
            start=start,
            end=end,
            limit=lookback_bars,
            adjustment="raw",
            feed=self.config.feed,
            sort=Sort.DESC,
        )
        attempts = max(1, int(self.config.request_retries))
        bars = None
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                bars = self.client.get_stock_bars(request).df
                last_error = None
                break
            except RequestException as exc:
                last_error = exc
                if attempt >= attempts:
                    break
                time.sleep(max(0.0, float(self.config.retry_delay_seconds)))

        if last_error is not None:
            raise RuntimeError(f"Falha ao buscar candles para {symbol}: {last_error}") from last_error
        if bars is None:
            raise RuntimeError(f"Falha ao buscar candles para {symbol}: resposta vazia da API")
        if bars.empty:
            raise RuntimeError(f"Nenhum candle retornado para {symbol}")

        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.xs(symbol, level="symbol")

        bars = bars.reset_index().rename(columns={"timestamp": "date"})
        bars = bars.sort_values("date").drop_duplicates(subset=["date"]).set_index("date")
        return bars[["open", "high", "low", "close", "volume"]].astype(float)
