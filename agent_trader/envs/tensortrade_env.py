from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import tensortrade.env.default as default
from tensortrade.feed import DataFeed, NameSpace, Stream
from tensortrade.oms.exchanges import Exchange, ExchangeOptions
from tensortrade.oms.instruments import Instrument, USD
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Portfolio, Wallet


@dataclass
class TensorTradeEnvConfig:
    symbol: str
    window_size: int = 30
    cash: float = 10000.0
    commission: float = 0.001


def create_asset(symbol: str) -> Instrument:
    return Instrument(symbol, 6, f"{symbol} asset")


def build_tensortrade_env(feature_df: pd.DataFrame, config: TensorTradeEnvConfig):
    if len(feature_df) < max(config.window_size + 5, 50):
        raise RuntimeError("Poucos candles apos engenharia de features; aumente o lookback.")

    asset = create_asset(config.symbol)
    price_stream = Stream.source(feature_df["close"].tolist(), dtype="float").rename(f"USD-{asset.symbol}")
    exchange = Exchange(
        "alpaca_sim",
        service=execute_order,
        options=ExchangeOptions(commission=config.commission),
    )(price_stream)

    portfolio = Portfolio(
        USD,
        [
            Wallet(exchange, config.cash * USD),
            Wallet(exchange, 0 * asset),
        ],
    )

    streams = []
    with NameSpace("alpaca"):
        for column in feature_df.columns:
            streams.append(Stream.source(feature_df[column].tolist(), dtype="float").rename(f"{asset.symbol}:{column}"))

    feed = DataFeed(streams)
    action_scheme = default.actions.BSH(
        cash=portfolio.get_wallet(exchange.id, USD),
        asset=portfolio.get_wallet(exchange.id, asset),
    )
    reward_scheme = default.rewards.SimpleProfit(window_size=5)
    return default.create(
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        feed=feed,
        window_size=config.window_size,
        min_periods=config.window_size,
        enable_logger=False,
    )
