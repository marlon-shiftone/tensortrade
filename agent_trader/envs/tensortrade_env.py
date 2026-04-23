from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd
import tensortrade.env.default as default
from tensortrade.feed import DataFeed, NameSpace, Stream
from tensortrade.oms.exchanges import Exchange, ExchangeOptions
from tensortrade.oms.instruments import Instrument, USD
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Portfolio, Wallet


class _SuppressTensorTradeCommissionPrecisionWarning(logging.Filter):
    _MESSAGE_FRAGMENT = "Commission is > 0 but less than instrument precision."

    def filter(self, record: logging.LogRecord) -> bool:
        return self._MESSAGE_FRAGMENT not in record.getMessage()


_ROOT_LOGGER = logging.getLogger()
if not any(isinstance(existing, _SuppressTensorTradeCommissionPrecisionWarning) for existing in _ROOT_LOGGER.filters):
    _ROOT_LOGGER.addFilter(_SuppressTensorTradeCommissionPrecisionWarning())


@dataclass
class TensorTradeEnvConfig:
    symbol: str
    window_size: int = 30
    cash: float = 10000.0
    commission: float = 0.001
    action_scheme: str = "bsh"
    trade_sizes: list[float] = field(default_factory=lambda: [1.0])
    flat_position_penalty: float = 0.0001


def create_asset(symbol: str) -> Instrument:
    return Instrument(symbol, 6, f"{symbol} asset")


class PositionAwareProfit(default.rewards.SimpleProfit):
    """Simple profit with a small penalty for staying flat forever."""

    def __init__(self, window_size: int = 5, flat_position_penalty: float = 0.0001):
        super().__init__(window_size=window_size)
        self._flat_position_penalty = float(flat_position_penalty)

    def get_reward(self, portfolio) -> float:
        reward = float(super().get_reward(portfolio))
        non_base_balance = 0.0
        for wallet in portfolio.wallets:
            if wallet.instrument == portfolio.base_instrument:
                continue
            non_base_balance += abs(wallet.total_balance.as_float())
        if non_base_balance <= 1e-8:
            reward -= self._flat_position_penalty
        return reward


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
    if config.action_scheme == "simple_orders":
        action_scheme = default.actions.SimpleOrders(
            trade_sizes=config.trade_sizes,
            min_order_pct=0.0,
            min_order_abs=0.0,
        )
    elif config.action_scheme == "bsh":
        action_scheme = default.actions.BSH(
            cash=portfolio.get_wallet(exchange.id, USD),
            asset=portfolio.get_wallet(exchange.id, asset),
        )
    else:
        raise ValueError(f"action_scheme invalido: {config.action_scheme}")
    reward_scheme = PositionAwareProfit(
        window_size=5,
        flat_position_penalty=config.flat_position_penalty,
    )
    return default.create(
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        feed=feed,
        window_size=config.window_size,
        min_periods=config.window_size,
        enable_logger=False,
    )
