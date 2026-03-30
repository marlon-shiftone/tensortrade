import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import tensorflow as tf

import tensortrade.env.default as default
from tensortrade.agents import DQNAgent
from tensortrade.feed import DataFeed, NameSpace, Stream
from tensortrade.oms.exchanges import Exchange, ExchangeOptions
from tensortrade.oms.instruments import Instrument, USD
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Portfolio, Wallet

from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce


LOGGER = logging.getLogger("alpaca_online_paper")


@dataclass
class Config:
    symbol: str
    timeframe_amount: int
    timeframe_unit: str
    lookback_bars: int
    window_size: int
    cash: float
    commission: float
    poll_seconds: int
    train_episodes: int
    train_steps: int
    paper_notional_usd: float
    min_trade_gap_seconds: int
    model_path: Optional[str]
    dry_run: bool


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Treino online simples com TensorTrade usando dados da Alpaca e execucao no paper account."
    )
    parser.add_argument("--symbol", default=os.getenv("ALPACA_SYMBOL", "AAPL"))
    parser.add_argument("--timeframe-amount", type=int, default=int(os.getenv("ALPACA_TIMEFRAME_AMOUNT", "1")))
    parser.add_argument("--timeframe-unit", default=os.getenv("ALPACA_TIMEFRAME_UNIT", "Minute"))
    parser.add_argument("--lookback-bars", type=int, default=int(os.getenv("ALPACA_LOOKBACK_BARS", "500")))
    parser.add_argument("--window-size", type=int, default=int(os.getenv("ALPACA_WINDOW_SIZE", "30")))
    parser.add_argument("--cash", type=float, default=float(os.getenv("ALPACA_SIM_CASH", "10000")))
    parser.add_argument("--commission", type=float, default=float(os.getenv("ALPACA_SIM_COMMISSION", "0.001")))
    parser.add_argument("--poll-seconds", type=int, default=int(os.getenv("ALPACA_POLL_SECONDS", "60")))
    parser.add_argument("--train-episodes", type=int, default=int(os.getenv("ALPACA_TRAIN_EPISODES", "3")))
    parser.add_argument("--train-steps", type=int, default=int(os.getenv("ALPACA_TRAIN_STEPS", "200")))
    parser.add_argument("--paper-notional-usd", type=float, default=float(os.getenv("ALPACA_PAPER_NOTIONAL_USD", "1000")))
    parser.add_argument(
        "--min-trade-gap-seconds",
        type=int,
        default=int(os.getenv("ALPACA_MIN_TRADE_GAP_SECONDS", "300")),
    )
    parser.add_argument("--model-path", default=os.getenv("ALPACA_MODEL_PATH", "artifacts/alpaca_online_policy.keras"))
    parser.add_argument("--dry-run", action="store_true", default=os.getenv("ALPACA_DRY_RUN", "false").lower() == "true")
    args = parser.parse_args()
    return Config(
        symbol=args.symbol.upper(),
        timeframe_amount=args.timeframe_amount,
        timeframe_unit=args.timeframe_unit,
        lookback_bars=args.lookback_bars,
        window_size=args.window_size,
        cash=args.cash,
        commission=args.commission,
        poll_seconds=args.poll_seconds,
        train_episodes=args.train_episodes,
        train_steps=args.train_steps,
        paper_notional_usd=args.paper_notional_usd,
        min_trade_gap_seconds=args.min_trade_gap_seconds,
        model_path=args.model_path,
        dry_run=args.dry_run,
    )


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Variavel de ambiente obrigatoria ausente: {name}")
    return value


def resolve_timeframe(amount: int, unit: str) -> TimeFrame:
    unit_normalized = unit.strip().lower()
    mapping = {
        "minute": TimeFrameUnit.Minute,
        "min": TimeFrameUnit.Minute,
        "hour": TimeFrameUnit.Hour,
        "day": TimeFrameUnit.Day,
    }
    if unit_normalized not in mapping:
        raise ValueError(f"timeframe_unit invalido: {unit}")
    return TimeFrame(amount, mapping[unit_normalized])


def create_asset(symbol: str) -> Instrument:
    return Instrument(symbol, 6, f"{symbol} asset")


def load_recent_bars(client: StockHistoricalDataClient, config: Config) -> pd.DataFrame:
    request = StockBarsRequest(
        symbol_or_symbols=config.symbol,
        timeframe=resolve_timeframe(config.timeframe_amount, config.timeframe_unit),
        limit=config.lookback_bars,
        adjustment="raw",
        feed="iex",
    )
    bars = client.get_stock_bars(request).df
    if bars.empty:
        raise RuntimeError(f"Nenhum candle retornado para {config.symbol}")

    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(config.symbol, level="symbol")

    bars = bars.reset_index().rename(columns={"timestamp": "date"})
    bars = bars.sort_values("date").drop_duplicates(subset=["date"]).set_index("date")
    return bars[["open", "high", "low", "close", "volume"]].astype(float)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["return_1"] = enriched["close"].pct_change().fillna(0.0)
    enriched["return_5"] = enriched["close"].pct_change(5).fillna(0.0)
    enriched["range_pct"] = ((enriched["high"] - enriched["low"]) / enriched["close"]).replace([np.inf, -np.inf], 0.0)
    enriched["sma_10_gap"] = (enriched["close"] / enriched["close"].rolling(10).mean() - 1.0).fillna(0.0)
    enriched["sma_30_gap"] = (enriched["close"] / enriched["close"].rolling(30).mean() - 1.0).fillna(0.0)
    enriched["volatility_10"] = enriched["return_1"].rolling(10).std().fillna(0.0)
    enriched["volume_zscore_20"] = (
        (enriched["volume"] - enriched["volume"].rolling(20).mean()) /
        enriched["volume"].rolling(20).std()
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return enriched.dropna().copy()


def build_env(feature_df: pd.DataFrame, config: Config):
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
    env = default.create(
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        feed=feed,
        window_size=config.window_size,
        min_periods=config.window_size,
        enable_logger=False,
    )
    return env


def build_policy_network(observation_shape, n_actions: int) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=observation_shape, dtype="float32")
    x = tf.keras.layers.Conv1D(32, kernel_size=3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv1D(32, kernel_size=3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(n_actions, activation="softmax")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def train_agent(feature_df: pd.DataFrame, config: Config) -> DQNAgent:
    env = build_env(feature_df, config)
    network = build_policy_network(env.observation_space.shape, env.action_space.n)
    agent = DQNAgent(env, policy_network=network)

    if config.model_path and Path(config.model_path).exists():
        LOGGER.info("Restaurando modelo salvo em %s", config.model_path)
        agent.restore(config.model_path)

    LOGGER.info(
        "Treinando com %s candles, window=%s, episodes=%s, steps=%s",
        len(feature_df),
        config.window_size,
        config.train_episodes,
        config.train_steps,
    )
    agent.train(
        n_episodes=config.train_episodes,
        n_steps=config.train_steps,
        save_path="",
        render_interval=None,
        batch_size=32,
        memory_capacity=max(500, config.train_steps * 5),
        learning_rate=1e-3,
        discount_factor=0.99,
        update_target_every=25,
    )
    return agent


def latest_action(agent: DQNAgent) -> int:
    state = agent.env.reset()
    done = False
    action = 0
    while not done:
        action = agent.get_action(state, threshold=0.0)
        state, _, done, _ = agent.env.step(action)
    return int(action)


def current_position_qty(trading_client: TradingClient, symbol: str) -> float:
    try:
        position = trading_client.get_open_position(symbol)
        return float(position.qty)
    except Exception:
        return 0.0


def submit_paper_order(trading_client: TradingClient, config: Config, signal: int, last_price: float) -> None:
    position_qty = current_position_qty(trading_client, config.symbol)
    wants_long = signal == 1

    if wants_long and position_qty > 0:
        LOGGER.info("Sinal comprado e posicao ja aberta em %s: %.6f", config.symbol, position_qty)
        return

    if not wants_long and position_qty <= 0:
        LOGGER.info("Sinal em caixa e sem posicao aberta em %s", config.symbol)
        return

    if wants_long:
        notional = max(config.paper_notional_usd, last_price)
        order = MarketOrderRequest(
            symbol=config.symbol,
            notional=round(notional, 2),
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        LOGGER.info("Enviando compra paper: %s USD em %s", notional, config.symbol)
    else:
        order = MarketOrderRequest(
            symbol=config.symbol,
            qty=abs(position_qty),
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        LOGGER.info("Enviando zeragem paper: %.6f %s", abs(position_qty), config.symbol)

    trading_client.submit_order(order_data=order)


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    config = parse_args()

    api_key = require_env("ALPACA_API_KEY")
    secret_key = require_env("ALPACA_SECRET_KEY")

    data_client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
    trading_client = TradingClient(api_key=api_key, secret_key=secret_key, paper=True)

    last_submitted_at = 0.0
    model_path = Path(config.model_path) if config.model_path else None
    if model_path:
        model_path.parent.mkdir(parents=True, exist_ok=True)

    while True:
        bars = load_recent_bars(data_client, config)
        features = engineer_features(bars)
        agent = train_agent(features, config)
        signal = latest_action(agent)
        last_price = float(features["close"].iloc[-1])

        if model_path:
            agent.policy_network.save(model_path)
            LOGGER.info("Modelo salvo em %s", model_path)

        LOGGER.info(
            "Sinal atual para %s: %s | close=%.4f",
            config.symbol,
            "LONG" if signal == 1 else "CASH",
            last_price,
        )

        now = time.time()
        if config.dry_run:
            LOGGER.info("dry-run ativo; nenhuma ordem sera enviada")
        elif now - last_submitted_at < config.min_trade_gap_seconds:
            LOGGER.info("Cooldown de ordens ativo; aguardando antes de novo envio")
        else:
            submit_paper_order(trading_client, config, signal, last_price)
            last_submitted_at = now

        LOGGER.info("Aguardando %s segundos para proximo ciclo", config.poll_seconds)
        time.sleep(config.poll_seconds)


if __name__ == "__main__":
    main()
