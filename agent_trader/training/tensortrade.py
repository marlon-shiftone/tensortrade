from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import tensorflow as tf

from tensortrade.agents import DQNAgent

from ..domain import ModelRecord, OrderIntent, OrderSide
from ..envs import TensorTradeEnvConfig, build_tensortrade_env
from ..features import engineer_features
from ..integrations.alpaca.market_data import AlpacaHistoricalDataClient
from ..registry import ModelRegistry


@dataclass
class TensorTradePipelineConfig:
    symbol: str
    timeframe_amount: int = 1
    timeframe_unit: str = "Minute"
    lookback_bars: int = 500
    window_size: int = 30
    cash: float = 10000.0
    commission: float = 0.001
    train_episodes: int = 12
    train_steps: int = 1000
    order_notional_usd: float = 1000.0
    action_scheme: str = "bsh"
    trade_sizes: list[float] = field(default_factory=lambda: [1.0])
    model_path: str = "artifacts/alpaca_online_policy.keras"
    flat_position_penalty: float = 0.0001

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe_amount": self.timeframe_amount,
            "timeframe_unit": self.timeframe_unit,
            "lookback_bars": self.lookback_bars,
            "window_size": self.window_size,
            "cash": self.cash,
            "commission": self.commission,
            "train_episodes": self.train_episodes,
            "train_steps": self.train_steps,
            "order_notional_usd": self.order_notional_usd,
            "action_scheme": self.action_scheme,
            "trade_sizes": list(self.trade_sizes),
            "model_path": self.model_path,
            "flat_position_penalty": self.flat_position_penalty,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TensorTradePipelineConfig":
        return cls(
            symbol=data["symbol"],
            timeframe_amount=data.get("timeframe_amount", 1),
            timeframe_unit=data.get("timeframe_unit", "Minute"),
            lookback_bars=data.get("lookback_bars", 500),
            window_size=data.get("window_size", 30),
            cash=data.get("cash", 10000.0),
            commission=data.get("commission", 0.001),
            train_episodes=data.get("train_episodes", 12),
            train_steps=data.get("train_steps", 1000),
            order_notional_usd=data.get("order_notional_usd", 1000.0),
            action_scheme=data.get("action_scheme", "bsh"),
            trade_sizes=[float(value) for value in data.get("trade_sizes", [1.0])],
            model_path=data.get("model_path", "artifacts/alpaca_online_policy.keras"),
            flat_position_penalty=data.get("flat_position_penalty", 0.0001),
        )

    def env_config(self) -> TensorTradeEnvConfig:
        return TensorTradeEnvConfig(
            symbol=self.symbol,
            window_size=self.window_size,
            cash=self.cash,
            commission=self.commission,
            action_scheme=self.action_scheme,
            trade_sizes=list(self.trade_sizes),
            flat_position_penalty=self.flat_position_penalty,
        )


class LegacyTradingEnvAdapter:
    """Compatibiliza a TradingEnv baseada em Gymnasium com o DQNAgent legado."""

    def __init__(self, env):
        self._env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self):
        observation, _info = self._env.reset()
        return observation

    def step(self, action):
        observation, reward, terminated, truncated, info = self._env.step(action)
        done = bool(terminated or truncated)
        return observation, reward, done, info

    def render(self, **kwargs):
        return self._env.render(**kwargs)

    def save(self):
        return self._env.save()

    def close(self):
        return self._env.close()

    def __getattr__(self, name: str):
        return getattr(self._env, name)


def build_policy_network(observation_shape, n_actions: int) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=observation_shape, dtype="float32")
    x = tf.keras.layers.Conv1D(32, kernel_size=3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv1D(32, kernel_size=3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    # DQN needs unconstrained Q-values, not normalized class probabilities.
    outputs = tf.keras.layers.Dense(n_actions, activation="linear")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def infer_latest_action(model: tf.keras.Model, feature_df: pd.DataFrame, config: TensorTradePipelineConfig) -> int:
    legacy_env = LegacyTradingEnvAdapter(build_tensortrade_env(feature_df, config.env_config()))
    state = legacy_env.reset()
    done = False
    action = 0

    while not done:
        action_scores = _flatten_action_scores(model(np.expand_dims(state, 0)))
        action = int(np.argmax(action_scores))
        state, _reward, done, _info = legacy_env.step(action)

    return action


def _flatten_action_scores(model_output: Any) -> np.ndarray:
    action_scores = np.asarray(model_output, dtype=float)
    if action_scores.ndim > 1:
        action_scores = action_scores[0]
    return action_scores.astype(float)


def _select_valid_action(
    *,
    action_scores: np.ndarray,
    action_scheme: str,
    action_scheme_instance: Any,
    latest_close: float,
    context: Mapping[str, Any] | None = None,
) -> tuple[int, int, str | None]:
    raw_action = int(np.argmax(action_scores))
    valid_actions = set(range(len(action_scores)))
    valid_buy_actions: list[int] = []
    valid_sell_actions: list[int] = []

    position_qty = None
    available_cash = None
    if context is not None:
        try:
            position_qty = float(context.get("position_qty")) if context.get("position_qty") is not None else None
        except (TypeError, ValueError):
            position_qty = None
        try:
            available_cash = float(context.get("available_cash")) if context.get("available_cash") is not None else None
        except (TypeError, ValueError):
            available_cash = None

    if action_scheme == "simple_orders":
        action_payload = getattr(action_scheme_instance, "actions", None) or []
        valid_actions = {0}
        for index, payload in enumerate(action_payload):
            if index == 0 or payload is None:
                continue
            _exchange_pair, (_criteria, proportion, _duration, trade_side) = payload
            side_name = getattr(trade_side, "name", str(trade_side)).lower()
            if side_name == "buy":
                spend_capacity = available_cash
                if spend_capacity is None:
                    valid_actions.add(index)
                    continue
                spend = spend_capacity * max(float(proportion), 0.0)
                if spend > 0 and spend >= min(1.0, latest_close):
                    valid_actions.add(index)
                    valid_buy_actions.append(index)
            elif side_name == "sell":
                if position_qty is None:
                    valid_actions.add(index)
                    valid_sell_actions.append(index)
                    continue
                if position_qty > 0:
                    valid_actions.add(index)
                    valid_sell_actions.append(index)
    elif action_scheme == "bsh":
        valid_actions = {0}
        if available_cash is None or available_cash > 0:
            valid_actions.add(1)
            valid_buy_actions.append(1)
        if position_qty is None or position_qty > 0:
            valid_actions.add(2)
            valid_sell_actions.append(2)

    if raw_action in valid_actions:
        return raw_action, raw_action, None

    is_flat = position_qty is not None and position_qty <= 0
    has_cash = available_cash is None or available_cash > min(1.0, latest_close)
    if is_flat and has_cash and valid_buy_actions:
        preferred_buy = max(valid_buy_actions, key=lambda index: float(action_scores[index]))
        return preferred_buy, raw_action, (
            f"masked_invalid_action raw_action={raw_action} prefer_valid_buy"
        )

    ranked_actions = np.argsort(action_scores)[::-1]
    for candidate in ranked_actions:
        candidate_index = int(candidate)
        if candidate_index in valid_actions:
            return candidate_index, raw_action, f"masked_invalid_action raw_action={raw_action}"

    return 0, raw_action, f"masked_invalid_action raw_action={raw_action} fallback_hold"


def build_feature_reference(feature_df: pd.DataFrame) -> dict[str, Any]:
    numeric_columns = [column for column in feature_df.columns if pd.api.types.is_numeric_dtype(feature_df[column])]
    reference_df = feature_df[numeric_columns].copy()
    return {
        "feature_means": {column: float(reference_df[column].mean()) for column in numeric_columns},
        "feature_stds": {
            column: float(reference_df[column].std(ddof=0) or 0.0) for column in numeric_columns
        },
        "rows": int(len(reference_df)),
    }


class TensorTradeTrainer:
    def __init__(self, registry: ModelRegistry | None = None):
        self.registry = registry

    def train_from_features(
        self,
        feature_df: pd.DataFrame,
        config: TensorTradePipelineConfig,
    ) -> tf.keras.Model:
        env = build_tensortrade_env(feature_df, config.env_config())
        legacy_env = LegacyTradingEnvAdapter(env)
        network = build_policy_network(legacy_env.observation_space.shape, legacy_env.action_space.n)
        agent = DQNAgent(legacy_env, policy_network=network)

        model_path = Path(config.model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        if model_path.exists():
            agent.restore(str(model_path))

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
        agent.policy_network.save(str(model_path))
        return agent.policy_network

    def train_from_alpaca(
        self,
        market_data_client: AlpacaHistoricalDataClient,
        config: TensorTradePipelineConfig,
    ) -> pd.DataFrame:
        bars = market_data_client.load_recent_bars(
            symbol=config.symbol,
            timeframe_amount=config.timeframe_amount,
            timeframe_unit=config.timeframe_unit,
            lookback_bars=config.lookback_bars,
        )
        features = engineer_features(bars)
        self.train_from_features(features, config)
        return features

    def register_training_output(
        self,
        model_id: str,
        config: TensorTradePipelineConfig,
        feature_df: pd.DataFrame | None = None,
        notes: str = "",
    ) -> ModelRecord:
        if not self.registry:
            raise RuntimeError("registry obrigatorio para registrar artefato treinado")

        metadata = {
            "training_backend": "tensortrade",
            "training_job": config.to_dict(),
        }
        if feature_df is not None:
            metadata["feature_reference"] = build_feature_reference(feature_df)
        return self.registry.register_candidate(
            model_id=model_id,
            artifact_path=config.model_path,
            trainer="tensortrade",
            metadata=metadata,
            notes=notes,
        )


class TensorTradeModelExecutor:
    def __init__(
        self,
        feature_df: pd.DataFrame,
        config: TensorTradePipelineConfig,
    ):
        self.feature_df = feature_df
        self.config = config

    def predict_order(
        self,
        features: Mapping[str, Any],
        model: ModelRecord,
        context: Mapping[str, Any] | None = None,
    ) -> OrderIntent:
        model_path = Path(model.artifact_path)
        if not model_path.exists():
            raise FileNotFoundError(f"artefato de modelo nao encontrado: {model.artifact_path}")

        keras_model = tf.keras.models.load_model(str(model_path))
        env = build_tensortrade_env(self.feature_df, self.config.env_config())
        legacy_env = LegacyTradingEnvAdapter(env)
        expected_action_count = legacy_env.action_space.n
        output_shape = keras_model.output_shape
        model_action_count = int(output_shape[-1]) if isinstance(output_shape, tuple) else int(output_shape[0][-1])
        if model_action_count != expected_action_count:
            raise ValueError(
                "saida do modelo incompatível com action_space atual: "
                f"modelo={model_action_count} env={expected_action_count} action_scheme={self.config.action_scheme}"
            )

        state = legacy_env.reset()
        done = False
        action = 0
        raw_action = 0
        action_mask_reason = None
        while not done:
            action_scores = _flatten_action_scores(keras_model(np.expand_dims(state, 0)))
            action, raw_action, action_mask_reason = _select_valid_action(
                action_scores=action_scores,
                action_scheme=self.config.action_scheme,
                action_scheme_instance=getattr(env, "action_scheme", None),
                latest_close=float(self.feature_df["close"].iloc[-1]),
                context=context,
            )
            state, _reward, done, _info = legacy_env.step(action)

        latest_close = float(self.feature_df["close"].iloc[-1])
        latest_features = self.feature_df.iloc[-1].to_dict()
        latest_features["close"] = latest_close
        intent = _build_order_intent_from_action(
            action=action,
            action_scheme=self.config.action_scheme,
            action_scheme_instance=getattr(env, "action_scheme", None),
            model_id=model.model_id,
            symbol=self.config.symbol,
            latest_close=latest_close,
            latest_features=latest_features,
            default_notional_usd=self.config.order_notional_usd,
        )
        if action_mask_reason:
            intent.reason = f"{intent.reason} {action_mask_reason}"
        if raw_action != action:
            intent.reason = f"{intent.reason} selected_action={action}"
        position_qty = None
        if context is not None:
            try:
                position_qty = float(context.get("position_qty")) if context.get("position_qty") is not None else None
            except (TypeError, ValueError):
                position_qty = None
        if intent.side is OrderSide.SELL and position_qty is not None and position_qty <= 0:
            return OrderIntent(
                model_id=model.model_id,
                symbol=self.config.symbol,
                side=OrderSide.HOLD,
                confidence=intent.confidence,
                reason=(
                    f"{intent.reason} blocked_no_position actual_position_qty={position_qty:.6f}"
                ),
                reference_price=latest_close,
            )
        return intent


def _build_order_intent_from_action(
    *,
    action: int,
    action_scheme: str,
    action_scheme_instance: Any,
    model_id: str,
    symbol: str,
    latest_close: float,
    latest_features: Mapping[str, Any],
    default_notional_usd: float,
) -> OrderIntent:
    reason_prefix = f"tensortrade_action={action} action_scheme={action_scheme}"

    if action_scheme == "simple_orders":
        if action == 0:
            return OrderIntent(
                model_id=model_id,
                symbol=symbol,
                side=OrderSide.HOLD,
                confidence=1.0,
                reason=f"{reason_prefix} hold features={dict(latest_features)}",
                reference_price=latest_close,
            )

        action_payload = getattr(action_scheme_instance, "actions", None)
        if not action_payload:
            raise RuntimeError("action scheme simple_orders nao expôs catalogo de acoes")

        _exchange_pair, (_criteria, proportion, _duration, trade_side) = action_payload[action]
        side_name = getattr(trade_side, "name", str(trade_side)).lower()
        side = OrderSide.BUY if side_name == "buy" else OrderSide.SELL
        proportion = float(proportion)
        return OrderIntent(
            model_id=model_id,
            symbol=symbol,
            side=side,
            notional_usd=max(default_notional_usd, latest_close) if side is OrderSide.BUY else 0.0,
            size_fraction=proportion,
            confidence=1.0,
            reason=f"{reason_prefix} proportion={proportion} side={side.value} features={dict(latest_features)}",
            reference_price=latest_close,
        )

    side = OrderSide.BUY if action == 1 else OrderSide.SELL
    return OrderIntent(
        model_id=model_id,
        symbol=symbol,
        side=side,
        notional_usd=max(default_notional_usd, latest_close),
        confidence=1.0,
        reason=f"{reason_prefix} legacy_bsh features={dict(latest_features)}",
        reference_price=latest_close,
    )
