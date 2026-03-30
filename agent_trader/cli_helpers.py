from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def add_pipeline_args(
    parser,
    include_model_id: bool = False,
    symbol_required: bool = False,
) -> None:
    if include_model_id:
        parser.add_argument("--model-id", required=True)
    parser.add_argument("--symbol", required=symbol_required)
    parser.add_argument("--timeframe-amount", type=int, default=None)
    parser.add_argument("--timeframe-unit", default=None)
    parser.add_argument("--lookback-bars", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--cash", type=float, default=None)
    parser.add_argument("--commission", type=float, default=None)
    parser.add_argument("--train-episodes", type=int, default=None)
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--order-notional-usd", type=float, default=None)
    parser.add_argument("--model-path", default=None)


def resolve_model_path(
    explicit_model_path: str | None,
    artifacts_dir: Path,
    model_id: str,
) -> str:
    if explicit_model_path:
        return str(Path(explicit_model_path))
    return str(artifacts_dir / f"{model_id}.keras")


def pipeline_config_from_args(
    args,
    artifacts_dir: Path,
    model_id: str,
    base_data: dict[str, Any] | None = None,
) -> "TensorTradePipelineConfig":
    from .training import TensorTradePipelineConfig

    base_data = base_data or {}

    symbol = (args.symbol or base_data.get("symbol"))
    if not symbol:
        raise ValueError("symbol obrigatorio para montar o pipeline")

    config_data = {
        "symbol": symbol.upper(),
        "timeframe_amount": args.timeframe_amount if args.timeframe_amount is not None else base_data.get("timeframe_amount", 1),
        "timeframe_unit": args.timeframe_unit if args.timeframe_unit is not None else base_data.get("timeframe_unit", "Minute"),
        "lookback_bars": args.lookback_bars if args.lookback_bars is not None else base_data.get("lookback_bars", 500),
        "window_size": args.window_size if args.window_size is not None else base_data.get("window_size", 30),
        "cash": args.cash if args.cash is not None else base_data.get("cash", 10000.0),
        "commission": args.commission if args.commission is not None else base_data.get("commission", 0.001),
        "train_episodes": args.train_episodes if args.train_episodes is not None else base_data.get("train_episodes", 3),
        "train_steps": args.train_steps if args.train_steps is not None else base_data.get("train_steps", 200),
        "order_notional_usd": (
            args.order_notional_usd if args.order_notional_usd is not None else base_data.get("order_notional_usd", 1000.0)
        ),
        "model_path": resolve_model_path(
            explicit_model_path=args.model_path or base_data.get("model_path"),
            artifacts_dir=artifacts_dir,
            model_id=model_id,
        ),
    }
    return TensorTradePipelineConfig.from_dict(config_data)
