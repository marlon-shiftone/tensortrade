from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from ...cli_helpers import add_pipeline_args, pipeline_config_from_args, print_json
from ...config import AppConfig
from ...domain import OrderSide, PerformanceSnapshot, RunMode
from ...registry import ModelRegistry
from ...storage import (
    ExecutionResultStore,
    JsonlStore,
    PerformanceSnapshotStore,
    SupervisorReportStore,
    SupervisorStateStore,
    TrainingRunStore,
)
from ...supervisor import Supervisor
from ...supervisor_loop import SupervisorLoop


def _build_app_config(args) -> AppConfig:
    config = AppConfig(
        project_name="alpaca",
        project_dir=Path(args.project_dir) if getattr(args, "project_dir", None) else None,
    )
    config.ensure_state_dir()
    return config


def _build_registry(args) -> tuple[AppConfig, ModelRegistry]:
    config = _build_app_config(args)
    return config, ModelRegistry(config.registry_path)


def _build_snapshot_from_args(args) -> PerformanceSnapshot:
    return PerformanceSnapshot(
        symbol=args.symbol.upper(),
        net_profit=args.net_profit,
        rolling_sharpe=args.rolling_sharpe,
        max_drawdown=args.max_drawdown,
        hit_rate=args.hit_rate,
        turnover=args.turnover,
        feature_drift=args.feature_drift,
        paper_days=args.paper_days,
        trades=args.trades,
    )


def _add_review_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--mode", choices=[mode.value for mode in RunMode], default=RunMode.PAPER.value)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--net-profit", type=float, default=0.0)
    parser.add_argument("--rolling-sharpe", type=float, required=True)
    parser.add_argument("--max-drawdown", type=float, required=True)
    parser.add_argument("--hit-rate", type=float, required=True)
    parser.add_argument("--turnover", type=float, default=0.0)
    parser.add_argument("--feature-drift", type=float, required=True)
    parser.add_argument("--paper-days", type=int, default=0)
    parser.add_argument("--trades", type=int, default=0)


def _add_metrics_collection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--history-period", default="1M")
    parser.add_argument("--history-timeframe", default="1D")
    parser.add_argument("--orders-limit", type=int, default=500)
    parser.add_argument("--after-days", type=int, default=30)


def _add_online_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--iterations", type=int, default=0, help="0 significa loop continuo.")
    parser.add_argument("--sleep-seconds", type=int, default=60)
    parser.add_argument("--retrain-suffix", default="rt")
    parser.add_argument("--shadow-eval-iterations", type=int, default=10)
    parser.add_argument("--shadow-min-trades", type=int, default=2)
    parser.add_argument("--shadow-wins-required", type=int, default=3)
    parser.add_argument("--shadow-max-eval-iterations", type=int, default=120)
    parser.add_argument("--shadow-max-zero-trade-iterations", type=int, default=30)
    parser.add_argument("--no-shadow-rebase-promotion", action="store_true")
    parser.add_argument("--shadow-cash", type=float, default=10000.0)
    parser.add_argument("--no-auto-promote-paper", action="store_true")
    parser.add_argument("--stop-on-pause", action="store_true")


def register_parser(subparsers) -> None:
    parser = subparsers.add_parser("alpaca", help="Comandos da integracao Alpaca.")
    parser.add_argument("--project-dir", default=None, help="Diretorio operacional opcional para o projeto Alpaca.")
    commands = parser.add_subparsers(dest="alpaca_command", required=True)

    init_state = commands.add_parser("init-state", help="Inicializa estado do projeto operacional.")
    init_state.set_defaults(handler=handle_init_state)

    list_models = commands.add_parser("list-models", help="Lista modelos registrados.")
    list_models.set_defaults(handler=handle_list_models)

    register = commands.add_parser("register-model", help="Registra um novo candidato.")
    register.add_argument("--model-id", required=True)
    register.add_argument("--artifact-path", required=True)
    register.add_argument("--trainer", default="tensortrade")
    register.add_argument("--notes", default="")
    register.set_defaults(handler=handle_register_model)

    train = commands.add_parser("train", help="Treina um modelo TensorTrade com dados da Alpaca.")
    add_pipeline_args(train, include_model_id=True, symbol_required=True)
    train.add_argument("--notes", default="")
    train.set_defaults(handler=handle_train)

    run_once = commands.add_parser("run", help="Executa um ciclo de inferencia e ordem.")
    add_pipeline_args(run_once, include_model_id=True, symbol_required=False)
    run_once.add_argument("--mode", choices=[mode.value for mode in RunMode], default=RunMode.PAPER.value)
    run_once.add_argument("--dry-run", action="store_true")
    run_once.set_defaults(handler=handle_run)

    collect_metrics = commands.add_parser("collect-metrics", help="Coleta automaticamente um snapshot de performance.")
    add_pipeline_args(collect_metrics, include_model_id=True, symbol_required=False)
    collect_metrics.add_argument("--mode", choices=[mode.value for mode in RunMode], default=RunMode.PAPER.value)
    _add_metrics_collection_args(collect_metrics)
    collect_metrics.add_argument("--source", default="alpaca_collector")
    collect_metrics.add_argument("--notes", default="")
    collect_metrics.set_defaults(handler=handle_collect_metrics)

    record_metrics = commands.add_parser("record-metrics", help="Persiste um snapshot de performance.")
    _add_review_args(record_metrics)
    record_metrics.add_argument("--source", default="manual")
    record_metrics.add_argument("--notes", default="")
    record_metrics.set_defaults(handler=handle_record_metrics)

    review = commands.add_parser("review", help="Avalia um snapshot sem alterar estado persistido.")
    _add_review_args(review)
    review.set_defaults(handler=handle_review)

    supervise = commands.add_parser("supervise", help="Executa o loop do supervisor usando snapshots persistidos.")
    supervise.add_argument("--model-id", required=True)
    supervise.add_argument("--mode", choices=[mode.value for mode in RunMode], default=RunMode.PAPER.value)
    supervise.add_argument("--no-auto-retrain", action="store_true")
    supervise.add_argument("--retrain-suffix", default="rt")
    supervise.set_defaults(handler=handle_supervise)

    operate = commands.add_parser("operate", help="Executa ciclos de run -> collect-metrics -> supervise.")
    add_pipeline_args(operate, include_model_id=True, symbol_required=False)
    operate.add_argument("--mode", choices=[mode.value for mode in RunMode], default=RunMode.PAPER.value)
    operate.add_argument("--dry-run", action="store_true")
    _add_metrics_collection_args(operate)
    operate.add_argument("--iterations", type=int, default=1)
    operate.add_argument("--sleep-seconds", type=int, default=60)
    operate.add_argument("--no-auto-retrain", action="store_true")
    operate.add_argument("--retrain-suffix", default="rt")
    operate.set_defaults(handler=handle_operate)

    online_train = commands.add_parser(
        "online-train",
        help="Opera em paper, monitora degradacao, retreina challenger e promove em shadow quando passar.",
    )
    add_pipeline_args(online_train, include_model_id=True, symbol_required=False)
    online_train.add_argument("--dry-run", action="store_true")
    _add_metrics_collection_args(online_train)
    _add_online_training_args(online_train)
    online_train.set_defaults(handler=handle_online_train)

    paper = commands.add_parser("mark-paper-ready", help="Marca modelo como pronto para paper.")
    paper.add_argument("--model-id", required=True)
    paper.add_argument("--notes", default="")
    paper.set_defaults(handler=handle_mark_paper_ready)

    approve = commands.add_parser("approve-live", help="Registra aprovacao humana para live.")
    approve.add_argument("--model-id", required=True)
    approve.add_argument("--approved-by", required=True)
    approve.add_argument("--notes", default="")
    approve.set_defaults(handler=handle_approve_live)

    activate = commands.add_parser("activate-live", help="Ativa um modelo em live.")
    activate.add_argument("--model-id", required=True)
    activate.add_argument("--notes", default="")
    activate.set_defaults(handler=handle_activate_live)


def handle_init_state(args) -> int:
    config, registry = _build_registry(args)
    print_json(
        {
            "project": config.project_name,
            "state": registry.state(),
        }
    )
    return 0


def handle_list_models(args) -> int:
    _config, registry = _build_registry(args)
    print_json({"models": [model.to_dict() for model in registry.list_models()]})
    return 0


def handle_register_model(args) -> int:
    _config, registry = _build_registry(args)
    record = registry.register_candidate(
        model_id=args.model_id,
        artifact_path=args.artifact_path,
        trainer=args.trainer,
        notes=args.notes,
    )
    print_json(record.to_dict())
    return 0


def handle_train(args) -> int:
    from ...integrations.alpaca import AlpacaHistoricalDataClient, AlpacaMarketDataConfig
    from ...training import TensorTradeTrainer

    config, registry = _build_registry(args)
    pipeline = pipeline_config_from_args(
        args=args,
        artifacts_dir=config.artifacts_dir,
        model_id=args.model_id,
    )
    trainer = TensorTradeTrainer(registry=registry)
    features = trainer.train_from_alpaca(
        market_data_client=AlpacaHistoricalDataClient(AlpacaMarketDataConfig.from_env()),
        config=pipeline,
    )
    record = trainer.register_training_output(
        model_id=args.model_id,
        config=pipeline,
        feature_df=features,
        notes=args.notes,
    )
    TrainingRunStore(config.training_history_path).append_run(
        project=config.project_name,
        model_id=args.model_id,
        trainer="tensortrade",
        payload={
            "rows_trained": len(features),
            "pipeline": pipeline.to_dict(),
            "source": "manual_train",
        },
    )
    print_json(
        {
            "model": record.to_dict(),
            "rows_trained": len(features),
            "model_path": pipeline.model_path,
        }
    )
    return 0


def handle_run(args) -> int:
    from ...features import engineer_features
    from ...integrations.alpaca import (
        AlpacaBrokerConfig,
        AlpacaHistoricalDataClient,
        AlpacaMarketDataConfig,
        AlpacaTradingBroker,
    )
    from ...runtime import AllowAllRiskManager, TradingRuntime
    from ...training import TensorTradeModelExecutor

    config, registry = _build_registry(args)
    current_model = registry.get_model(args.model_id)
    base_training_job = current_model.metadata.get("training_job", {})
    pipeline = pipeline_config_from_args(
        args=args,
        artifacts_dir=config.artifacts_dir,
        model_id=args.model_id,
        base_data=base_training_job,
    )
    mode = RunMode(args.mode)
    market_data = AlpacaHistoricalDataClient(AlpacaMarketDataConfig.from_env())
    bars = market_data.load_recent_bars(
        symbol=pipeline.symbol,
        timeframe_amount=pipeline.timeframe_amount,
        timeframe_unit=pipeline.timeframe_unit,
        lookback_bars=pipeline.lookback_bars,
    )
    feature_df = engineer_features(bars)
    broker = AlpacaTradingBroker(AlpacaBrokerConfig.from_env(dry_run=args.dry_run))
    runtime = TradingRuntime(
        registry=registry,
        model_executor=TensorTradeModelExecutor(feature_df=feature_df, config=pipeline),
        risk_manager=AllowAllRiskManager(),
        broker=broker,
    )
    runtime.start(args.model_id, mode)
    position_qty = broker.current_position_qty(mode, pipeline.symbol)
    available_cash = broker.available_cash(mode)
    result = runtime.run_once(
        model_id=args.model_id,
        mode=mode,
        features=feature_df.iloc[-1].to_dict(),
        context={
            "latest_close": float(feature_df["close"].iloc[-1]),
            "position_qty": position_qty,
            "available_cash": available_cash,
        },
    )
    payload = {
        "mode": mode.value,
        "latest_close": float(feature_df["close"].iloc[-1]),
        "result": result,
        "pipeline": pipeline.to_dict(),
    }
    ExecutionResultStore(config.execution_history_path).append_result(
        project=config.project_name,
        model_id=args.model_id,
        mode=mode,
        payload=payload,
    )
    print_json(payload)
    return 0


def _resolved_pipeline_for_model(args, config: AppConfig, registry: ModelRegistry):
    return _resolved_pipeline_for_model_id(args, config, registry, args.model_id)


def _metrics_config_from_args(args, config: AppConfig):
    from ...integrations.alpaca import AlpacaMetricsConfig

    return AlpacaMetricsConfig(
        history_period=args.history_period,
        history_timeframe=args.history_timeframe,
        orders_limit=args.orders_limit,
        after_days=args.after_days,
        execution_history_path=str(config.execution_history_path),
    )


def _resolved_pipeline_for_model_id(
    args,
    config: AppConfig,
    registry: ModelRegistry,
    model_id: str,
):
    current_model = registry.get_model(model_id)
    base_training_job = current_model.metadata.get("training_job", {})
    pipeline = pipeline_config_from_args(
        args=args,
        artifacts_dir=config.artifacts_dir,
        model_id=model_id,
        base_data=base_training_job,
    )
    return current_model, pipeline


def _train_alpaca_challenger(
    *,
    args,
    config: AppConfig,
    registry: ModelRegistry,
    parent_model_id: str,
    new_model_id: str,
    snapshot: PerformanceSnapshot,
    market_data_client,
    trainer_source: str,
) -> dict:
    from ...training import TensorTradePipelineConfig, TensorTradeTrainer

    current_model = registry.get_model(parent_model_id)
    training_job = current_model.metadata.get("training_job")
    if not training_job:
        raise RuntimeError(
            f"modelo {current_model.model_id} nao possui metadados de treino para auto-retreino"
        )

    retrain_pipeline = TensorTradePipelineConfig.from_dict(training_job)
    retrain_pipeline.model_path = str(config.artifacts_dir / f"{new_model_id}.keras")
    retrain_pipeline.train_episodes = max(int(retrain_pipeline.train_episodes), 12)
    retrain_pipeline.train_steps = max(int(retrain_pipeline.train_steps), 1000)
    retrain_pipeline.flat_position_penalty = max(
        float(getattr(retrain_pipeline, "flat_position_penalty", 0.0001)),
        0.0001,
    )

    trainer = TensorTradeTrainer(registry=registry)
    features = trainer.train_from_alpaca(
        market_data_client=market_data_client,
        config=retrain_pipeline,
    )
    record = trainer.register_training_output(
        model_id=new_model_id,
        config=retrain_pipeline,
        feature_df=features,
        notes=f"auto_retrain_from={current_model.model_id} snapshot_at={snapshot.as_of}",
    )
    training_payload = {
        "rows_trained": len(features),
        "pipeline": retrain_pipeline.to_dict(),
        "source": trainer_source,
        "trigger_model_id": current_model.model_id,
        "trigger_snapshot": snapshot.to_dict(),
    }
    TrainingRunStore(config.training_history_path).append_run(
        project=config.project_name,
        model_id=new_model_id,
        trainer="tensortrade",
        payload=training_payload,
    )
    return {
        "new_model": record.to_dict(),
        "training_payload": training_payload,
        "pipeline": retrain_pipeline,
        "feature_rows": len(features),
    }


def handle_collect_metrics(args) -> int:
    from ...integrations.alpaca import (
        AlpacaBrokerConfig,
        AlpacaHistoricalDataClient,
        AlpacaMarketDataConfig,
        AlpacaMetricsCollector,
        AlpacaTradingBroker,
    )

    config, registry = _build_registry(args)
    model, pipeline = _resolved_pipeline_for_model(args, config, registry)
    mode = RunMode(args.mode)
    market_data_client = AlpacaHistoricalDataClient(AlpacaMarketDataConfig.from_env())
    broker = AlpacaTradingBroker(AlpacaBrokerConfig.from_env())
    collector = AlpacaMetricsCollector(
        trading_client=broker.client_for_mode(mode),
        market_data_client=market_data_client,
        config=_metrics_config_from_args(args, config),
    )
    snapshot = collector.collect_snapshot(
        model=model,
        symbol=pipeline.symbol,
        timeframe_amount=pipeline.timeframe_amount,
        timeframe_unit=pipeline.timeframe_unit,
        lookback_bars=pipeline.lookback_bars,
    )
    record = PerformanceSnapshotStore(config.metrics_history_path).append_snapshot(
        project=config.project_name,
        model_id=args.model_id,
        mode=mode,
        snapshot=snapshot,
        source=args.source,
        notes=args.notes,
    )
    print_json(record)
    return 0


def handle_record_metrics(args) -> int:
    config, registry = _build_registry(args)
    registry.get_model(args.model_id)
    snapshot = _build_snapshot_from_args(args)
    record = PerformanceSnapshotStore(config.metrics_history_path).append_snapshot(
        project=config.project_name,
        model_id=args.model_id,
        mode=RunMode(args.mode),
        snapshot=snapshot,
        source=args.source,
        notes=args.notes,
    )
    print_json(record)
    return 0


def handle_review(args) -> int:
    config, registry = _build_registry(args)
    model = registry.get_model(args.model_id)
    snapshot = _build_snapshot_from_args(args)
    supervisor = Supervisor(config.thresholds)
    report, state = supervisor.review(
        model=model,
        snapshot=snapshot,
        mode=RunMode(args.mode),
    )
    print_json(
        {
            "model": model.to_dict(),
            "snapshot": snapshot.to_dict(),
            "supervisor_report": report.to_dict(),
            "supervisor_state": state.to_dict(),
        }
    )
    return 0


def handle_supervise(args) -> int:
    from ...integrations.alpaca import AlpacaHistoricalDataClient, AlpacaMarketDataConfig

    config, registry = _build_registry(args)
    supervisor = Supervisor(config.thresholds)
    loop = SupervisorLoop(
        project=config.project_name,
        registry=registry,
        snapshot_store=PerformanceSnapshotStore(config.metrics_history_path),
        report_store=SupervisorReportStore(config.supervisor_reports_path),
        state_store=SupervisorStateStore(config.supervisor_state_path),
        supervisor=supervisor,
    )
    current_model = registry.get_model(args.model_id)

    def retrain_callback(snapshot: PerformanceSnapshot) -> dict:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        new_model_id = f"{current_model.model_id}_{args.retrain_suffix}_{timestamp}"
        retrain_result = _train_alpaca_challenger(
            args=args,
            config=config,
            registry=registry,
            parent_model_id=current_model.model_id,
            new_model_id=new_model_id,
            snapshot=snapshot,
            market_data_client=AlpacaHistoricalDataClient(AlpacaMarketDataConfig.from_env()),
            trainer_source="supervisor_auto_retrain",
        )
        paper_record = registry.mark_paper_ready(
            new_model_id,
            notes=f"auto retrain challenger derivado de {current_model.model_id}",
        )
        return {
            "new_model": retrain_result["new_model"],
            "paper_record": paper_record.to_dict(),
            "training_payload": retrain_result["training_payload"],
        }

    result = loop.run_once(
        model_id=args.model_id,
        mode=RunMode(args.mode),
        retrain_callback=None if args.no_auto_retrain else retrain_callback,
    )
    print_json(result.to_dict())
    return 0


def handle_operate(args) -> int:
    from ...features import engineer_features
    from ...integrations.alpaca import (
        AlpacaBrokerConfig,
        AlpacaHistoricalDataClient,
        AlpacaMarketDataConfig,
        AlpacaMetricsCollector,
        AlpacaTradingBroker,
    )
    from ...runtime import AllowAllRiskManager, TradingRuntime
    from ...training import TensorTradeModelExecutor

    config, registry = _build_registry(args)
    model, pipeline = _resolved_pipeline_for_model(args, config, registry)
    mode = RunMode(args.mode)
    market_data_client = AlpacaHistoricalDataClient(AlpacaMarketDataConfig.from_env())
    broker = AlpacaTradingBroker(AlpacaBrokerConfig.from_env(dry_run=args.dry_run))
    supervisor = Supervisor(config.thresholds)
    loop = SupervisorLoop(
        project=config.project_name,
        registry=registry,
        snapshot_store=PerformanceSnapshotStore(config.metrics_history_path),
        report_store=SupervisorReportStore(config.supervisor_reports_path),
        state_store=SupervisorStateStore(config.supervisor_state_path),
        supervisor=supervisor,
    )

    def retrain_callback(snapshot: PerformanceSnapshot) -> dict:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        new_model_id = f"{model.model_id}_{args.retrain_suffix}_{timestamp}"
        retrain_result = _train_alpaca_challenger(
            args=args,
            config=config,
            registry=registry,
            parent_model_id=model.model_id,
            new_model_id=new_model_id,
            snapshot=snapshot,
            market_data_client=market_data_client,
            trainer_source="supervisor_auto_retrain",
        )
        paper_record = registry.mark_paper_ready(
            new_model_id,
            notes=f"auto retrain challenger derivado de {model.model_id}",
        )
        return {
            "new_model": retrain_result["new_model"],
            "paper_record": paper_record.to_dict(),
            "training_payload": retrain_result["training_payload"],
        }

    cycle_results = []
    keep_cycle_results = args.iterations > 0
    for iteration in range(args.iterations):
        bars = market_data_client.load_recent_bars(
            symbol=pipeline.symbol,
            timeframe_amount=pipeline.timeframe_amount,
            timeframe_unit=pipeline.timeframe_unit,
            lookback_bars=pipeline.lookback_bars,
        )
        feature_df = engineer_features(bars)
        runtime = TradingRuntime(
            registry=registry,
            model_executor=TensorTradeModelExecutor(feature_df=feature_df, config=pipeline),
            risk_manager=AllowAllRiskManager(),
            broker=broker,
        )
        runtime.start(args.model_id, mode)
        position_qty = broker.current_position_qty(mode, pipeline.symbol)
        available_cash = broker.available_cash(mode)
        run_payload = runtime.run_once(
            model_id=args.model_id,
            mode=mode,
            features=feature_df.iloc[-1].to_dict(),
            context={
                "latest_close": float(feature_df["close"].iloc[-1]),
                "position_qty": position_qty,
                "available_cash": available_cash,
            },
        )
        execution_record = ExecutionResultStore(config.execution_history_path).append_result(
            project=config.project_name,
            model_id=args.model_id,
            mode=mode,
            payload={
                "iteration": iteration + 1,
                "latest_close": float(feature_df["close"].iloc[-1]),
                "result": run_payload,
                "pipeline": pipeline.to_dict(),
            },
        )

        collector = AlpacaMetricsCollector(
            trading_client=broker.client_for_mode(mode),
            market_data_client=market_data_client,
            config=_metrics_config_from_args(args, config),
        )
        snapshot = collector.collect_snapshot(
            model=model,
            symbol=pipeline.symbol,
            timeframe_amount=pipeline.timeframe_amount,
            timeframe_unit=pipeline.timeframe_unit,
            lookback_bars=pipeline.lookback_bars,
        )
        snapshot_record = PerformanceSnapshotStore(config.metrics_history_path).append_snapshot(
            project=config.project_name,
            model_id=args.model_id,
            mode=mode,
            snapshot=snapshot,
            source="operate_loop",
            notes=f"iteration={iteration + 1}",
        )
        supervise_result = loop.run_once(
            model_id=args.model_id,
            mode=mode,
            retrain_callback=None if args.no_auto_retrain else retrain_callback,
        )
        cycle_results.append(
            {
                "iteration": iteration + 1,
                "execution_record": execution_record,
                "snapshot_record": snapshot_record,
                "supervise_result": supervise_result.to_dict(),
            }
        )

        if iteration + 1 < args.iterations:
            time.sleep(args.sleep_seconds)

    print_json(
        {
            "project": config.project_name,
            "mode": mode.value,
            "iterations": args.iterations,
            "results": cycle_results,
        }
    )
    return 0


def _shadow_ledger_path(config: AppConfig, model_id: str) -> Path:
    return config.state_dir / "shadow_training" / f"{model_id}.json"


def _parse_bar_timestamp(bar_at: str | None) -> datetime | None:
    if not bar_at:
        return None
    try:
        parsed = datetime.fromisoformat(bar_at)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _market_status_label(bar_at: str | None) -> str:
    parsed = _parse_bar_timestamp(bar_at)
    if parsed is None:
        return "market=unknown"

    ny_tz = ZoneInfo("America/New_York")
    br_tz = ZoneInfo("America/Sao_Paulo")
    ny_time = parsed.astimezone(ny_tz)
    br_time = parsed.astimezone(br_tz)

    session = "closed"
    if ny_time.weekday() < 5:
        minutes = ny_time.hour * 60 + ny_time.minute
        if 570 <= minutes < 960:
            session = "open"
        elif 240 <= minutes < 570:
            session = "pre"
        elif 960 <= minutes < 1200:
            session = "after"

    return (
        f"market={session} "
        f"ny={ny_time.strftime('%Y-%m-%d %H:%M')} "
        f"br={br_time.strftime('%Y-%m-%d %H:%M')}"
    )


def _compact_execution_reason(reason: object) -> str | None:
    if not isinstance(reason, str):
        return None

    compact_parts: list[str] = []
    action_name = None

    if "masked_invalid_action" in reason:
        compact_parts.append("masked")
    if "blocked_no_position" in reason:
        compact_parts.append("no_position")
    if "sell bloqueado antes do broker" in reason:
        compact_parts.append("blocked_pre_broker")

    action_token = "tensortrade_action="
    if action_token in reason:
        action_fragment = reason.split(action_token, 1)[1].split()[0]
        action_name = f"a{action_fragment}"

    if " side=buy" in reason or "side=buy " in f"{reason} ":
        compact_parts.append("buy")
    elif " side=sell" in reason or "side=sell " in f"{reason} ":
        compact_parts.append("sell")
    elif " hold " in f" {reason} ":
        compact_parts.append("hold")

    selected_token = "selected_action="
    if selected_token in reason:
        selected_fragment = reason.split(selected_token, 1)[1].split()[0]
        compact_parts.append(f"selected=a{selected_fragment}")

    if action_name:
        compact_parts.insert(0, action_name)

    if not compact_parts:
        trimmed = reason.split("features=", 1)[0].strip()
        return trimmed[:120] if trimmed else None

    return ":".join(compact_parts)


def _format_execution_status(run_payload: dict[str, object], latest_close: float) -> str:
    intent = run_payload.get("intent", {}) if isinstance(run_payload, dict) else {}
    if not isinstance(intent, dict):
        intent = {}

    side = str(intent.get("side", "hold")).lower()
    reference_price = intent.get("reference_price", latest_close)
    try:
        reference_price_value = float(reference_price)
    except (TypeError, ValueError):
        reference_price_value = float(latest_close)

    status_parts = [f"decision={side}", f"px={reference_price_value:.2f}"]

    size_fraction = intent.get("size_fraction")
    if size_fraction is not None:
        try:
            status_parts.append(f"size={float(size_fraction) * 100:.1f}%")
        except (TypeError, ValueError):
            pass

    notional_usd = intent.get("notional_usd")
    if notional_usd is not None:
        try:
            notional_value = float(notional_usd)
        except (TypeError, ValueError):
            notional_value = 0.0
        if notional_value > 0:
            status_parts.append(f"notional=${notional_value:.2f}")

    if run_payload.get("submitted"):
        status_parts.append(f"exec={run_payload.get('reference', 'submitted')}")
    else:
        reason = run_payload.get("reason")
        compact_reason = _compact_execution_reason(reason)
        if compact_reason:
            status_parts.append(f"exec={compact_reason}")
        error = run_payload.get("error")
        if error:
            status_parts.append(f"error={error}")

    return " ".join(status_parts)


def _run_shadow_snapshot(
    *,
    args,
    config: AppConfig,
    registry: ModelRegistry,
    model_id: str,
    feature_df,
    latest_close: float,
    shadow_cash: float,
    source: str,
) -> dict:
    from ...integrations.alpaca import ShadowPaperLedger, compute_feature_drift
    from ...training import TensorTradeModelExecutor

    model, pipeline = _resolved_pipeline_for_model_id(args, config, registry, model_id)
    executor = TensorTradeModelExecutor(feature_df=feature_df, config=pipeline)
    intent = executor.predict_order(feature_df.iloc[-1].to_dict(), model)
    ledger = ShadowPaperLedger(
        path=_shadow_ledger_path(config, model_id),
        initial_cash_usd=shadow_cash,
    )
    if intent.side != OrderSide.HOLD:
        reference = ledger.submit(intent)
    else:
        reference = "shadow-noop-hold"

    ledger.record_mark_to_market(
        symbol_prices={pipeline.symbol: latest_close},
        source=source,
    )
    snapshot = ledger.build_snapshot(
        symbol=pipeline.symbol,
        feature_drift=compute_feature_drift(model.metadata.get("feature_reference", {}), feature_df),
        metadata={
            "source": source,
            "latest_close": latest_close,
            "reference": reference,
        },
    )
    return {
        "model": model,
        "pipeline": pipeline,
        "intent": intent.to_dict(),
        "reference": reference,
        "snapshot": snapshot,
    }


def handle_online_train(args) -> int:
    from ...features import engineer_features
    from ...integrations.alpaca import (
        AlpacaBrokerConfig,
        AlpacaHistoricalDataClient,
        AlpacaMarketDataConfig,
        AlpacaMetricsCollector,
        AlpacaTradingBroker,
        ChallengerPromotionPolicy,
        OnlineTrainingState,
        OnlineTrainingStateStore,
        ShadowPaperLedger,
        evaluate_challenger_promotion,
    )
    from ...runtime import AllowAllRiskManager, TradingRuntime
    from ...training import TensorTradeModelExecutor

    config, registry = _build_registry(args)
    market_data_client = AlpacaHistoricalDataClient(AlpacaMarketDataConfig.from_env())
    broker = AlpacaTradingBroker(AlpacaBrokerConfig.from_env(dry_run=args.dry_run))
    supervisor = Supervisor(config.thresholds)
    loop = SupervisorLoop(
        project=config.project_name,
        registry=registry,
        snapshot_store=PerformanceSnapshotStore(config.metrics_history_path),
        report_store=SupervisorReportStore(config.supervisor_reports_path),
        state_store=SupervisorStateStore(config.supervisor_state_path),
        supervisor=supervisor,
    )
    online_state_store = OnlineTrainingStateStore(config.state_dir / "online_training_state.json")
    shadow_store = JsonlStore(config.reports_dir / "online_training_shadow.jsonl")
    promotion_policy = ChallengerPromotionPolicy(
        evaluation_iterations=args.shadow_eval_iterations,
        evaluation_min_trades=args.shadow_min_trades,
        challenger_wins_required=args.shadow_wins_required,
        max_evaluation_iterations=args.shadow_max_eval_iterations,
        max_zero_trade_iterations=args.shadow_max_zero_trade_iterations,
        allow_rebase_promotion=not args.no_shadow_rebase_promotion,
    )

    registry_state = registry.state()
    active_paper_model_id = registry_state.get("paper_model_id") or args.model_id
    if registry_state.get("paper_model_id") is None:
        registry.mark_paper_ready(active_paper_model_id, notes="champion inicial do online-train")

    online_state = online_state_store.load()
    if online_state.champion_model_id != active_paper_model_id:
        online_state = OnlineTrainingState(
            champion_model_id=active_paper_model_id,
            last_seen_bar_at=online_state.last_seen_bar_at,
            last_processed_bar_at=online_state.last_processed_bar_at,
            last_challenger_retired_at=online_state.last_challenger_retired_at,
            last_challenger_retired_reason=online_state.last_challenger_retired_reason,
        )
        online_state_store.save(online_state)

    def spawn_challenger(
        *,
        trigger_snapshot: PerformanceSnapshot,
        champion_model_id: str,
        inherited_state: OnlineTrainingState,
        replacement_of: str | None = None,
        replacement_reason: str | None = None,
    ) -> dict:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        challenger_model_id = f"{champion_model_id}_{args.retrain_suffix}_{timestamp}"
        retrain_result = _train_alpaca_challenger(
            args=args,
            config=config,
            registry=registry,
            parent_model_id=champion_model_id,
            new_model_id=challenger_model_id,
            snapshot=trigger_snapshot,
            market_data_client=market_data_client,
            trainer_source="online_train_auto_retrain",
        )

        ShadowPaperLedger(
            path=_shadow_ledger_path(config, champion_model_id),
            initial_cash_usd=args.shadow_cash,
        ).reset()
        ShadowPaperLedger(
            path=_shadow_ledger_path(config, challenger_model_id),
            initial_cash_usd=args.shadow_cash,
        ).reset()

        next_state = OnlineTrainingState(
            champion_model_id=champion_model_id,
            challenger_model_id=challenger_model_id,
            evaluation_iterations=0,
            challenger_created_at=datetime.now(timezone.utc).isoformat(),
            last_seen_bar_at=inherited_state.last_seen_bar_at,
            last_processed_bar_at=inherited_state.last_processed_bar_at,
            last_cycle_status=inherited_state.last_cycle_status,
            last_challenger_retired_at=(
                datetime.now(timezone.utc).isoformat() if replacement_of else inherited_state.last_challenger_retired_at
            ),
            last_challenger_retired_reason=(
                (
                    f"substituido {replacement_of}: {replacement_reason}"
                    if replacement_of and replacement_reason
                    else f"substituido {replacement_of}"
                )
                if replacement_of
                else inherited_state.last_challenger_retired_reason
            ),
        )
        online_state_store.save(next_state)
        return {
            "new_model": retrain_result["new_model"],
            "training_payload": retrain_result["training_payload"],
            "shadow_state": next_state.to_dict(),
            "replacement_of": replacement_of,
            "replacement_reason": replacement_reason,
        }

    cycle_results = []
    keep_cycle_results = args.iterations > 0
    iteration = 0
    poll_iteration = 0
    while True:
        poll_iteration += 1
        active_paper_model_id = registry.state().get("paper_model_id") or active_paper_model_id
        if online_state.champion_model_id != active_paper_model_id:
            online_state = OnlineTrainingState(
                champion_model_id=active_paper_model_id,
                last_seen_bar_at=online_state.last_seen_bar_at,
                last_processed_bar_at=online_state.last_processed_bar_at,
                last_challenger_retired_at=online_state.last_challenger_retired_at,
                last_challenger_retired_reason=online_state.last_challenger_retired_reason,
            )
            online_state_store.save(online_state)

        model, pipeline = _resolved_pipeline_for_model_id(
            args,
            config,
            registry,
            active_paper_model_id,
        )
        try:
            bars = market_data_client.load_recent_bars(
                symbol=pipeline.symbol,
                timeframe_amount=pipeline.timeframe_amount,
                timeframe_unit=pipeline.timeframe_unit,
                lookback_bars=pipeline.lookback_bars,
            )
        except RuntimeError as exc:
            message = str(exc)
            if "Nenhum candle retornado" not in message and "Falha ao buscar candles" not in message:
                raise

            status = "waiting_market_data" if "Nenhum candle retornado" in message else "market_data_error"
            online_state.last_cycle_status = status
            online_state_store.save(online_state)
            wait_record = ExecutionResultStore(config.execution_history_path).append_result(
                project=config.project_name,
                model_id=active_paper_model_id,
                mode=RunMode.PAPER,
                payload={
                    "poll_iteration": poll_iteration,
                    "processed_iterations": iteration,
                    "source": f"online_train_{status}",
                    "status": status,
                    "symbol": pipeline.symbol,
                    "last_processed_bar_at": online_state.last_processed_bar_at,
                    "message": message,
                    "pipeline": pipeline.to_dict(),
                },
            )
            cycle_payload = {
                "poll_iteration": poll_iteration,
                "iteration": iteration,
                "active_paper_model_id": active_paper_model_id,
                "execution_record": wait_record,
                "snapshot_record": None,
                "supervise_result": None,
                "online_state": online_state.to_dict(),
                "shadow_result": None,
            }
            if keep_cycle_results:
                cycle_results.append(cycle_payload)

            print(
                (
                    f"[online-train] poll={poll_iteration} processed={iteration} "
                    f"status={status} model={active_paper_model_id} "
                    f"symbol={pipeline.symbol} last_bar={online_state.last_processed_bar_at or 'none'}"
                ),
                flush=True,
            )

            if args.iterations > 0 and iteration >= args.iterations:
                break

            time.sleep(args.sleep_seconds)
            continue

        feature_df = engineer_features(bars)
        latest_bar_at = bars.index[-1].isoformat() if hasattr(bars.index[-1], "isoformat") else str(bars.index[-1])
        latest_close = float(feature_df["close"].iloc[-1])

        if online_state.last_seen_bar_at == latest_bar_at:
            repeated_status = "waiting_new_bar"
            if online_state.last_cycle_status == "waiting_history":
                repeated_status = "waiting_history"
            online_state.last_cycle_status = repeated_status
            online_state_store.save(online_state)
            wait_record = ExecutionResultStore(config.execution_history_path).append_result(
                project=config.project_name,
                model_id=active_paper_model_id,
                mode=RunMode.PAPER,
                payload={
                    "poll_iteration": poll_iteration,
                    "processed_iterations": iteration,
                    "source": f"online_train_{repeated_status}",
                    "status": repeated_status,
                    "latest_bar_at": latest_bar_at,
                    "latest_close": latest_close,
                    "feature_rows": len(feature_df),
                    "pipeline": pipeline.to_dict(),
                },
            )
            cycle_payload = {
                "poll_iteration": poll_iteration,
                "iteration": iteration,
                "active_paper_model_id": active_paper_model_id,
                "execution_record": wait_record,
                "snapshot_record": None,
                "supervise_result": None,
                "online_state": online_state.to_dict(),
                "shadow_result": None,
            }
            if keep_cycle_results:
                cycle_results.append(cycle_payload)

            print(
                (
                    f"[online-train] poll={poll_iteration} processed={iteration} "
                    f"status={repeated_status} model={active_paper_model_id} "
                    f"symbol={pipeline.symbol} bar_at={latest_bar_at} "
                    f"{_market_status_label(latest_bar_at)}"
                ),
                flush=True,
            )

            if args.iterations > 0 and iteration >= args.iterations:
                break

            time.sleep(args.sleep_seconds)
            continue

        minimum_feature_rows = max(pipeline.window_size + 5, 50)
        if len(feature_df) < minimum_feature_rows:
            online_state.last_seen_bar_at = latest_bar_at
            online_state.last_cycle_status = "waiting_history"
            online_state_store.save(online_state)
            wait_record = ExecutionResultStore(config.execution_history_path).append_result(
                project=config.project_name,
                model_id=active_paper_model_id,
                mode=RunMode.PAPER,
                payload={
                    "poll_iteration": poll_iteration,
                    "processed_iterations": iteration,
                    "source": "online_train_waiting_history",
                    "status": "waiting_history",
                    "latest_bar_at": latest_bar_at,
                    "latest_close": latest_close,
                    "feature_rows": len(feature_df),
                    "minimum_feature_rows": minimum_feature_rows,
                    "pipeline": pipeline.to_dict(),
                },
            )
            cycle_payload = {
                "poll_iteration": poll_iteration,
                "iteration": iteration,
                "active_paper_model_id": active_paper_model_id,
                "execution_record": wait_record,
                "snapshot_record": None,
                "supervise_result": None,
                "online_state": online_state.to_dict(),
                "shadow_result": None,
            }
            if keep_cycle_results:
                cycle_results.append(cycle_payload)

            print(
                (
                    f"[online-train] poll={poll_iteration} processed={iteration} "
                    f"status=waiting_history model={active_paper_model_id} "
                    f"symbol={pipeline.symbol} bar_at={latest_bar_at} "
                    f"feature_rows={len(feature_df)}/{minimum_feature_rows} "
                    f"{_market_status_label(latest_bar_at)}"
                ),
                flush=True,
            )

            if args.iterations > 0 and iteration >= args.iterations:
                break

            time.sleep(args.sleep_seconds)
            continue

        iteration += 1

        runtime = TradingRuntime(
            registry=registry,
            model_executor=TensorTradeModelExecutor(feature_df=feature_df, config=pipeline),
            risk_manager=AllowAllRiskManager(),
            broker=broker,
        )
        runtime.start(active_paper_model_id, RunMode.PAPER)
        position_qty = broker.current_position_qty(RunMode.PAPER, pipeline.symbol)
        available_cash = broker.available_cash(RunMode.PAPER)
        run_payload = runtime.run_once(
            model_id=active_paper_model_id,
            mode=RunMode.PAPER,
            features=feature_df.iloc[-1].to_dict(),
            context={
                "latest_close": latest_close,
                "position_qty": position_qty,
                "available_cash": available_cash,
            },
        )
        execution_record = ExecutionResultStore(config.execution_history_path).append_result(
            project=config.project_name,
            model_id=active_paper_model_id,
            mode=RunMode.PAPER,
            payload={
                "iteration": iteration,
                "source": "online_train",
                "latest_close": latest_close,
                "result": run_payload,
                "pipeline": pipeline.to_dict(),
            },
        )

        collector = AlpacaMetricsCollector(
            trading_client=broker.client_for_mode(RunMode.PAPER),
            market_data_client=market_data_client,
            config=_metrics_config_from_args(args, config),
        )
        snapshot = collector.collect_snapshot(
            model=model,
            symbol=pipeline.symbol,
            timeframe_amount=pipeline.timeframe_amount,
            timeframe_unit=pipeline.timeframe_unit,
            lookback_bars=pipeline.lookback_bars,
        )
        snapshot_record = PerformanceSnapshotStore(config.metrics_history_path).append_snapshot(
            project=config.project_name,
            model_id=active_paper_model_id,
            mode=RunMode.PAPER,
            snapshot=snapshot,
            source="online_train",
            notes=f"iteration={iteration}",
        )

        def retrain_callback(trigger_snapshot: PerformanceSnapshot) -> dict:
            current_state = online_state_store.load()
            if current_state.challenger_model_id:
                return {
                    "skipped": True,
                    "reason": (
                        f"challenger {current_state.challenger_model_id} ja esta em avaliacao shadow"
                    ),
                }

            return spawn_challenger(
                trigger_snapshot=trigger_snapshot,
                champion_model_id=active_paper_model_id,
                inherited_state=current_state,
            )

        supervise_result = loop.run_once(
            model_id=active_paper_model_id,
            mode=RunMode.PAPER,
            retrain_callback=retrain_callback,
        )
        online_state = online_state_store.load()

        shadow_result = None
        if online_state.challenger_model_id:
            champion_shadow = _run_shadow_snapshot(
                args=args,
                config=config,
                registry=registry,
                model_id=online_state.champion_model_id,
                feature_df=feature_df,
                latest_close=latest_close,
                shadow_cash=args.shadow_cash,
                source="shadow_champion",
            )
            challenger_shadow = _run_shadow_snapshot(
                args=args,
                config=config,
                registry=registry,
                model_id=online_state.challenger_model_id,
                feature_df=feature_df,
                latest_close=latest_close,
                shadow_cash=args.shadow_cash,
                source="shadow_challenger",
            )
            online_state.evaluation_iterations += 1
            online_state_store.save(online_state)

            promotion_decision = evaluate_challenger_promotion(
                champion_snapshot=champion_shadow["snapshot"],
                challenger_snapshot=challenger_shadow["snapshot"],
                evaluation_iterations=online_state.evaluation_iterations,
                thresholds=config.thresholds,
                policy=promotion_policy,
            )
            shadow_result = {
                "champion_model_id": online_state.champion_model_id,
                "challenger_model_id": online_state.challenger_model_id,
                "evaluation_iterations": online_state.evaluation_iterations,
                "champion_shadow_snapshot": champion_shadow["snapshot"].to_dict(),
                "challenger_shadow_snapshot": challenger_shadow["snapshot"].to_dict(),
                "promotion_decision": promotion_decision,
            }
            shadow_store.append(
                {
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                    "project": config.project_name,
                    "payload": shadow_result,
                }
            )

            if (
                promotion_decision["eligible"]
                or promotion_decision.get("rebase_eligible")
                or promotion_decision.get("activity_rebase_eligible")
            ) and not args.no_auto_promote_paper:
                promotion_reason = (
                    f"promovido automaticamente em shadow apos {online_state.evaluation_iterations} iteracoes"
                )
                if promotion_decision.get("rebase_eligible") and not promotion_decision["eligible"]:
                    promotion_reason = (
                        "promovido automaticamente por rebase de regime "
                        f"apos {online_state.evaluation_iterations} iteracoes"
                    )
                if promotion_decision.get("activity_rebase_eligible") and not (
                    promotion_decision["eligible"] or promotion_decision.get("rebase_eligible")
                ):
                    promotion_reason = (
                        "promovido automaticamente por atividade shadow inicial "
                        f"apos {online_state.evaluation_iterations} iteracoes"
                    )
                paper_record = registry.mark_paper_ready(
                    online_state.challenger_model_id,
                    notes=promotion_reason,
                )
                online_state = OnlineTrainingState(champion_model_id=online_state.challenger_model_id)
                online_state_store.save(online_state)
                shadow_result["paper_promotion"] = paper_record.to_dict()
            elif promotion_decision.get("replacement_required"):
                retired_model_id = online_state.challenger_model_id
                replacement_reason = "; ".join(promotion_decision.get("replacement_reasons", []))
                replacement_result = spawn_challenger(
                    trigger_snapshot=snapshot,
                    champion_model_id=online_state.champion_model_id,
                    inherited_state=online_state,
                    replacement_of=retired_model_id,
                    replacement_reason=replacement_reason,
                )
                online_state = online_state_store.load()
                shadow_result["challenger_replaced"] = replacement_result

        online_state.last_seen_bar_at = latest_bar_at
        online_state.last_processed_bar_at = latest_bar_at
        online_state.last_cycle_status = "processed"
        online_state_store.save(online_state)

        cycle_payload = {
            "poll_iteration": poll_iteration,
            "iteration": iteration,
            "active_paper_model_id": active_paper_model_id,
            "execution_record": execution_record,
            "snapshot_record": snapshot_record,
            "supervise_result": supervise_result.to_dict(),
            "online_state": online_state.to_dict(),
            "shadow_result": shadow_result,
        }
        if keep_cycle_results:
            cycle_results.append(cycle_payload)

        report = supervise_result.report_record.get("report", {})
        execution_status = _format_execution_status(run_payload, latest_close)
        print(
            (
                f"[online-train] poll={poll_iteration} processed={iteration} "
                f"status=processed model={active_paper_model_id} symbol={pipeline.symbol} "
                f"bar_at={latest_bar_at} {execution_status} "
                f"supervisor={report.get('action', 'unknown')} "
                f"{_market_status_label(latest_bar_at)}"
            ),
            flush=True,
        )

        if args.stop_on_pause and report.get("action") == "pause":
            print_json(
                {
                    "project": config.project_name,
                    "stopped": True,
                    "reason": "supervisor_pause",
                    "last_cycle": cycle_payload,
                }
            )
            return 0

        if args.iterations > 0 and iteration >= args.iterations:
            break

        time.sleep(args.sleep_seconds)

    print_json(
        {
            "project": config.project_name,
            "mode": RunMode.PAPER.value,
            "iterations": iteration,
            "poll_iterations": poll_iteration,
            "results": cycle_results if keep_cycle_results else [],
            "last_cycle": cycle_payload if not keep_cycle_results else None,
        }
    )
    return 0


def handle_mark_paper_ready(args) -> int:
    _config, registry = _build_registry(args)
    record = registry.mark_paper_ready(args.model_id, notes=args.notes)
    print_json(record.to_dict())
    return 0


def handle_approve_live(args) -> int:
    _config, registry = _build_registry(args)
    record = registry.approve_for_live(
        args.model_id,
        approved_by=args.approved_by,
        notes=args.notes,
    )
    print_json(record.to_dict())
    return 0


def handle_activate_live(args) -> int:
    _config, registry = _build_registry(args)
    record = registry.activate_live(args.model_id, notes=args.notes)
    print_json(record.to_dict())
    return 0
