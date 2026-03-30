from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

from ...cli_helpers import add_pipeline_args, pipeline_config_from_args, print_json
from ...config import AppConfig
from ...domain import PerformanceSnapshot, RunMode
from ...features import engineer_features
from ...registry import ModelRegistry
from ...storage import (
    ExecutionResultStore,
    PerformanceSnapshotStore,
    SupervisorReportStore,
    SupervisorStateStore,
    TrainingRunStore,
)
from ...supervisor import Supervisor
from ...supervisor_loop import SupervisorLoop


def _build_app_config(args) -> AppConfig:
    config = AppConfig(
        project_name="kucoin",
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
    parser.add_argument("--after-days", type=int, default=30)
    parser.add_argument("--trades-limit", type=int, default=500)
    parser.add_argument("--equity-curve-limit", type=int, default=500)


def _resolved_pipeline_for_model(args, config: AppConfig, registry: ModelRegistry):
    current_model = registry.get_model(args.model_id)
    base_training_job = current_model.metadata.get("training_job", {})
    pipeline = pipeline_config_from_args(
        args=args,
        artifacts_dir=config.artifacts_dir,
        model_id=args.model_id,
        base_data=base_training_job,
    )
    return current_model, pipeline


def _metrics_config_from_args(args):
    from ...integrations.kucoin import KuCoinMetricsConfig

    return KuCoinMetricsConfig(
        after_days=args.after_days,
        trades_limit=args.trades_limit,
        equity_curve_limit=args.equity_curve_limit,
    )


def _train_with_kucoin_market_data(
    *,
    registry: ModelRegistry,
    pipeline,
    notes: str,
    model_id: str,
):
    from ...integrations.kucoin import KuCoinHistoricalDataClient, KuCoinMarketDataConfig
    from ...training import TensorTradeTrainer

    market_data = KuCoinHistoricalDataClient(KuCoinMarketDataConfig.from_env())
    bars = market_data.load_recent_bars(
        symbol=pipeline.symbol,
        timeframe_amount=pipeline.timeframe_amount,
        timeframe_unit=pipeline.timeframe_unit,
        lookback_bars=pipeline.lookback_bars,
    )
    features = engineer_features(bars)
    trainer = TensorTradeTrainer(registry=registry)
    trainer.train_from_features(features, pipeline)
    record = trainer.register_training_output(
        model_id=model_id,
        config=pipeline,
        feature_df=features,
        notes=notes,
    )
    return record, features


def register_parser(subparsers) -> None:
    parser = subparsers.add_parser("kucoin", help="Comandos da integracao KuCoin.")
    parser.add_argument("--project-dir", default=None, help="Diretorio operacional opcional para o projeto KuCoin.")
    commands = parser.add_subparsers(dest="kucoin_command", required=True)

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

    train = commands.add_parser("train", help="Treina um modelo TensorTrade com dados da KuCoin.")
    add_pipeline_args(train, include_model_id=True, symbol_required=True)
    train.add_argument("--notes", default="")
    train.set_defaults(handler=handle_train)

    run_once = commands.add_parser("run", help="Executa um ciclo de inferencia e ordem na KuCoin.")
    add_pipeline_args(run_once, include_model_id=True, symbol_required=False)
    run_once.add_argument("--mode", choices=[mode.value for mode in RunMode], default=RunMode.PAPER.value)
    run_once.add_argument("--dry-run", action="store_true")
    run_once.set_defaults(handler=handle_run)

    collect_metrics = commands.add_parser("collect-metrics", help="Coleta automaticamente um snapshot de performance.")
    add_pipeline_args(collect_metrics, include_model_id=True, symbol_required=False)
    collect_metrics.add_argument("--mode", choices=[mode.value for mode in RunMode], default=RunMode.PAPER.value)
    _add_metrics_collection_args(collect_metrics)
    collect_metrics.add_argument("--source", default="kucoin_collector")
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
    print_json({"project": config.project_name, "state": registry.state()})
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
    config, registry = _build_registry(args)
    pipeline = pipeline_config_from_args(
        args=args,
        artifacts_dir=config.artifacts_dir,
        model_id=args.model_id,
    )
    record, features = _train_with_kucoin_market_data(
        registry=registry,
        pipeline=pipeline,
        notes=args.notes,
        model_id=args.model_id,
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
    from ...integrations.kucoin import (
        KuCoinBrokerConfig,
        KuCoinHistoricalDataClient,
        KuCoinMarketDataConfig,
        KuCoinTradingBroker,
    )
    from ...runtime import AllowAllRiskManager, TradingRuntime
    from ...training import TensorTradeModelExecutor

    config, registry = _build_registry(args)
    current_model, pipeline = _resolved_pipeline_for_model(args, config, registry)
    mode = RunMode(args.mode)
    market_data = KuCoinHistoricalDataClient(KuCoinMarketDataConfig.from_env())
    bars = market_data.load_recent_bars(
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
        broker=KuCoinTradingBroker(
            KuCoinBrokerConfig.from_env(
                state_dir=config.state_dir,
                dry_run=args.dry_run,
            )
        ),
    )
    runtime.start(current_model.model_id, mode)
    result = runtime.run_once(
        model_id=current_model.model_id,
        mode=mode,
        features=feature_df.iloc[-1].to_dict(),
        context={"latest_close": float(feature_df["close"].iloc[-1])},
    )
    payload = {
        "mode": mode.value,
        "latest_close": float(feature_df["close"].iloc[-1]),
        "result": result,
        "pipeline": pipeline.to_dict(),
    }
    ExecutionResultStore(config.execution_history_path).append_result(
        project=config.project_name,
        model_id=current_model.model_id,
        mode=mode,
        payload=payload,
    )
    print_json(payload)
    return 0


def handle_collect_metrics(args) -> int:
    from ...integrations.kucoin import (
        KuCoinBrokerConfig,
        KuCoinHistoricalDataClient,
        KuCoinMarketDataConfig,
        KuCoinMetricsCollector,
    )

    config, registry = _build_registry(args)
    model, pipeline = _resolved_pipeline_for_model(args, config, registry)
    mode = RunMode(args.mode)
    collector = KuCoinMetricsCollector(
        broker_config=KuCoinBrokerConfig.from_env(state_dir=config.state_dir),
        market_data_client=KuCoinHistoricalDataClient(KuCoinMarketDataConfig.from_env()),
        state_dir=config.state_dir,
        config=_metrics_config_from_args(args),
    )
    snapshot = collector.collect_snapshot(
        model=model,
        symbol=pipeline.symbol,
        timeframe_amount=pipeline.timeframe_amount,
        timeframe_unit=pipeline.timeframe_unit,
        lookback_bars=pipeline.lookback_bars,
        mode=mode,
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
    report, state = Supervisor(config.thresholds).review(
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
    from ...training import TensorTradePipelineConfig

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
        training_job = current_model.metadata.get("training_job")
        if not training_job:
            raise RuntimeError(
                f"modelo {current_model.model_id} nao possui metadados de treino para auto-retreino"
            )

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        new_model_id = f"{current_model.model_id}_{args.retrain_suffix}_{timestamp}"
        pipeline = TensorTradePipelineConfig.from_dict(training_job)
        pipeline.model_path = str(config.artifacts_dir / f"{new_model_id}.keras")
        record, features = _train_with_kucoin_market_data(
            registry=registry,
            pipeline=pipeline,
            notes=f"auto_retrain_from={current_model.model_id} snapshot_at={snapshot.as_of}",
            model_id=new_model_id,
        )
        paper_record = registry.mark_paper_ready(
            new_model_id,
            notes=f"auto retrain challenger derivado de {current_model.model_id}",
        )
        training_payload = {
            "rows_trained": len(features),
            "pipeline": pipeline.to_dict(),
            "source": "supervisor_auto_retrain",
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
            "paper_record": paper_record.to_dict(),
            "training_payload": training_payload,
        }

    result = loop.run_once(
        model_id=args.model_id,
        mode=RunMode(args.mode),
        retrain_callback=None if args.no_auto_retrain else retrain_callback,
    )
    print_json(result.to_dict())
    return 0


def handle_operate(args) -> int:
    from ...integrations.kucoin import (
        KuCoinBrokerConfig,
        KuCoinHistoricalDataClient,
        KuCoinMarketDataConfig,
        KuCoinMetricsCollector,
        KuCoinTradingBroker,
    )
    from ...runtime import AllowAllRiskManager, TradingRuntime
    from ...training import TensorTradeModelExecutor, TensorTradePipelineConfig

    config, registry = _build_registry(args)
    model, pipeline = _resolved_pipeline_for_model(args, config, registry)
    mode = RunMode(args.mode)
    market_data = KuCoinHistoricalDataClient(KuCoinMarketDataConfig.from_env())
    broker = KuCoinTradingBroker(
        KuCoinBrokerConfig.from_env(
            state_dir=config.state_dir,
            dry_run=args.dry_run,
        )
    )
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
        training_job = model.metadata.get("training_job")
        if not training_job:
            raise RuntimeError(f"modelo {model.model_id} nao possui metadados de treino para auto-retreino")

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        new_model_id = f"{model.model_id}_{args.retrain_suffix}_{timestamp}"
        retrain_pipeline = TensorTradePipelineConfig.from_dict(training_job)
        retrain_pipeline.model_path = str(config.artifacts_dir / f"{new_model_id}.keras")
        record, features = _train_with_kucoin_market_data(
            registry=registry,
            pipeline=retrain_pipeline,
            notes=f"auto_retrain_from={model.model_id} snapshot_at={snapshot.as_of}",
            model_id=new_model_id,
        )
        paper_record = registry.mark_paper_ready(
            new_model_id,
            notes=f"auto retrain challenger derivado de {model.model_id}",
        )
        training_payload = {
            "rows_trained": len(features),
            "pipeline": retrain_pipeline.to_dict(),
            "source": "supervisor_auto_retrain",
            "trigger_model_id": model.model_id,
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
            "paper_record": paper_record.to_dict(),
            "training_payload": training_payload,
        }

    cycle_results = []
    for iteration in range(args.iterations):
        bars = market_data.load_recent_bars(
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
        run_payload = runtime.run_once(
            model_id=args.model_id,
            mode=mode,
            features=feature_df.iloc[-1].to_dict(),
            context={"latest_close": float(feature_df["close"].iloc[-1])},
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

        collector = KuCoinMetricsCollector(
            broker_config=KuCoinBrokerConfig.from_env(state_dir=config.state_dir),
            market_data_client=market_data,
            state_dir=config.state_dir,
            config=_metrics_config_from_args(args),
        )
        snapshot = collector.collect_snapshot(
            model=model,
            symbol=pipeline.symbol,
            timeframe_amount=pipeline.timeframe_amount,
            timeframe_unit=pipeline.timeframe_unit,
            lookback_bars=pipeline.lookback_bars,
            mode=mode,
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
