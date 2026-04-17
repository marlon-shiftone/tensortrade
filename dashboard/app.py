from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


ROOT_DIR = Path(__file__).resolve().parents[1]
ONLINE_STATUS_META = {
    "processed": {
        "label": "Processed",
        "color": "#166534",
        "background": "#dcfce7",
        "description": "Ciclo completo executado com barra nova.",
    },
    "waiting_new_bar": {
        "label": "Waiting New Bar",
        "color": "#92400e",
        "background": "#fef3c7",
        "description": "Mercado respondeu, mas ainda sem nova barra processavel.",
    },
    "waiting_market_data": {
        "label": "Waiting Market Data",
        "color": "#991b1b",
        "background": "#fee2e2",
        "description": "A integracao nao retornou candles neste poll.",
    },
    "market_data_error": {
        "label": "Market Data Error",
        "color": "#7f1d1d",
        "background": "#fecaca",
        "description": "Falha transitoria ao consultar a API de market data.",
    },
    "waiting_history": {
        "label": "Waiting History",
        "color": "#1d4ed8",
        "background": "#dbeafe",
        "description": "Existe barra nova, mas ainda falta historico minimo para o TensorTrade.",
    },
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def candidate_projects(root_dir: Path) -> list[Path]:
    candidates = []
    for child in sorted(root_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("_"):
            continue
        if (child / "state" / "model_registry.json").exists():
            candidates.append(child)
    return candidates


def auto_refresh(enabled: bool, refresh_seconds: int) -> None:
    if not enabled:
        return
    milliseconds = max(1, refresh_seconds) * 1000
    components.html(
        f"""
        <script>
            setTimeout(function() {{
                window.parent.location.reload();
            }}, {milliseconds});
        </script>
        """,
        height=0,
    )


def flatten_snapshot_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records:
        snapshot = record.get("snapshot", {})
        rows.append(
            {
                "recorded_at": record.get("recorded_at"),
                "model_id": record.get("model_id"),
                "mode": record.get("mode"),
                "source": record.get("source"),
                "symbol": snapshot.get("symbol"),
                "net_profit": snapshot.get("net_profit"),
                "rolling_sharpe": snapshot.get("rolling_sharpe"),
                "max_drawdown": snapshot.get("max_drawdown"),
                "hit_rate": snapshot.get("hit_rate"),
                "turnover": snapshot.get("turnover"),
                "feature_drift": snapshot.get("feature_drift"),
                "paper_days": snapshot.get("paper_days"),
                "trades": snapshot.get("trades"),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["recorded_at"] = pd.to_datetime(df["recorded_at"], utc=True, errors="coerce")
    return df


def flatten_execution_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records:
        payload = record.get("payload", {})
        result = payload.get("result", {})
        intent = result.get("intent", {})
        rows.append(
            {
                "recorded_at": record.get("recorded_at"),
                "model_id": record.get("model_id"),
                "mode": record.get("mode"),
                "poll_iteration": payload.get("poll_iteration"),
                "iteration": payload.get("iteration"),
                "processed_iterations": payload.get("processed_iterations"),
                "source": payload.get("source"),
                "status": payload.get("status"),
                "latest_bar_at": payload.get("latest_bar_at"),
                "latest_close": payload.get("latest_close"),
                "feature_rows": payload.get("feature_rows"),
                "minimum_feature_rows": payload.get("minimum_feature_rows"),
                "message": payload.get("message"),
                "submitted": result.get("submitted"),
                "reference": result.get("reference"),
                "reason": result.get("reason"),
                "error_type": result.get("error_type"),
                "error": result.get("error"),
                "side": intent.get("side"),
                "symbol": intent.get("symbol"),
                "notional_usd": intent.get("notional_usd"),
                "size_fraction": intent.get("size_fraction"),
                "confidence": intent.get("confidence"),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["recorded_at"] = pd.to_datetime(df["recorded_at"], utc=True, errors="coerce")
        if "latest_bar_at" in df.columns:
            df["latest_bar_at"] = pd.to_datetime(df["latest_bar_at"], utc=True, errors="coerce")
    return df


def flatten_supervisor_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records:
        report = record.get("report", {})
        evaluation = report.get("evaluation", {})
        rows.append(
            {
                "recorded_at": record.get("recorded_at"),
                "model_id": record.get("model_id"),
                "mode": record.get("mode"),
                "action": report.get("action"),
                "requires_human_review": report.get("requires_human_review"),
                "healthy": evaluation.get("healthy"),
                "allow_live_candidate": evaluation.get("allow_live_candidate"),
                "requires_retraining": evaluation.get("requires_retraining"),
                "pause_trading": evaluation.get("pause_trading"),
                "reasons": " | ".join(report.get("reasons", [])),
                "consecutive_bad_windows": record.get("extra", {})
                .get("supervisor_state", {})
                .get("consecutive_bad_windows"),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["recorded_at"] = pd.to_datetime(df["recorded_at"], utc=True, errors="coerce")
    return df


def flatten_shadow_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records:
        payload = record.get("payload", {})
        decision = payload.get("promotion_decision", {})
        rows.append(
            {
                "recorded_at": record.get("recorded_at"),
                "champion_model_id": payload.get("champion_model_id"),
                "challenger_model_id": payload.get("challenger_model_id"),
                "evaluation_iterations": payload.get("evaluation_iterations"),
                "challenger_wins": decision.get("challenger_wins"),
                "eligible": decision.get("eligible"),
                "required_wins": decision.get("policy", {}).get("challenger_wins_required"),
                "champion_net_profit": payload.get("champion_shadow_snapshot", {}).get("net_profit"),
                "challenger_net_profit": payload.get("challenger_shadow_snapshot", {}).get("net_profit"),
                "champion_sharpe": payload.get("champion_shadow_snapshot", {}).get("rolling_sharpe"),
                "challenger_sharpe": payload.get("challenger_shadow_snapshot", {}).get("rolling_sharpe"),
                "champion_trades": payload.get("champion_shadow_snapshot", {}).get("trades"),
                "challenger_trades": payload.get("challenger_shadow_snapshot", {}).get("trades"),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["recorded_at"] = pd.to_datetime(df["recorded_at"], utc=True, errors="coerce")
    return df


def metric_value(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def status_meta(status: str | None) -> dict[str, str]:
    if not status:
        return {
            "label": "Unknown",
            "color": "#374151",
            "background": "#f3f4f6",
            "description": "Sem status online persistido.",
        }
    return ONLINE_STATUS_META.get(
        status,
        {
            "label": status.replace("_", " ").title(),
            "color": "#374151",
            "background": "#f3f4f6",
            "description": "Status operacional nao categorizado.",
        },
    )


def render_status_banner(online_state: dict[str, Any]) -> None:
    status = online_state.get("last_cycle_status")
    meta = status_meta(status)
    last_seen_bar_at = online_state.get("last_seen_bar_at") or "-"
    last_processed_bar_at = online_state.get("last_processed_bar_at") or "-"
    evaluation_iterations = online_state.get("evaluation_iterations", 0)
    st.markdown(
        f"""
        <div style="
            border-radius: 12px;
            padding: 14px 16px;
            margin: 6px 0 16px 0;
            background: {meta["background"]};
            border: 1px solid {meta["color"]}33;
        ">
            <div style="font-size: 0.85rem; color: {meta["color"]}; font-weight: 700; letter-spacing: 0.03em;">
                ONLINE STATUS
            </div>
            <div style="font-size: 1.35rem; color: {meta["color"]}; font-weight: 700; margin-top: 2px;">
                {meta["label"]}
            </div>
            <div style="font-size: 0.95rem; color: {meta["color"]}; margin-top: 4px;">
                {meta["description"]}
            </div>
            <div style="font-size: 0.9rem; color: #374151; margin-top: 10px;">
                <strong>Last seen bar:</strong> {last_seen_bar_at}
                &nbsp;&nbsp;|&nbsp;&nbsp;
                <strong>Last processed bar:</strong> {last_processed_bar_at}
                &nbsp;&nbsp;|&nbsp;&nbsp;
                <strong>Shadow eval iterations:</strong> {evaluation_iterations}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_state_panel(project_dir: Path, registry_state: dict[str, Any], online_state: dict[str, Any]) -> None:
    st.subheader("Estado Atual")
    render_status_banner(online_state or {})
    paper_model_id = registry_state.get("paper_model_id")
    live_model_id = registry_state.get("live_model_id")
    champion_model_id = online_state.get("champion_model_id")
    challenger_model_id = online_state.get("challenger_model_id")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Paper Model", paper_model_id or "-")
    col2.metric("Live Model", live_model_id or "-")
    col3.metric("Champion", champion_model_id or "-")
    col4.metric("Challenger", challenger_model_id or "-")
    col5.metric("Last Cycle", online_state.get("last_cycle_status") or "-")

    st.caption(f"Projeto operacional: `{project_dir}`")
    st.json(online_state or {"status": "sem estado online"})


def show_registry_panel(registry_state: dict[str, Any]) -> None:
    st.subheader("Registry")
    models = registry_state.get("models", {})
    if not models:
        st.info("Nenhum modelo registrado.")
        return

    rows = []
    for model_id, payload in models.items():
        metadata = payload.get("metadata", {})
        training_job = metadata.get("training_job", {})
        rows.append(
            {
                "model_id": model_id,
                "stage": payload.get("stage"),
                "trainer": payload.get("trainer"),
                "symbol": training_job.get("symbol"),
                "created_at": payload.get("created_at"),
                "artifact_path": payload.get("artifact_path"),
                "approved_for_live": payload.get("human_approved_for_live"),
            }
        )
    df = pd.DataFrame(rows).sort_values("created_at", ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True)


def show_snapshots_panel(snapshot_df: pd.DataFrame) -> None:
    st.subheader("Performance")
    if snapshot_df.empty:
        st.info("Nenhum snapshot de performance encontrado.")
        return

    latest = snapshot_df.sort_values("recorded_at").iloc[-1]
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Net Profit", metric_value(latest["net_profit"], 2))
    col2.metric("Sharpe", metric_value(latest["rolling_sharpe"], 2))
    col3.metric("Drawdown", metric_value(latest["max_drawdown"], 4))
    col4.metric("Hit Rate", metric_value(latest["hit_rate"], 3))
    col5.metric("Feature Drift", metric_value(latest["feature_drift"], 3))

    chart_columns = ["net_profit", "rolling_sharpe", "max_drawdown", "hit_rate", "feature_drift"]
    chart_df = snapshot_df.sort_values("recorded_at").set_index("recorded_at")[chart_columns]
    st.line_chart(chart_df, use_container_width=True)
    st.dataframe(
        snapshot_df.sort_values("recorded_at", ascending=False),
        use_container_width=True,
        hide_index=True,
    )


def show_supervisor_panel(supervisor_df: pd.DataFrame) -> None:
    st.subheader("Supervisor")
    if supervisor_df.empty:
        st.info("Nenhum relatorio do supervisor encontrado.")
        return

    latest = supervisor_df.sort_values("recorded_at").iloc[-1]
    col1, col2, col3 = st.columns(3)
    col1.metric("Ultima Acao", latest["action"] or "-")
    col2.metric("Bad Windows", latest["consecutive_bad_windows"] or 0)
    col3.metric("Requires Human Review", str(bool(latest["requires_human_review"])))

    action_counts = supervisor_df["action"].value_counts().rename_axis("action").reset_index(name="count")
    left, right = st.columns((1, 2))
    left.dataframe(action_counts, use_container_width=True, hide_index=True)
    right.dataframe(
        supervisor_df.sort_values("recorded_at", ascending=False),
        use_container_width=True,
        hide_index=True,
    )


def show_execution_panel(execution_df: pd.DataFrame) -> None:
    st.subheader("Execucao")
    if execution_df.empty:
        st.info("Nenhum resultado de execucao encontrado.")
        return

    latest = execution_df.sort_values("recorded_at").iloc[-1]
    latest_status = latest["status"] if "status" in latest and pd.notna(latest["status"]) else "processed"
    latest_status_meta = status_meta(str(latest_status))
    st.markdown(
        f"""
        <div style="
            border-radius: 10px;
            padding: 10px 12px;
            margin-bottom: 12px;
            background: {latest_status_meta["background"]};
            color: {latest_status_meta["color"]};
            border: 1px solid {latest_status_meta["color"]}33;
            font-weight: 600;
        ">
            Execution status: {latest_status_meta["label"]}
        </div>
        """,
        unsafe_allow_html=True,
    )

    processed_iterations = (
        latest["processed_iterations"]
        if "processed_iterations" in latest.index and pd.notna(latest["processed_iterations"])
        else latest["iteration"] if pd.notna(latest["iteration"]) else "-"
    )
    latest_bar_display = "-"
    if "latest_bar_at" in latest.index and pd.notna(latest["latest_bar_at"]):
        latest_bar_display = latest["latest_bar_at"].isoformat()

    feature_rows_display = "-"
    if "feature_rows" in latest.index and pd.notna(latest["feature_rows"]):
        feature_rows_display = f"{int(latest['feature_rows'])}"
        if "minimum_feature_rows" in latest.index and pd.notna(latest["minimum_feature_rows"]):
            feature_rows_display = f"{feature_rows_display}/{int(latest['minimum_feature_rows'])}"

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(
        "Ultimo Poll",
        latest["poll_iteration"] if "poll_iteration" in latest.index and pd.notna(latest["poll_iteration"]) else "-",
    )
    col2.metric("Iteracoes Processadas", processed_iterations)
    col3.metric("Ultimo Lado", latest["side"] or "-")
    col4.metric("Submitted", str(bool(latest["submitted"])))
    col5.metric("Latest Close", metric_value(latest["latest_close"], 2))

    info_col1, info_col2, info_col3 = st.columns(3)
    info_col1.metric("Status", latest_status_meta["label"])
    info_col2.metric("Latest Bar", latest_bar_display)
    info_col3.metric("Feature Rows", feature_rows_display)

    if "message" in latest.index and pd.notna(latest["message"]):
        st.caption(f"Mensagem operacional: {latest['message']}")
    elif "error" in latest.index and pd.notna(latest["error"]):
        st.caption(f"Erro de execucao: {latest.get('error_type')}: {latest['error']}")

    st.dataframe(
        execution_df.sort_values("recorded_at", ascending=False),
        use_container_width=True,
        hide_index=True,
    )


def show_shadow_panel(shadow_df: pd.DataFrame, shadow_records: list[dict[str, Any]]) -> None:
    st.subheader("Shadow Evaluation")
    if shadow_df.empty:
        st.info("Nenhuma avaliacao shadow encontrada.")
        return

    latest = shadow_df.sort_values("recorded_at").iloc[-1]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Evaluation Iterations", latest["evaluation_iterations"] or 0)
    col2.metric("Challenger Wins", latest["challenger_wins"] or 0)
    col3.metric("Required Wins", latest["required_wins"] or 0)
    col4.metric("Eligible", str(bool(latest["eligible"])))

    left, right = st.columns((2, 1))
    left.dataframe(
        shadow_df.sort_values("recorded_at", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
    latest_payload = shadow_records[-1].get("payload", {}) if shadow_records else {}
    right.json(latest_payload)


def show_training_runs_panel(training_runs: list[dict[str, Any]]) -> None:
    st.subheader("Training Runs")
    if not training_runs:
        st.info("Nenhum treino registrado.")
        return

    rows = []
    for record in training_runs:
        payload = record.get("payload", {})
        rows.append(
            {
                "recorded_at": record.get("recorded_at"),
                "model_id": record.get("model_id"),
                "trainer": record.get("trainer"),
                "source": payload.get("source"),
                "rows_trained": payload.get("rows_trained"),
                "trigger_model_id": payload.get("trigger_model_id"),
            }
        )
    df = pd.DataFrame(rows).sort_values("recorded_at", ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(
        page_title="Agent Trader Dashboard",
        layout="wide",
    )
    st.title("Agent Trader Dashboard")
    st.caption("Painel operacional para os projetos Alpaca, KuCoin e outros projetos compativeis.")

    projects = candidate_projects(ROOT_DIR)
    if not projects:
        st.error("Nenhum projeto operacional encontrado no root do repositorio.")
        return

    project_names = [project.name for project in projects]
    query_params = st.query_params
    requested_project = query_params.get("project", "alpaca")
    if requested_project not in project_names:
        requested_project = "alpaca" if "alpaca" in project_names else project_names[0]
    default_index = project_names.index(requested_project)

    with st.sidebar:
        selected_name = st.selectbox("Projeto", options=project_names, index=default_index)
        auto = st.checkbox("Auto-refresh", value=query_params.get("auto", "true") == "true")
        refresh_seconds = st.slider(
            "Refresh (segundos)",
            min_value=5,
            max_value=120,
            value=int(query_params.get("refresh", 10)),
            step=5,
        )
        history_limit = st.slider(
            "Linhas recentes",
            min_value=20,
            max_value=1000,
            value=int(query_params.get("history", 200)),
            step=20,
        )
        st.caption("Arquivos lidos diretamente do diretorio operacional.")

    st.query_params["project"] = selected_name
    st.query_params["auto"] = str(auto).lower()
    st.query_params["refresh"] = str(refresh_seconds)
    st.query_params["history"] = str(history_limit)

    auto_refresh(auto, refresh_seconds)

    project_dir = next(project for project in projects if project.name == selected_name)
    reports_dir = project_dir / "reports"
    state_dir = project_dir / "state"

    registry_state = read_json(state_dir / "model_registry.json")
    online_state = read_json(state_dir / "online_training_state.json")
    execution_records = read_jsonl(reports_dir / "execution_results.jsonl")[-history_limit:]
    snapshot_records = read_jsonl(reports_dir / "performance_snapshots.jsonl")[-history_limit:]
    supervisor_records = read_jsonl(reports_dir / "supervisor_reports.jsonl")[-history_limit:]
    shadow_records = read_jsonl(reports_dir / "online_training_shadow.jsonl")[-history_limit:]
    training_runs = read_jsonl(reports_dir / "training_runs.jsonl")[-history_limit:]

    snapshot_df = flatten_snapshot_records(snapshot_records)
    execution_df = flatten_execution_records(execution_records)
    supervisor_df = flatten_supervisor_records(supervisor_records)
    shadow_df = flatten_shadow_records(shadow_records)

    show_state_panel(project_dir, registry_state, online_state)

    top_left, top_right = st.columns((2, 3))
    with top_left:
        show_registry_panel(registry_state)
    with top_right:
        show_snapshots_panel(snapshot_df)

    mid_left, mid_right = st.columns((2, 3))
    with mid_left:
        show_supervisor_panel(supervisor_df)
    with mid_right:
        show_execution_panel(execution_df)

    bottom_left, bottom_right = st.columns((3, 2))
    with bottom_left:
        show_shadow_panel(shadow_df, shadow_records)
    with bottom_right:
        show_training_runs_panel(training_runs)


if __name__ == "__main__":
    main()
