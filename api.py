"""Thin FastAPI adapter for the existing Insight AI LangGraph backend."""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

from main3 import db_registry, memory_manager, build_workflow

logger = logging.getLogger(__name__)


app = FastAPI(title="Insight AI API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
workflow = build_workflow()
threads: dict[str, dict[str, Any]] = {}


class AnalyzeRequest(BaseModel):
    thread_id: str = "thread_01"
    query: str = Field(min_length=1)


class ThreadCreateRequest(BaseModel):
    name: Optional[str] = None
    thread_id: Optional[str] = None


class DatabaseConnectRequest(BaseModel):
    thread_id: str
    database_uri: str = Field(min_length=1)


NODE_LABELS = {
    "intent_analyzer": "Intent Analysis",
    "decomposition_node": "Schema Retrieval",
    "sql_agent": "SQL Generation",
    "execution_planner": "SQL Validation",
    "answer_agent": "SQL Execution",
    "diagnostic_agent": "Diagnostic Analysis",
    "visualization_agent": "Visualization",
    "insights_agent": "Insights",
    "final_composer": "Final Analysis",
    "clarification_node": "Final Analysis",
    "unsupported_node": "Final Analysis",
    "security_blocked_node": "Final Analysis",
}


def _model_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: _model_dump(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_model_dump(item) for item in value]
    return value


def _state_payload(state: Any) -> dict[str, Any]:
    data = _model_dump(state)
    if not isinstance(data, dict):
        return {}
    if data.get("plotly_figure_json"):
        try:
            data["plotly_figure"] = json.loads(data["plotly_figure_json"])
        except (TypeError, json.JSONDecodeError):
            data["plotly_figure"] = None
    return {
        "thread_id": data.get("thread_id"),
        "user_query": data.get("user_query", ""),
        "schema_overview": data.get("schema_overview", ""),
        "intent": data.get("intent"),
        "execution_plan": data.get("execution_plan"),
        "generated_sql": data.get("generated_sql"),
        "validated_sql": data.get("validated_sql"),
        "executed_sqls": data.get("executed_sqls", []),
        "sql_error": data.get("sql_error"),
        "primary_result": data.get("primary_result"),
        "answer": data.get("answer"),
        "chart_spec": data.get("chart_spec"),
        "plotly_figure": data.get("plotly_figure"),
        "plotly_html_path": data.get("plotly_html_path"),
        "terminal_chart": data.get("terminal_chart"),
        "insights": data.get("insights", []),
        "diagnostic_report": data.get("diagnostic_report"),
        "final_response": data.get("final_response"),
        "security_blocked": data.get("security_blocked", False),
    }


def _memory_messages(thread_id: str, query: str) -> list[Any]:
    context = memory_manager.retrieve_context(thread_id, query)
    messages: list[Any] = []
    if context:
        messages.append(SystemMessage(content=context))
    messages.append(HumanMessage(content=query))
    return messages


def _input(request: AnalyzeRequest) -> dict[str, Any]:
    return {"thread_id": request.thread_id, "user_query": request.query, "messages": _memory_messages(request.thread_id, request.query)}


def _safe_error_message(exc: Exception) -> str:
    text = str(exc).lower()
    if "free-models-per-day" in text or "rate limit exceeded" in text or "rate_limit" in text:
        return "OpenRouter free-model daily limit reached. Wait for the quota reset, add credits, or configure another available model/provider."
    if "connection error" in text or "connecterror" in text or "10013" in text:
        return "LLM connection failed. Check the configured provider, API key, and network connection."
    if "sql" in text or "database" in text:
        return "Database analysis failed. Check the database connection and SQL error details."
    return "Analysis failed. Check the backend service logs for details."


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/threads")
def create_thread(request: ThreadCreateRequest | None = None) -> dict[str, Any]:
    request = request or ThreadCreateRequest()
    thread_id = request.thread_id or f"thread_{uuid4().hex[:10]}"
    value = {"thread_id": thread_id, "name": request.name or "New Analysis", "created_at": datetime.utcnow().isoformat()}
    threads[thread_id] = value
    return value


@app.get("/api/threads")
def list_threads() -> list[dict[str, Any]]:
    values = list(threads.values())
    if "thread_01" not in threads:
        values.insert(0, {"thread_id": "thread_01", "name": "Employee Analysis", "created_at": None})
    return values


@app.post("/api/database/connect")
def connect_database(request: DatabaseConnectRequest) -> dict[str, Any]:
    try:
        db_registry.register_database(request.thread_id, request.database_uri)
        threads.setdefault(request.thread_id, {"thread_id": request.thread_id, "name": "Database Analysis"})
        return {"thread_id": request.thread_id, "connected": True}
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Connection failed") from exc


@app.get("/api/schema/{thread_id}")
def get_schema(thread_id: str) -> dict[str, Any]:
    try:
        return {"thread_id": thread_id, "schema_overview": db_registry.get_clean_schema_overview(thread_id)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Schema unavailable") from exc


@app.get("/api/threads/{thread_id}/history")
def get_thread_history(thread_id: str) -> list[dict[str, Any]]:
    """Read completed structured results from this thread's existing checkpointer."""
    history: list[dict[str, Any]] = []
    seen: set[tuple[str, str | None]] = set()
    config = {"configurable": {"thread_id": thread_id}}
    try:
        for snapshot in workflow.get_state_history(config):
            payload = _state_payload(snapshot.values)
            query = payload.get("user_query")
            if not query or not payload.get("final_response"):
                continue
            checkpoint_id = snapshot.config.get("configurable", {}).get("checkpoint_id")
            key = (query, checkpoint_id)
            if key in seen:
                continue
            seen.add(key)
            history.append(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Thread history unavailable") from exc
    history.reverse()
    return history


@app.get("/api/chart")
def get_chart() -> FileResponse:
    chart_path = Path("chart.html").resolve()
    if not chart_path.exists():
        raise HTTPException(status_code=404, detail="Visualization unavailable")
    return FileResponse(chart_path, media_type="text/html")


@app.post("/api/analyze")
def analyze(request: AnalyzeRequest) -> dict[str, Any]:
    try:
        result = workflow.invoke(_input(request), config={"configurable": {"thread_id": request.thread_id}})
        payload = _state_payload(result)
        if not payload.get("security_blocked"):
            memory_manager.remember(request.thread_id, request.query, payload.get("final_response") or "")
        return payload
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_safe_error_message(exc)) from exc


async def _stream_analysis(request: AnalyzeRequest) -> AsyncIterator[str]:
    config = {"configurable": {"thread_id": request.thread_id}}
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
    active_steps: set[str] = set()

    def run_workflow() -> None:
        try:
            for event in workflow.stream(_input(request), config=config, stream_mode="debug"):
                loop.call_soon_threadsafe(queue.put_nowait, ("event", event))
            loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))

    try:
        yield f"data: {json.dumps({'type': 'connected', 'thread_id': request.thread_id})}\n\n"
        worker = asyncio.create_task(asyncio.to_thread(run_workflow))
        while True:
            event_type, event = await queue.get()
            if event_type == "error":
                logger.error("Insight AI workflow failed: %r", event)
                for step in active_steps:
                    yield f"data: {json.dumps({'type': 'workflow', 'step': step, 'label': NODE_LABELS.get(step, step), 'status': 'failed'})}\n\n"
                message = _safe_error_message(event)
                yield f"data: {json.dumps({'type': 'error', 'message': message})}\n\n"
                return
            if event_type == "done":
                break
            name = event.get("payload", {}).get("name", "")
            label = NODE_LABELS.get(name)
            if not label:
                continue
            debug_type = event.get("type")
            if debug_type == "task":
                active_steps.add(name)
                payload = {"type": "workflow", "step": name, "label": label, "status": "running"}
            elif debug_type == "task_result":
                active_steps.discard(name)
                payload = {"type": "workflow", "step": name, "label": label, "status": "completed"}
                output = event.get("payload", {}).get("result")
                if output:
                    payload["data"] = _state_payload(output)
            else:
                continue
            yield f"data: {json.dumps(payload, default=str)}\n\n"
        await worker

        result = _state_payload(workflow.get_state(config).values)
        if not result.get("security_blocked"):
            memory_manager.remember(request.thread_id, request.query, result.get("final_response") or "")
        yield f"data: {json.dumps({'type': 'result', 'data': result}, default=str)}\n\n"
        yield "data: {\"type\": \"done\"}\n\n"
    except Exception as exc:
        logger.exception("Insight AI streaming request failed")
        yield f"data: {json.dumps({'type': 'error', 'message': _safe_error_message(exc)})}\n\n"


@app.get("/api/analyze/{thread_id}/stream")
async def stream_analysis(thread_id: str, query: str = Query(min_length=1)) -> StreamingResponse:
    return StreamingResponse(_stream_analysis(AnalyzeRequest(thread_id=thread_id, query=query)), media_type="text/event-stream")
