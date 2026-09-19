"""
Insight AI — Complete Single-File Production Data Analyst
- Uses `create_sql_agent` with SQLDatabaseToolkit for BOTH SQL generation and Diagnostic analysis
- Timeout-hardened & early-stopping enabled (prevents hanging/freezing)
- Dynamic Schema Introspection (SQLAlchemy) & Dynamic Vector RAG (ChromaDB)
- Multi-Model LLM Factory: OpenRouter (NVIDIA / Gemini / OpenAI) + Ollama fallback
- Complete 6-Section Structured Output for general / analytical queries:
    1. 💡 Direct Answer
    2. 🔍 Executed SQL Query
    3. 📋 Real Column-Named Results Table (No generic 'col_1' placeholders)
    4. 📊 Dual Visualization (In-Terminal Proportional Visuals + Standalone Interactive 'chart.html')
    5. 🔬 Automated Multi-Hypothesis Diagnostic Analysis (via create_sql_agent)
    6. 📈 Quantitative Key Insights
- Granular Intent Routing ("give me sql only", "diagnosis only", "chart only", "details")
"""

import os
import re
import ast
import json
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence

import pandas as pd
import plotly.express as px
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

try:
    import chromadb
except ImportError:
    chromadb = None

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from security.security_gateway import SecurityError, security_gateway
from security.sql_security import ReadOnlyDatabaseProxy
from memory.memory_manager import MemoryManager

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# =====================================================================
# 1. SETTINGS & LLM FACTORY (OpenRouter + Ollama Fallback)
# =====================================================================
class Settings(BaseModel):
    llm_provider: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openrouter"))
    
    # OpenRouter Settings (NVIDIA, Gemini, OpenAI, DeepSeek, Meta)
    openrouter_api_key: str = Field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    openrouter_model: str = Field(default_factory=lambda: os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free"))
    fallback_model: str = Field(default_factory=lambda: os.getenv("FALLBACK_MODEL", "openrouter/free"))
    
    # Ollama Local Settings
    ollama_base_url: str = Field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_model: str = Field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.2:latest"))
    
    # Database Settings
    default_database_uri: str = Field(
        default_factory=lambda: os.getenv("DEFAULT_DATABASE_URI") or os.getenv("DATABASE_URL") or ""
    )
    
    confidence_threshold: float = 0.70
    max_sql_repairs: int = 3
    agent_max_iterations: int = 6
    agent_timeout_seconds: float = 30.0
    auto_open_browser: bool = False

settings = Settings()

_ollama_fallback_until = 0.0


def _activate_ollama_fallback(exc: Exception) -> None:
    """Temporarily route later calls to Ollama after an OpenRouter failure."""
    global _ollama_fallback_until
    if settings.llm_provider.lower().strip() == "openrouter":
        _ollama_fallback_until = time.monotonic() + 300.0


def _using_ollama_fallback() -> bool:
    return time.monotonic() < _ollama_fallback_until


def invoke_llm(llm: Any, messages: Any):
    try:
        return llm.invoke(messages)
    except Exception as exc:
        _activate_ollama_fallback(exc)
        raise

def get_llm(temperature: float = 0.0, streaming: bool = False, use_fallback: bool = False):
    provider = settings.llm_provider.lower().strip()

    if provider == "openrouter" and (use_fallback or _using_ollama_fallback()):
        provider = "ollama"
    
    if provider == "openrouter":
        from langchain_openai import ChatOpenAI
        api_key = settings.openrouter_api_key or os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set in your .env file.")
        
        target_model = settings.fallback_model if use_fallback else settings.openrouter_model
        return ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            model=target_model,
            temperature=temperature,
            streaming=streaming,
            max_retries=0,
            timeout=min(settings.agent_timeout_seconds, security_gateway.config.llm_timeout_seconds),
            default_headers={
                "HTTP-Referer": "https://github.com/insight-ai",
                "X-Title": "Insight AI Analyst"
            }
        )
    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=temperature,
            streaming=streaming,
            client_kwargs={"timeout": security_gateway.config.llm_timeout_seconds},
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model=os.getenv("DEFAULT_MODEL", "gpt-4o-mini"),
            temperature=temperature,
            streaming=streaming,
            max_retries=0,
            timeout=min(settings.agent_timeout_seconds, security_gateway.config.llm_timeout_seconds),
        )


# =====================================================================
# 2. DATABASE REGISTRY & DYNAMIC SCHEMA INTROSPECTION
# =====================================================================
class DatabaseRegistry:
    def __init__(self) -> None:
        self._engines: Dict[str, Engine] = {}
        self._uris: Dict[str, str] = {}

    def register_database(self, thread_id: str, db_uri: str) -> None:
        if not db_uri.strip():
            raise ValueError("Database URI cannot be empty.")
        self._uris[thread_id] = db_uri.strip()
        self._engines[thread_id] = create_engine(db_uri.strip())

    def get_uri(self, thread_id: str) -> str:
        uri = self._uris.get(thread_id, settings.default_database_uri)
        if not uri:
            raise ValueError("No database URI found! Please set DEFAULT_DATABASE_URI in your .env file.")
        return uri

    def get_engine(self, thread_id: str) -> Engine:
        if thread_id not in self._engines:
            self._engines[thread_id] = create_engine(self.get_uri(thread_id))
        return self._engines[thread_id]

    def get_db(self, thread_id: str) -> SQLDatabase:
        return SQLDatabase.from_uri(self.get_uri(thread_id))

    def get_toolkit(self, thread_id: str, llm: Any) -> SQLDatabaseToolkit:
        return SQLDatabaseToolkit(db=self.get_db(thread_id), llm=llm)

    def get_clean_schema_overview(self, thread_id: str) -> str:
        """Dynamically inspects any database catalog without assumptions."""
        engine = self.get_engine(thread_id)
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        if not tables:
            return "No tables found in the connected database."

        schema_lines = []
        for table in tables:
            columns = inspector.get_columns(table)
            col_desc = [f"{col['name']} ({str(col['type'])})" for col in columns]
            schema_lines.append(f"Table `{table}`: " + ", ".join(col_desc))
            fks = inspector.get_foreign_keys(table)
            for fk in fks:
                schema_lines.append(
                    f"  └─ FK: `{table}`.{fk['constrained_columns']} -> `{fk['referred_table']}`.{fk['referred_columns']}"
                )
        return "\n".join(schema_lines)

    def run_query_with_columns(self, thread_id: str, query: str) -> tuple[List[str], List[Dict[str, Any]]]:
        """Executes a SQL query and preserves exact database column headers."""
        query = security_gateway.validate_sql(query, thread_id)
        engine = self.get_engine(thread_id)

        def _execute():
            with engine.connect() as conn:
                result = conn.execute(text(query))
                col_names = list(result.keys())
                rows = [dict(zip(col_names, row)) for row in result.fetchmany(security_gateway.config.max_rows)]
                return col_names, security_gateway.bound_rows(rows)

        return security_gateway.run_with_timeout(_execute, security_gateway.config.sql_timeout_seconds)

db_registry = DatabaseRegistry()
memory_manager = MemoryManager()


def security_safe_for_llm(value: Any) -> Any:
    return security_gateway.mask_for_llm(value)


# =====================================================================
# 3. DYNAMIC HYBRID RAG STORE
# =====================================================================
class SemanticContext(BaseModel):
    schema_docs: List[str] = Field(default_factory=list)
    general_guidelines: List[str] = Field(default_factory=list)

class SemanticRAGStore:
    def __init__(self) -> None:
        if chromadb:
            self.client = chromadb.Client()
            self.collection = self.client.get_or_create_collection("dynamic_database_knowledge")
        else:
            self.collection = None
        self._synced_threads = set()

    def sync_database_knowledge(self, thread_id: str) -> None:
        if not self.collection or thread_id in self._synced_threads:
            return

        try:
            db = db_registry.get_db(thread_id)
            tables = db.get_usable_table_names()
            docs, metadatas, ids = [], [], []

            for i, table in enumerate(tables):
                info = db.get_table_info([table])
                docs.append(f"Database Table `{table}` Schema: {info}")
                metadatas.append({"table": table, "type": "ddl_schema"})
                ids.append(f"{thread_id}_table_{table}_{i}")

            guidelines = [
                "Use the exact table and column names returned in the schema; never invent or rename tables.",
                "For hierarchical relationships (e.g. manager pointing to employee ID in same table), perform a self-join.",
                "Match string values case-insensitively using exact uppercase matching (e.g. column = 'VALUE').",
                "Only query and join tables that contain the columns required to answer the question.",
                "In GROUP BY queries, include all non-aggregated SELECT columns in the GROUP BY clause."
            ]

            for idx, g in enumerate(guidelines):
                docs.append(g)
                metadatas.append({"table": "general", "type": "guideline"})
                ids.append(f"{thread_id}_guideline_{idx}")

            if docs:
                self.collection.add(documents=docs, metadatas=metadatas, ids=ids)
            self._synced_threads.add(thread_id)
        except Exception:
            pass

    def retrieve(self, query: str, n_results: int = 4) -> SemanticContext:
        if not self.collection or self.collection.count() == 0:
            return SemanticContext(
                general_guidelines=[
                    "Use exact table and column names from the schema overview.",
                    "Only join tables that are necessary for the requested metrics.",
                    "Match text filters using exact uppercase matching (e.g. ENAME = 'KING')."
                ]
            )

        res = self.collection.query(query_texts=[query], n_results=min(n_results * 2, self.collection.count()))
        candidates = []
        if res and res.get("documents") and res["documents"][0]:
            for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
                q_words = set(re.findall(r"\w+", query.lower()))
                doc_words = set(re.findall(r"\w+", doc.lower()))
                overlap = len(q_words & doc_words) / max(len(q_words), 1)
                candidates.append((overlap, doc, meta))

        candidates.sort(key=lambda x: x[0], reverse=True)
        top_k = candidates[:n_results]

        schema_docs: List[str] = []
        guidelines: List[str] = []

        for _, doc, meta in top_k:
            if not meta:
                continue
            if meta.get("type") == "ddl_schema":
                schema_docs.append(doc)
            else:
                guidelines.append(doc)

        return SemanticContext(schema_docs=schema_docs, general_guidelines=guidelines)

rag_store = SemanticRAGStore()


# =====================================================================
# 4. STRUCTURED SCHEMAS
# =====================================================================
QueryType = Literal[
    "direct_lookup",
    "analytical",
    "comparison",
    "diagnostic",
    "clarification_needed",
    "unsupported"
]

RequestedOutput = Literal[
    "answer",
    "visualization",
    "insights",
    "sql",
    "results",
    "diagnosis"
]

class IntentAnalysis(BaseModel):
    query_type: QueryType = Field(
        description="Classification: direct_lookup, analytical, comparison, diagnostic, clarification_needed, unsupported"
    )
    confidence_score: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Confidence score (0.0 to 1.0). If below 0.70, triggers clarification."
    )
    needs_clarification: bool = Field(
        default=False, description="True if query is vague or confidence < 0.70"
    )
    clarification_question: Optional[str] = Field(
        default=None, description="Concise question to ask user when clarification is needed."
    )
    user_goal: str = Field(description="Summary of the user's explicit goal")
    requested_outputs: List[RequestedOutput] = Field(
        default_factory=lambda: ["answer", "sql", "results", "diagnosis", "visualization", "insights"],
        description="Outputs requested. If user asks specifically for 'only diagnosis' / 'why this only', requested_outputs=['diagnosis']. If 'give me sql only', requested_outputs=['sql']. If general query or 'details', returns all sections."
    )

intent_parser = PydanticOutputParser(pydantic_object=IntentAnalysis)


class QueryTask(BaseModel):
    task_id: str
    description: str
    depends_on: List[str] = Field(default_factory=list)
    required_output: str

class DecompositionPlan(BaseModel):
    is_multi_step: bool = Field(default=False)
    tasks: List[QueryTask] = Field(default_factory=list)

decomp_parser = PydanticOutputParser(pydantic_object=DecompositionPlan)


class ExecutionPlan(BaseModel):
    run_answer: bool = True
    run_visualization: bool = True
    run_insights: bool = True
    run_diagnostic: bool = True
    run_sql_only: bool = False
    output_mode: Literal["single_requested_output", "full_analysis"] = "full_analysis"


class DiagnosticReport(BaseModel):
    observation: str
    diagnosis: str
    hypotheses_evaluated: List[str] = Field(default_factory=list)
    confidence: float = 1.0


class ChartSpecification(BaseModel):
    visualization_required: bool = True
    chart_type: Literal["bar", "line", "scatter", "pie", "none"] = "bar"
    x: Optional[str] = None
    y: Optional[str] = None
    color: Optional[str] = None
    title: str = "Data Visualization"
    reasoning: Optional[str] = None

chart_parser = PydanticOutputParser(pydantic_object=ChartSpecification)


class NormalizedResult(BaseModel):
    columns: List[str] = Field(default_factory=list)
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0


# =====================================================================
# 5. LANGGRAPH CENTRAL STATE
# =====================================================================
class AgentState(BaseModel):
    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)
    thread_id: str = "default_thread"
    user_query: str = ""

    # Schema & RAG
    schema_overview: str = ""
    semantic_context: SemanticContext = Field(default_factory=SemanticContext)

    # Intent & Decomposition
    intent: Optional[IntentAnalysis] = None
    decomposition: Optional[DecompositionPlan] = None
    execution_plan: Optional[ExecutionPlan] = None

    # SQL Generation & Validation
    generated_sql: Optional[str] = None
    validated_sql: Optional[str] = None
    executed_sqls: List[str] = Field(default_factory=list)
    sql_error: Optional[str] = None

    # Primary Tabular Result (Preserving real column names)
    primary_result: Optional[NormalizedResult] = None
    primary_df: Optional[Any] = None

    # Branch Outputs
    answer: Optional[str] = None
    chart_spec: Optional[ChartSpecification] = None
    terminal_chart: Optional[str] = None
    plotly_figure_json: Optional[str] = None
    plotly_html_path: Optional[str] = None
    insights: List[str] = Field(default_factory=list)
    diagnostic_report: Optional[DiagnosticReport] = None

    # Final Response
    final_response: Optional[str] = None
    security_blocked: bool = False
    security_message: Optional[str] = None


# =====================================================================
# 6. SQL UTILITIES, SAFETY & SAFE MARKDOWN FORMATTER
# =====================================================================
def extract_clean_sql(raw_text: str) -> str:
    """Extract one executable SQL statement without damaging quoted identifiers."""
    text_str = str(raw_text).strip()
    fence_match = re.search(r"```(?:sql)?\s*([\s\S]*?)\s*```", text_str, re.IGNORECASE)
    if fence_match:
        text_str = fence_match.group(1).strip()

    # Remove backticks only when they wrap the entire statement. Never strip a
    # trailing backtick from a MySQL identifier such as `SAL`.
    if text_str.startswith("`") and text_str.endswith("`"):
        text_str = text_str[1:-1].strip()

    start_match = re.search(r"(?is)\b(SELECT|WITH|SHOW|\()\b", text_str)
    if start_match:
        text_str = text_str[start_match.start():].strip()

    if ";" in text_str:
        text_str = text_str.split(";", 1)[0].strip()

    for stop_word in ["\n\nNote:", "\nNote:", "\n\nExplanation:", "\nExplanation:", "\n\nThe issue", "\n---"]:
        if stop_word in text_str:
            text_str = text_str.split(stop_word, 1)[0].strip()

    text_str = re.sub(r"SELECT\s+\(([^)]+)\)", r"SELECT \1", text_str, flags=re.IGNORECASE)
    return text_str.strip()

def is_safe_read_query(sql: str) -> bool:
    normalized = re.sub(r"\s+", " ", sql.strip()).lower()
    if not (normalized.startswith("select") or normalized.startswith("with") or normalized.startswith("show") or normalized.startswith("(")):
        return False
    forbidden = re.compile(
        r"\b(insert|update|delete|drop|alter|create|truncate|replace|attach|detach|pragma|vacuum|grant|revoke)\b",
        re.IGNORECASE,
    )
    return forbidden.search(normalized) is None

def safe_to_markdown(df: pd.DataFrame) -> str:
    """Zero-dependency markdown table formatter (avoids tabulate dependency crash)."""
    if df.empty:
        return ""
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(str(h) for h in headers) + " |",
        "| " + " | ".join([":---"] * len(headers)) + " |"
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) if v is not None else "" for v in row.values) + " |")
    return "\n".join(lines)

def render_terminal_bar(rows: list, cat_col: str, val_col: str, max_width: int = 28) -> str:
    """Draws a clean proportional bar chart directly in the terminal text with real column names."""
    valid_items = []
    for r in rows:
        cat = str(r.get(cat_col, "N/A"))
        try:
            val = float(r.get(val_col, 0))
            valid_items.append((cat, val))
        except (ValueError, TypeError):
            continue
    if not valid_items:
        return ""
    max_val = max(v for _, v in valid_items) or 1.0
    lines = [f"\n📊 **Terminal Bar Chart ({val_col} by {cat_col})**:"]
    lines.append("─" * 58)
    for cat, val in valid_items:
        bar_len = int((val / max_val) * max_width) if max_val > 0 else 0
        bar = "█" * bar_len
        lines.append(f"{cat[:14]:<15} │ {bar:<{max_width}} {val:,.2f}")
    lines.append("─" * 58)
    return "\n".join(lines)


# =====================================================================
# 7. INTENT ANALYZER NODE
# =====================================================================
def intent_analyzer_node(state: AgentState) -> dict:
    try:
        security_gateway.check_request(state.user_query, state.thread_id)
    except SecurityError as exc:
        return {"security_blocked": True, "security_message": str(exc)}

    llm = get_llm(temperature=0)
    safe_query = security_safe_for_llm(state.user_query)
    db = db_registry.get_db(state.thread_id)
    available_tables = db.get_usable_table_names()
    dialect = db.dialect
    clean_schema = db_registry.get_clean_schema_overview(state.thread_id)

    prev_context = ""
    for msg in state.messages[:-1][-4:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        prev_context += f"{role}: {security_safe_for_llm(msg.content[:200])}\n"

    prompt = f"""You are the Intent and Request Analyzer for an AI Data Analyst on a {dialect} database.

Conversation History:
{prev_context or 'No prior history.'}

User Question: "{safe_query}"
Available Tables in Database: {available_tables}

Instructions:
1. Contextual Awareness: If the user asks a follow-up or refinement ('show it as graph', 'why?'), interpret it from context.
2. Classify query_type:
   - direct_lookup: Single row or basic query
   - analytical: Aggregations, calculations, comparisons, groups
   - comparison: Comparing entities or groups
   - diagnostic: Inquiring WHY an anomaly or difference occurred ('why', 'reason', 'cause')
   - clarification_needed: Truly ambiguous query with multiple conflicting interpretations
   - unsupported: Request cannot be answered with connected database tables
3. Determine requested_outputs:
   - If user asks specifically for SQL only: ['sql']
   - If user asks specifically for Chart only: ['visualization']
   - If user asks specifically for Diagnosis only ('why this only', 'only diagnosis', 'why only'): ['diagnosis']
   - For standard analytical queries, causal 'why' queries, or requests for 'details', request all standard branches: ['answer', 'sql', 'results', 'diagnosis', 'visualization', 'insights']
4. Confidence Score: Set between 0.0 and 1.0. If < 0.70, set needs_clarification=true and provide clarification_question.

{intent_parser.get_format_instructions()}
"""
    try:
        response = invoke_llm(llm, [SystemMessage(content=prompt)])
        intent = intent_parser.parse(response.content)
    except Exception:
        q_lower = state.user_query.lower()
        is_diag_only = ("only" in q_lower or "just" in q_lower) and any(w in q_lower for w in ["why", "diagnosis", "cause", "reason"])
        is_sql_only = ("only" in q_lower or "just" in q_lower) and "sql" in q_lower
        is_chart_only = ("only" in q_lower or "just" in q_lower) and any(w in q_lower for w in ["chart", "graph", "plot", "visualization"])
        
        if is_diag_only:
            req = ["diagnosis"]
        elif is_sql_only:
            req = ["sql"]
        elif is_chart_only:
            req = ["visualization"]
        else:
            req = ["answer", "sql", "results", "diagnosis", "visualization", "insights"]

        intent = IntentAnalysis(
            query_type="diagnostic" if any(w in q_lower for w in ["why", "cause", "reason", "diff"]) else "analytical",
            confidence_score=0.95,
            needs_clarification=False,
            user_goal=state.user_query,
            requested_outputs=req
        )

    if intent.confidence_score < settings.confidence_threshold:
        intent.needs_clarification = True
        if not intent.clarification_question:
            intent.clarification_question = f"Could you please specify which information or metrics you need from tables: {available_tables}?"

    return {
        "schema_overview": clean_schema,
        "intent": intent
    }

def route_after_intent(state: AgentState) -> str:
    if state.security_blocked:
        return "security_blocked_node"
    if not state.intent or state.intent.needs_clarification or state.intent.query_type == "clarification_needed":
        return "clarification_node"
    if state.intent.query_type == "unsupported":
        return "unsupported_node"
    return "decomposition_node"

def clarification_node(state: AgentState) -> dict:
    q = state.intent.clarification_question if state.intent else "Could you please clarify your request?"
    return {"final_response": f"❓ **Clarification Needed**:\n{q}"}

def unsupported_node(state: AgentState) -> dict:
    return {"final_response": "⚠️ **Unsupported Request**: The requested question cannot be answered with the tables and columns available in the connected database catalog."}


def security_blocked_node(state: AgentState) -> dict:
    return {"final_response": state.security_message or "Security policy blocked this request."}


# =====================================================================
# 8. DECOMPOSITION & RAG RETRIEVAL
# =====================================================================
def decomposition_node(state: AgentState) -> dict:
    llm = get_llm(temperature=0)
    safe_query = security_safe_for_llm(state.user_query)
    rag_store.sync_database_knowledge(state.thread_id)
    semantic_context = rag_store.retrieve(state.user_query)

    prompt = f"""You are the Request Decomposition Planner.
User Goal: {security_safe_for_llm(state.intent.user_goal if state.intent else safe_query)}
Live Database Schema:
{state.schema_overview}

Decompose complex multi-step questions into logical tasks.

{decomp_parser.get_format_instructions()}
"""
    try:
        resp = invoke_llm(llm, [SystemMessage(content=prompt)])
        decomp = decomp_parser.parse(resp.content)
    except Exception:
        decomp = DecompositionPlan(is_multi_step=False, tasks=[
            QueryTask(task_id="T1", description=state.user_query, required_output="Direct answer data")
        ])

    return {
        "semantic_context": semantic_context,
        "decomposition": decomp
    }


# =====================================================================
# 9. SQL AGENT USING `create_sql_agent` (Fast ReAct Execution)
# =====================================================================
def sql_agent_node(state: AgentState) -> dict:
    """
    Executes a multi-step ReAct agent using `create_sql_agent` with SQLDatabaseToolkit.
    Captures live SQL executions and database observations directly from intermediate_steps.
    """
    llm = get_llm(temperature=0)
    safe_query = security_safe_for_llm(state.user_query)
    db = db_registry.get_db(state.thread_id)
    secure_db = ReadOnlyDatabaseProxy(
        db,
        lambda query: security_gateway.validate_sql(query, state.thread_id),
        timeout_seconds=security_gateway.config.sql_timeout_seconds,
    )
    toolkit = SQLDatabaseToolkit(db=secure_db, llm=llm)
    dialect = db.dialect

    guidelines = "\n".join([f"- {g}" for g in state.semantic_context.general_guidelines])

    custom_prefix = f"""You are an expert SQL Data Analyst working with a live {dialect} database.
You have access to tools: `sql_db_schema`, `sql_db_query`, `sql_db_list_tables`, `sql_db_query_checker`.

AVAILABLE DATABASE CATALOG:
{state.schema_overview}

EXECUTION PRINCIPLES:
{guidelines}
- Use ONLY exact table and column names shown in the catalog above. Never autocorrect table names.
- In `Action Input`, never wrap SQL strings in markdown backticks. Write raw SQL.
- For hierarchical lookups, perform a self-join.
- Ensure all non-aggregated columns in SELECT are in GROUP BY.
"""

    sql_agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        prefix=custom_prefix,
        verbose=False,
        max_iterations=settings.agent_max_iterations,
        max_execution_time=settings.agent_timeout_seconds,
        early_stopping_method="generate",
        agent_executor_kwargs={
            "return_intermediate_steps": True,
            "handle_parsing_errors": True,
        },
    )

    clean_sql = None
    executed_sqls = []
    col_names = []
    rows = []
    agent_output = ""
    last_error = None

    try:
        result = sql_agent.invoke({"input": safe_query})
        agent_output = result.get("output", "")
        steps = result.get("intermediate_steps", [])

        # Extract executed SQL and observations from tool actions
        for action, observation in steps:
            if hasattr(action, "tool") and action.tool in ["sql_db_query", "sql_db_query_checker"]:
                tool_input = action.tool_input
                q_str = tool_input.get("query", "") if isinstance(tool_input, dict) else str(tool_input)
                extracted = extract_clean_sql(q_str)
                if extracted:
                    clean_sql = extracted
                    executed_sqls.append(clean_sql)
    except Exception as exc:
        if isinstance(exc, SecurityError):
            return {
                "generated_sql": None,
                "validated_sql": None,
                "executed_sqls": executed_sqls,
                "sql_error": str(exc),
                "primary_result": NormalizedResult(),
                "primary_df": pd.DataFrame(),
            }
        _activate_ollama_fallback(exc)
        last_error = str(exc)

    # Fallback to direct schema execution if agent didn't run a tool query
    if not clean_sql:
        fallback_llm = get_llm(temperature=0, use_fallback=True)
        direct_prompt = f"""You are an SQL Developer for {dialect}.
Question: "{safe_query}"
Catalog:
{state.schema_overview}
        Write ONLY one read-only SQL query with exact column names. Output raw SQL only."""
        try:
            raw_sql = invoke_llm(fallback_llm, [SystemMessage(content=direct_prompt)]).content.strip()
            clean_sql = extract_clean_sql(raw_sql)
        except Exception:
            llm_fb = get_llm(temperature=0, use_fallback=True)
            raw_sql = invoke_llm(llm_fb, [SystemMessage(content=direct_prompt)]).content.strip()
            clean_sql = extract_clean_sql(raw_sql)

    # Execute and extract true column names from database connection
    for attempt in range(settings.max_sql_repairs):
        clean_sql = extract_clean_sql(clean_sql)
        if not is_safe_read_query(clean_sql):
            clean_sql = f"SELECT * FROM ({clean_sql}) AS safe_subquery"
        try:
            executed_sqls.append(clean_sql)
            col_names, rows = db_registry.run_query_with_columns(state.thread_id, clean_sql)
            last_error = None
            break
        except SecurityError as e:
            last_error = str(e)
            break
        except Exception as e:
            last_error = str(e)
            repair_prompt = f"""Fix this {dialect} SQL error:
Question: "{safe_query}"
Failed SQL: {clean_sql}
Error: {last_error}
Schema:
{state.schema_overview}
Output ONLY the corrected raw SQL query with exact table and column names."""
            rep_resp = invoke_llm(llm, [SystemMessage(content=repair_prompt)]).content.strip()
            clean_sql = extract_clean_sql(rep_resp)

    norm_res = NormalizedResult(columns=col_names, rows=rows, row_count=len(rows)) if not last_error else NormalizedResult()
    df = pd.DataFrame(rows) if rows else pd.DataFrame()

    return {
        "generated_sql": clean_sql,
        "validated_sql": clean_sql if not last_error else None,
        "executed_sqls": executed_sqls,
        "sql_error": last_error,
        "primary_result": norm_res,
        "primary_df": df,
        "answer": agent_output if agent_output and not last_error else None
    }


# =====================================================================
# 10. EXECUTION PLANNER
# =====================================================================
def execution_planner_node(state: AgentState) -> dict:
    intent = state.intent
    requested = intent.requested_outputs if intent else ["answer", "sql", "results", "diagnosis", "visualization", "insights"]

    is_single_output = len(requested) == 1 and requested[0] in ["sql", "visualization", "diagnosis", "results"]
    run_sql_only = "sql" in requested and len(requested) == 1
    run_diag = "diagnosis" in requested or intent.query_type == "diagnostic"
    run_vis = ("visualization" in requested) and state.primary_result and state.primary_result.row_count > 0 and len(state.primary_result.columns) >= 2
    run_ins = "insights" in requested and not is_single_output and state.primary_result and state.primary_result.row_count > 0
    run_ans = "answer" in requested or not is_single_output

    plan = ExecutionPlan(
        run_answer=run_ans,
        run_visualization=bool(run_vis),
        run_insights=bool(run_ins),
        run_diagnostic=bool(run_diag),
        run_sql_only=run_sql_only,
        output_mode="single_requested_output" if is_single_output else "full_analysis"
    )

    return {"execution_plan": plan}


# =====================================================================
# 11. PARALLEL BRANCH AGENTS: ANSWER, DIAGNOSTIC (create_sql_agent), VISUALIZATION, INSIGHTS
# =====================================================================
def answer_agent_node(state: AgentState) -> dict:
    if not state.execution_plan or not state.execution_plan.run_answer:
        return {"answer": None}

    if state.sql_error:
        return {"answer": f"I encountered an error executing the query: `{state.sql_error}`."}

    if not state.primary_result or state.primary_result.row_count == 0:
        return {"answer": "No matching records were found in the database for your query."}

    if state.answer and "Agent stopped" not in state.answer:
        return {"answer": state.answer}

    llm = get_llm(temperature=0)
    safe_query = security_safe_for_llm(state.user_query)
    safe_rows = security_safe_for_llm(state.primary_result.rows)
    prompt = f"""You are the Answer Agent.
Answer the user's question clearly, directly, and factually using ONLY the database results below.
Never invent numbers. Use bullet points when appropriate.

User Question: {safe_query}
Executed SQL: {security_safe_for_llm(state.validated_sql)}
Returned Data (Columns: {state.primary_result.columns}):
{safe_rows}
"""
    ans = invoke_llm(llm, [SystemMessage(content=prompt)]).content.strip()
    return {"answer": ans}


def diagnostic_agent_node(state: AgentState) -> dict:
    """
    Diagnostic Agent: Uses `create_sql_agent` to test root-cause hypotheses against live database.
    """
    if not state.execution_plan or not state.execution_plan.run_diagnostic:
        return {"diagnostic_report": None}

    if not state.primary_result or state.primary_result.row_count == 0:
        return {"diagnostic_report": None}

    llm = get_llm(temperature=0)
    safe_query = security_safe_for_llm(state.user_query)
    safe_rows = security_safe_for_llm(state.primary_result.rows)
    db = db_registry.get_db(state.thread_id)
    secure_db = ReadOnlyDatabaseProxy(
        db,
        lambda query: security_gateway.validate_sql(query, state.thread_id, diagnostic=True),
        timeout_seconds=security_gateway.config.diagnostic_timeout_seconds,
    )
    toolkit = SQLDatabaseToolkit(db=secure_db, llm=llm)
    dialect = db.dialect

    diag_prefix = f"""You are an expert Root-Cause Diagnostic Agent for a {dialect} database.
Given an observation and user question, analyze WHY this result or discrepancy occurred.
Run investigative SQL queries against the database using `sql_db_query` to verify potential drivers (e.g., headcount, job mix, outlier compensation).

DATABASE CATALOG:
{state.schema_overview}

CRITICAL:
- Use exact table and column names from the catalog.
- Explain the root cause supported strictly by the evidence.
"""

    diag_agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        prefix=diag_prefix,
        verbose=False,
        max_iterations=4,
        max_execution_time=20.0,
        early_stopping_method="generate",
        agent_executor_kwargs={
            "return_intermediate_steps": True,
            "handle_parsing_errors": True,
        },
    )

    diag_query = f"""Analyze WHY this occurred:
Question: "{safe_query}"
Observation Data: {safe_rows[:5]}
Synthesize a 2-3 sentence evidence-backed diagnosis of the primary drivers."""

    try:
        res = diag_agent.invoke({"input": diag_query})
        output = res.get("output", "")
        steps = res.get("intermediate_steps", [])
        queries_run = [str(act.tool_input) for act, _ in steps if hasattr(act, "tool") and act.tool == "sql_db_query"]
        
        return {
            "diagnostic_report": DiagnosticReport(
                observation=f"Query returned {state.primary_result.row_count} rows.",
                diagnosis=output,
                hypotheses_evaluated=queries_run
            )
        }
    except SecurityError as exc:
        return {
            "diagnostic_report": DiagnosticReport(
                observation="Diagnostic execution was blocked by security policy.",
                diagnosis=str(exc),
                hypotheses_evaluated=[],
            )
        }
    except Exception as exc:
        _activate_ollama_fallback(exc)
        # Fast 1-shot fallback synthesis if diagnostic agent hit a timeout
        fallback_prompt = f"""Explain the primary root-causes or business drivers for this data in 2 concise sentences:
Question: {safe_query}
Data: {safe_rows[:5]}"""
        fallback_llm = get_llm(temperature=0, use_fallback=True)
        fallback_ans = invoke_llm(fallback_llm, [SystemMessage(content=fallback_prompt)]).content.strip()
        return {
            "diagnostic_report": DiagnosticReport(
                observation="Completed data observation.",
                diagnosis=fallback_ans,
                hypotheses_evaluated=[]
            )
        }


def visualization_agent_node(state: AgentState) -> dict:
    if not state.execution_plan or not state.execution_plan.run_visualization:
        return {"plotly_figure_json": None, "plotly_html_path": None, "terminal_chart": None}

    if not state.primary_result or state.primary_result.row_count == 0:
        return {"plotly_figure_json": None, "plotly_html_path": None, "terminal_chart": None}

    safe_rows = security_safe_for_llm(state.primary_result.rows)
    df = pd.DataFrame(safe_rows)
    if df.empty or len(df.columns) < 2 or len(df) <= 1:
        return {"plotly_figure_json": None, "plotly_html_path": None, "terminal_chart": None}

    llm = get_llm(temperature=0)
    prompt = f"""You are the Visualization Agent.
Select the optimal chart configuration for this tabular data.

User Question: {security_safe_for_llm(state.user_query)}
Columns: {list(df.columns)}
Data Sample: {df.head(3).to_dict(orient='records')}

Rules:
- Categorical comparisons -> bar
- Time trends -> line
- Numeric correlations -> scatter
- Proportions -> pie

{chart_parser.get_format_instructions()}
"""
    try:
        resp = invoke_llm(llm, [SystemMessage(content=prompt)])
        spec = chart_parser.parse(resp.content)
        x, y, c_type, title = spec.x, spec.y, spec.chart_type, spec.title

        if not x or x not in df.columns:
            x = df.columns[0]
        if not y or y not in df.columns:
            y = df.columns[1] if len(df.columns) > 1 else None

        if y and y in df.columns:
            df[y] = pd.to_numeric(df[y], errors='ignore')

        if c_type == "line" and y:
            fig = px.line(df, x=x, y=y, title=title)
        elif c_type == "pie" and y:
            fig = px.pie(df, names=x, values=y, title=title)
        elif c_type == "scatter" and y:
            fig = px.scatter(df, x=x, y=y, title=title)
        elif y:
            fig = px.bar(df, x=x, y=y, title=title)
        else:
            return {"plotly_figure_json": None, "plotly_html_path": None, "terminal_chart": None}

        fig.update_layout(template="plotly_white", margin=dict(l=40, r=40, t=50, b=40))
        html_filename = "chart.html"
        fig.write_html(html_filename, include_plotlyjs="cdn")

        if settings.auto_open_browser:
            webbrowser.open(os.path.abspath(html_filename))

        # Render in-terminal bar chart with actual column names
        t_chart = render_terminal_bar(safe_rows, str(x), str(y))

        return {
            "chart_spec": spec,
            "terminal_chart": t_chart,
            "plotly_figure_json": fig.to_json(),
            "plotly_html_path": html_filename
        }
    except Exception:
        return {"plotly_figure_json": None, "plotly_html_path": None, "terminal_chart": None}


def insights_agent_node(state: AgentState) -> dict:
    if not state.execution_plan or not state.execution_plan.run_insights:
        return {"insights": []}

    if not state.primary_result or state.primary_result.row_count <= 1:
        return {"insights": []}

    llm = get_llm(temperature=0)
    safe_rows = security_safe_for_llm(state.primary_result.rows)
    prompt = f"""You are the Insights Analyst Agent.
Extract 2-4 high-value, quantitative findings directly calculated from this dataset.
Do not hallucinate facts. Derive key patterns, spreads, maximums, minimums, or ratios.

Question: {security_safe_for_llm(state.user_query)}
Data: {safe_rows}

Format: Return a bulleted list of 2-4 concise bullet points starting with `- `.
"""
    resp = invoke_llm(llm, [SystemMessage(content=prompt)]).content.strip()
    bullet_lines = [line.strip("- *").strip() for line in resp.splitlines() if line.strip().startswith(("-", "*", "•"))]
    return {"insights": bullet_lines or [resp]}


# =====================================================================
# 12. FINAL RESPONSE COMPOSER (All Standard Sections / Granular Modes)
# =====================================================================
def final_composer_node(state: AgentState) -> dict:
    plan = state.execution_plan
    intent = state.intent
    requested = intent.requested_outputs if intent else ["answer", "sql", "results", "diagnosis", "visualization", "insights"]

    # Explicit Single Output Handling
    if plan and plan.output_mode == "single_requested_output":
        if "sql" in requested and state.validated_sql:
            response = f"```sql\n{state.validated_sql};\n```"
            return {"final_response": security_gateway.mask_for_output(response)}
        if "visualization" in requested:
            res_str = ""
            if state.terminal_chart:
                res_str += f"{state.terminal_chart}\n\n"
            res_str += "### 📊 Visualization\n*(Interactive chart saved to `chart.html`)*"
            return {"final_response": security_gateway.mask_for_output(res_str)}
        if "diagnosis" in requested and state.diagnostic_report:
            diag = state.diagnostic_report.diagnosis or "Diagnostic analysis completed."
            return {"final_response": security_gateway.mask_for_output(f"### 🔬 Diagnosis\n{diag}")}
        if "results" in requested and state.primary_result:
            df = pd.DataFrame(state.primary_result.rows)
            return {"final_response": security_gateway.mask_for_output(f"### 📋 Results\n{safe_to_markdown(df)}")}

    # Default Complete Multi-Section Response: [Answer, SQL, Results, Terminal Chart, Diagnostic, Insights]
    sections = []

    if state.answer:
        sections.append(f"### 💡 Answer\n{state.answer}")

    if state.validated_sql:
        sections.append(f"### 🔍 SQL Query\n```sql\n{state.validated_sql};\n```")

    if state.primary_result and state.primary_result.row_count > 0:
        df = pd.DataFrame(state.primary_result.rows)
        if len(df) <= 12:
            sections.append(f"### 📋 Results\n{safe_to_markdown(df)}")

    if state.terminal_chart:
        sections.append(state.terminal_chart)

    if state.diagnostic_report and state.diagnostic_report.diagnosis:
        sections.append(f"### 🔬 Diagnostic Analysis\n{state.diagnostic_report.diagnosis}")

    if state.plotly_html_path:
        sections.append(f"### 📊 Interactive Chart\n*(Saved to `{state.plotly_html_path}`)*")

    if state.insights:
        bullets = "\n".join([f"- {ins}" for ins in state.insights])
        sections.append(f"### 📈 Key Insights\n{bullets}")

    if state.sql_error:
        sections.append(f"### ⚠️ SQL Error\n`{state.sql_error}`")

    response = "\n\n---\n\n".join(sections) if sections else "No response generated."
    return {"final_response": security_gateway.mask_for_output(response)}


# =====================================================================
# 13. LANGGRAPH WORKFLOW ASSEMBLY
# =====================================================================
def build_workflow():
    workflow = StateGraph(AgentState)

    workflow.add_node("intent_analyzer", intent_analyzer_node)
    workflow.add_node("clarification_node", clarification_node)
    workflow.add_node("unsupported_node", unsupported_node)
    workflow.add_node("security_blocked_node", security_blocked_node)
    workflow.add_node("decomposition_node", decomposition_node)
    workflow.add_node("sql_agent", sql_agent_node)
    workflow.add_node("execution_planner", execution_planner_node)

    workflow.add_node("answer_agent", answer_agent_node)
    workflow.add_node("diagnostic_agent", diagnostic_agent_node)
    workflow.add_node("visualization_agent", visualization_agent_node)
    workflow.add_node("insights_agent", insights_agent_node)
    workflow.add_node("final_composer", final_composer_node)

    workflow.add_edge(START, "intent_analyzer")
    workflow.add_conditional_edges("intent_analyzer", route_after_intent, {
        "clarification_node": "clarification_node",
        "unsupported_node": "unsupported_node",
        "security_blocked_node": "security_blocked_node",
        "decomposition_node": "decomposition_node"
    })

    workflow.add_edge("clarification_node", END)
    workflow.add_edge("unsupported_node", END)
    workflow.add_edge("security_blocked_node", END)

    workflow.add_edge("decomposition_node", "sql_agent")
    workflow.add_edge("sql_agent", "execution_planner")

    # Fan out to parallel branches
    workflow.add_edge("execution_planner", "answer_agent")
    workflow.add_edge("execution_planner", "diagnostic_agent")
    workflow.add_edge("execution_planner", "visualization_agent")
    workflow.add_edge("execution_planner", "insights_agent")

    # Fan in to final response composer
    workflow.add_edge("answer_agent", "final_composer")
    workflow.add_edge("diagnostic_agent", "final_composer")
    workflow.add_edge("visualization_agent", "final_composer")
    workflow.add_edge("insights_agent", "final_composer")

    workflow.add_edge("final_composer", END)

    return workflow.compile(checkpointer=memory_manager.working.get_checkpointer())


# =====================================================================
# 14. INTERACTIVE CLI RUNNER
# =====================================================================
def main() -> None:
    app = build_workflow()
    thread_id = "thread_01"
    default_uri = db_registry.get_uri(thread_id)

    print("=" * 75)
    print("Insight AI — Agentic Text-to-SQL + Diagnostics + Visuals")
    print(f"LLM Provider : {settings.llm_provider.upper()} ({settings.openrouter_model if settings.llm_provider == 'openrouter' else settings.ollama_model})")
    print(f"Connected DB : {default_uri}")
    print("=" * 75)

    while True:
        try:
            query = input(f"\n[{thread_id}] User > ").strip()
            if not query:
                continue
            if query.lower() in {"exit", "quit", "q"}:
                print("Exiting...")
                break

            print("\n⚡ Processing with Insight AI Pipeline...\n")

            memory_context = memory_manager.retrieve_context(thread_id, query)
            request_messages = []
            if memory_context:
                request_messages.append(SystemMessage(content=memory_context))
            request_messages.append(HumanMessage(content=query))

            result = security_gateway.run_with_timeout(
                lambda: app.invoke({
                    "thread_id": thread_id,
                    "user_query": query,
                    "messages": request_messages,
                }, config={"configurable": {"thread_id": thread_id}}),
                security_gateway.config.overall_timeout_seconds,
            )

            if not result.get("security_blocked"):
                memory_manager.remember(thread_id, query, result.get("final_response", ""))

            print("\n" + "=" * 75)
            print("FINAL RESPONSE")
            print("=" * 75)
            print(result.get("final_response", "No response generated."))
            print("=" * 75)

        except KeyboardInterrupt:
            break
        except Exception as exc:
            print(f"Error: {exc}")

if __name__ == "__main__":
    main()
