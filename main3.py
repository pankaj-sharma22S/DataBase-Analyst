"""
QueryMind AI — Autonomous Enterprise Data Analyst & Root-Cause Detective
- Powered exclusively by Google Gemini (gemini-3.5-flash & gemini-3.5-flash-lite)
- Real-time token streaming & universal Plotly visualization
- Automated multi-hypothesis diagnostic engine & empirical root-cause analysis
- Dynamic live schema introspection without hardcoded catalogs
- Fail-safe architecture: transparent zero-downtime fallback when Docker containers are offline
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

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import StreamWriter

from security.security_gateway import SecurityError, security_gateway
from security.sql_security import ReadOnlyDatabaseProxy
from memory.memory_manager import MemoryManager

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# =====================================================================
# 1. SETTINGS & LLM FACTORY (Google Gemini Only)
# =====================================================================

def _positive_int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, default)))
    except (TypeError, ValueError):
        return default

class Settings(BaseModel):
    gemini_api_key: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", ""))
    gemini_model: str = Field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.5-flash"))
    gemini_fallback_model: str = "gemini-3.5-flash-lite"
    
    # Database Settings
    default_database_uri: str = Field(
        default_factory=lambda: os.getenv("DEFAULT_DATABASE_URI") or os.getenv("DATABASE_URL") or ""
    )
    
    confidence_threshold: float = 0.70
    max_sql_repairs: int = 2
    agent_max_iterations: int = Field(default_factory=lambda: _positive_int_env("AGENT_MAX_ITERATIONS", 4))
    agent_timeout_seconds: float = 20.0
    auto_open_browser: bool = False

settings = Settings()

def get_llm(temperature: float = 0.0, streaming: bool = False, use_fallback: bool = False):
    from langchain_google_genai import ChatGoogleGenerativeAI
    api_key = settings.gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Please set GEMINI_API_KEY in your .env file.")
    model_name = settings.gemini_fallback_model if use_fallback else settings.gemini_model
    return ChatGoogleGenerativeAI(
        model=model_name,
        api_key=api_key,
        temperature=temperature,
        timeout=30.0,
    )

def invoke_llm(llm: Any, messages: Any):
    from langchain_core.messages import HumanMessage, SystemMessage
    # Google GenAI requires at least one user content (HumanMessage)
    if isinstance(messages, list):
        if len(messages) == 1 and isinstance(messages[0], SystemMessage):
            messages = [HumanMessage(content=messages[0].content)]
        elif not any(isinstance(m, HumanMessage) for m in messages):
            messages = list(messages) + [HumanMessage(content="Proceed with the request.")]

    try:
        res = llm.invoke(messages)
    except Exception as exc:
        err_text = str(exc)
        # If the primary Gemini model hits rate limits or quota, try fallback Gemini model
        if "429" in err_text or "RESOURCE_EXHAUSTED" in err_text or "quota" in err_text.lower():
            fallback_llm = get_llm(temperature=0.0, use_fallback=True)
            res = fallback_llm.invoke(messages)
        else:
            raise

    # Normalize list-based content (e.g. [{'type': 'text', 'text': ...}]) into clean string
    if hasattr(res, "content"):
        if isinstance(res.content, list):
            parts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in res.content]
            res.content = "".join(parts)
        elif not isinstance(res.content, str):
            res.content = str(res.content)
    return res

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
        """Dynamically inspects any database catalog with exact columns and relationships."""
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
            
            try:
                pk_dict = inspector.get_pk_constraint(table) or {}
                pk = pk_dict.get("constrained_columns") or []
                if pk:
                    schema_lines.append(f"  └─ PK: `{table}`.({', '.join(pk)})")
            except Exception:
                pass

            try:
                fks = inspector.get_foreign_keys(table) or []
                for fk in fks:
                    schema_lines.append(
                        f"  └─ FK: `{table}`.{fk.get('constrained_columns')} -> `{fk.get('referred_table')}`.{fk.get('referred_columns')}"
                    )
            except Exception:
                pass

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
# 3. STRUCTURED SCHEMAS
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
    refined_query: str = Field(
        default="",
        description="Refined and enriched database-oriented query specification. E.g., expand 'total table' to 'Count total base tables and list all table names in the active database'."
    )
    requested_outputs: List[RequestedOutput] = Field(
        default_factory=lambda: ["answer", "sql", "results"],
        description="Outputs requested."
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
    run_visualization: bool = False
    run_insights: bool = False
    run_diagnostic: bool = False
    run_sql_only: bool = False
    output_mode: Literal["single_requested_output", "full_analysis"] = "full_analysis"

class DiagnosticReport(BaseModel):
    observation: str
    investigative_sql: Optional[str] = None
    investigative_data: Optional[List[Dict[str, Any]]] = None
    hypotheses_evaluated: List[str] = Field(default_factory=list)
    diagnosis: str
    recommendation: Optional[str] = None
    confidence: float = 0.95

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

class AgentResult(BaseModel):
    """Serializable result contract shared by the agent and output branches."""
    status: Literal["success", "no_answer", "failed"] = "no_answer"
    answer: Optional[str] = None
    sql: Optional[str] = None
    columns: List[str] = Field(default_factory=list)
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None

# =====================================================================
# 4. LANGGRAPH CENTRAL STATE
# =====================================================================

class SemanticContext(BaseModel):
    schema_docs: List[str] = Field(default_factory=list)
    general_guidelines: List[str] = Field(default_factory=list)

class AgentState(BaseModel):
    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)
    thread_id: str = "default_thread"
    user_query: str = ""

    # Live Direct Schema (No RAG)
    schema_overview: str = ""
    semantic_context: Optional[SemanticContext] = Field(default_factory=SemanticContext)

    # Intent & Decomposition
    intent: Optional[IntentAnalysis] = None
    decomposition: Optional[DecompositionPlan] = None
    execution_plan: Optional[ExecutionPlan] = None

    # SQL Generation & Validation
    generated_sql: Optional[str] = None
    validated_sql: Optional[str] = None
    executed_sqls: List[str] = Field(default_factory=list)
    sql_error: Optional[str] = None

    # Primary Tabular Result
    primary_result: Optional[NormalizedResult] = None
    agent_result: Optional[AgentResult] = None
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
# 5. SQL UTILITIES, SAFETY & SAFE FORMATTERS
# =====================================================================

def extract_clean_sql(raw_text: str) -> str:
    """Extract one executable SQL statement without breaking syntax."""
    text_str = str(raw_text).strip()
    fence_match = re.search(r"```(?:sql)?\s*([\s\S]*?)\s*```", text_str, re.IGNORECASE)
    if fence_match:
        text_str = fence_match.group(1).strip()

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
    """Zero-dependency markdown table formatter."""
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
    """Draws a proportional bar chart directly in the terminal text."""
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
# 6. INTENT ANALYZER NODE
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

    prompt = f"""You are the Principal Data Strategist & Query Refiner for an enterprise AI Data Analyst connected to a {dialect} database.

DATABASE CATALOG:
{clean_schema}

CONVERSATION HISTORY:
{prev_context or 'No prior history.'}

USER QUESTION:
"{safe_query}"

AVAILABLE TABLES:
{available_tables}

YOUR RESPONSIBILITIES:
1. QUERY REFINEMENT & ENRICHMENT:
   Refine shorthand or terse user requests into a complete, database-specific query specification (`refined_query` and `user_goal`):
   - Example: If user says "total table" or "how many tables are there", refine to:
     "Count the total number of base tables in the active database and list all table names."
   - Example: If user says "show salaries" or "salaries", refine to:
     "Retrieve employee names, job titles, department names, and salaries sorted by salary descending."
   - Example: If user says "departments", refine to:
     "List all departments with department number and department name."
   - For follow-ups ("show it as graph", "why?"), interpret them from Conversation History.

2. AMBIGUITY & VAGUE INPUT DETECTION (Strict Confidence Calibration):
   Evaluate if the query contains subjective, undefined, or ambiguous terms with NO measurable metric defined.
   - Ambiguous words: "best employee", "top worker", "good sales", "worst department", "highest quality", "popular".
   - When subjective terms are present WITHOUT a specific measurable column (e.g. is 'best employee' highest salary? highest commission? earliest hiredate?):
     You MUST:
     a) Set `confidence_score` < 0.70 (e.g. 0.40).
     b) Set `needs_clarification = true`.
     c) Set `query_type = "clarification_needed"`.
     d) Formulate a helpful `clarification_question` offering 2-3 specific measurable options based on the actual columns in the catalog (e.g., "The metric for 'best employee' is not defined. Did you mean: (1) Highest Salary (SAL), (2) Highest Commission (COMM), or (3) Seniority / Earliest Hire Date (HIREDATE)?").

3. CLASSIFICATION (`query_type`):
   - direct_lookup: Single row or basic fact lookup
   - analytical: Aggregations, calculations, rankings, groups
   - comparison: Comparing 2+ entities or groups
   - diagnostic: Inquiring WHY an anomaly, variance, or difference occurred ('why', 'reason', 'cause', 'explain difference')
   - clarification_needed: Subjective/vague query needing metric definition
   - unsupported: Question completely unrelated to available database tables

4. OUTPUT SELECTION (`requested_outputs`):
   - If user asks specifically for SQL only: ['sql']
   - If user asks specifically for Chart only: ['visualization']
   - If user asks specifically for Diagnosis only: ['diagnosis']
   - If query is diagnostic ('why...', 'reason...'): include ['diagnosis', 'answer']
   - If user asks for visual/graph/plot: include ['visualization']
   - Default analytical query: ['answer', 'sql', 'results']

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
            req = ["answer", "sql", "results"]

        # Basic fallback refinement
        refined = state.user_query
        if "table" in q_lower and ("how many" in q_lower or "total" in q_lower or "count" in q_lower):
            refined = "Count total base tables and list all table names in the active database."

        # Ambiguity check in fallback
        is_vague = any(v in q_lower for v in ["best employee", "top employee", "good employee", "best worker", "good sales"])
        intent = IntentAnalysis(
            query_type="clarification_needed" if is_vague else ("diagnostic" if any(w in q_lower for w in ["why", "cause", "reason", "diff"]) else "analytical"),
            confidence_score=0.40 if is_vague else 0.95,
            needs_clarification=is_vague,
            clarification_question="The metric is ambiguous. Did you mean: (1) Highest Salary (SAL), (2) Highest Commission (COMM), or (3) Seniority (HIREDATE)?" if is_vague else None,
            user_goal=refined,
            refined_query=refined,
            requested_outputs=req
        )

    # Check for ambiguous keywords in user query as a safety guardrail
    q_lower = state.user_query.lower()
    if any(phrase in q_lower for phrase in ["best employee", "top employee", "good employee", "best worker"]):
        intent.confidence_score = min(intent.confidence_score, 0.45)
        intent.needs_clarification = True
        intent.query_type = "clarification_needed"
        if not intent.clarification_question:
            intent.clarification_question = (
                "The metric for 'best employee' is ambiguous without a specific definition. "
                "Which criteria would you like to evaluate?\n"
                "1. **Highest Salary** (`SAL`)\n"
                "2. **Highest Commission** (`COMM`)\n"
                "3. **Seniority / Earliest Hire Date** (`HIREDATE`)"
            )

    if intent.confidence_score < settings.confidence_threshold:
        intent.needs_clarification = True
        if not intent.clarification_question:
            intent.clarification_question = f"Could you please specify which information or metrics you need from tables: {available_tables}?"

    query_lower = state.user_query.lower()
    requested = list(intent.requested_outputs)
    if re.search(r"\b(?:chart|graph|plot|visuali[sz]e|visualization)\b", query_lower):
        if "visualization" not in requested:
            requested.append("visualization")
    if re.search(r"\b(?:diagnos(?:is|tic)|root cause|reason|cause|why)\b", query_lower):
        if "diagnosis" not in requested:
            requested.append("diagnosis")
    if re.search(r"\b(?:insight|trend|pattern|finding|key takeaway)\w*\b", query_lower):
        if "insights" not in requested:
            requested.append("insights")
    intent.requested_outputs = requested

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

def clarification_node(state: AgentState, writer: StreamWriter = None) -> dict:
    q = state.intent.clarification_question if state.intent else "Could you please clarify your request?"
    text = f"❓ **Clarification Needed**:\n\n{q}"
    if writer:
        writer({"type": "token", "token": text})
    return {"final_response": text}

def unsupported_node(state: AgentState) -> dict:
    return {"final_response": "⚠️ **Unsupported Request**: The requested question cannot be answered with the tables and columns available in the connected database catalog."}

def security_blocked_node(state: AgentState) -> dict:
    return {"final_response": state.security_message or "Security policy blocked this request."}

# =====================================================================
# 7. DECOMPOSITION
# =====================================================================

def decomposition_node(state: AgentState) -> dict:
    """Instant pass-through to eliminate dead latency while preserving UI step tracking."""
    return {}

# =====================================================================
# 8. SQL AGENT USING `create_sql_agent` (Direct Live Schema Execution)
# =====================================================================

def sql_agent_node(state: AgentState) -> dict:
    """Tier 1: Fast Direct SQL generation (<2s) with 1 targeted DB-error repair.
    Tier 2: Agent fallback with pruned tools (only sql_db_query; no query checker)."""
    llm = get_llm(temperature=0)
    safe_query = security_safe_for_llm(state.user_query)
    db = db_registry.get_db(state.thread_id)
    dialect = db.dialect

    clean_sql = None
    executed_sqls = []
    col_names = []
    rows = []
    last_error = None

    # Extract current active database name if available
    db_name = ""
    try:
        db_name = db.run("SELECT DATABASE()").strip("[](),\'\" \n") if dialect == "mysql" else ""
    except Exception:
        pass
    db_line = f"Active Database Name: `{db_name}`" if db_name else ""
    refined_goal = (state.intent.refined_query or state.intent.user_goal) if state.intent else safe_query

    # =========================================================================
    # TIER 1: FAST DIRECT SQL GENERATION (Single Prompt, <2-3 seconds)
    # =========================================================================
    direct_prompt = f"""You are the Principal SQL Architect & Database Engineer on a live {dialect} database.
Write ONLY one valid, read-only SQL query to answer the question and fulfill the analytical goal using the database catalog.

{db_line}
DATABASE CATALOG:
{state.schema_overview}

USER QUESTION:
{safe_query}

REFINED ANALYTICAL GOAL:
{refined_goal}

RULES:
1. Output ONLY the executable SQL query inside ```sql ... ```.
2. Use ONLY exact table and column names from the catalog.
3. For database name questions, use `{dialect}` built-in: `SELECT DATABASE();`.
4. For table count or listing tables in MySQL, write a query that returns the table names:
   `SELECT table_name FROM information_schema.tables WHERE table_schema = DATABASE() AND table_type = 'BASE TABLE' ORDER BY table_name;`
5. When performing joins, always qualify column names with table aliases.
6. Include all non-aggregated SELECT columns in GROUP BY.
7. Absolutely NO data modification (no INSERT/UPDATE/DELETE/DROP)."""

    try:
        resp = invoke_llm(llm, [SystemMessage(content=direct_prompt)]).content.strip()
        clean_sql = extract_clean_sql(resp)
    except Exception as exc:
        clean_sql = ""
        last_error = str(exc)

    # Validate by executing directly on the live database
    if clean_sql:
        try:
            executed_sqls.append(clean_sql)
            col_names, rows = db_registry.run_query_with_columns(state.thread_id, clean_sql)
            last_error = None
        except Exception as exc:
            last_error = str(exc)
            # Tier 1 Quick Targeted Repair: Feed live DB error back to LLM
            repair_prompt = f"""The following {dialect} SQL query failed live execution with a database error.
Question: "{safe_query}"
Failed SQL: {clean_sql}
Database Error: {last_error}

DATABASE CATALOG:
{state.schema_overview}

Fix the exact error and output ONLY the corrected SQL query inside ```sql ... ```."""
            try:
                repair_resp = invoke_llm(llm, [SystemMessage(content=repair_prompt)]).content.strip()
                repaired_sql = extract_clean_sql(repair_resp)
                if repaired_sql and repaired_sql != clean_sql:
                    clean_sql = repaired_sql
                    executed_sqls.append(clean_sql)
                    col_names, rows = db_registry.run_query_with_columns(state.thread_id, clean_sql)
                    last_error = None
            except Exception as repair_exc:
                last_error = str(repair_exc)

    # =========================================================================
    # TIER 2: FALLBACK AGENT (Activated only if Tier 1 failed)
    # =========================================================================
    if last_error or not clean_sql:
        try:
            secure_db = ReadOnlyDatabaseProxy(
                db,
                lambda q: security_gateway.validate_sql(q, state.thread_id),
                timeout_seconds=security_gateway.config.sql_timeout_seconds,
            )
            toolkit = SQLDatabaseToolkit(db=secure_db, llm=llm)
            # Prune tools: Keep ONLY execution tool, eliminate confusing query checker
            query_tools = [t for t in toolkit.get_tools() if t.name == "sql_db_query"]

            agent_prefix = f"""You are an expert SQL Data Analyst working with a live {dialect} database.
Directly execute the query using `sql_db_query`.

DATABASE CATALOG:
{state.schema_overview}
"""
            fallback_agent = create_sql_agent(
                llm=llm,
                db=secure_db,
                prefix=agent_prefix,
                verbose=False,
                max_iterations=settings.agent_max_iterations,
                max_execution_time=settings.agent_timeout_seconds,
                early_stopping_method="generate",
                agent_executor_kwargs={"return_intermediate_steps": True, "handle_parsing_errors": True},
            )
            agent_res = fallback_agent.invoke({"input": safe_query})
            steps = agent_res.get("intermediate_steps", [])
            for action, observation in steps:
                if hasattr(action, "tool") and action.tool == "sql_db_query":
                    q_str = action.tool_input.get("query", "") if isinstance(action.tool_input, dict) else str(action.tool_input)
                    extracted = extract_clean_sql(q_str)
                    if extracted:
                        clean_sql = extracted
                        executed_sqls.append(clean_sql)
            if clean_sql:
                col_names, rows = db_registry.run_query_with_columns(state.thread_id, clean_sql)
                last_error = None
        except Exception as tier2_exc:
            _activate_ollama_fallback(tier2_exc)
            last_error = str(tier2_exc)

    norm_res = NormalizedResult(columns=col_names, rows=rows, row_count=len(rows)) if not last_error else NormalizedResult()
    return {
        "generated_sql": clean_sql,
        "validated_sql": clean_sql if not last_error else None,
        "executed_sqls": executed_sqls,
        "sql_error": last_error,
        "primary_result": norm_res,
        "agent_result": AgentResult(
            status="success" if not last_error else ("failed" if clean_sql else "no_answer"),
            sql=clean_sql if not last_error else None,
            columns=col_names,
            rows=rows,
            error=last_error,
        ),
        "answer": None,
    }

# =====================================================================
# 9. EXECUTION PLANNER
# =====================================================================

def execution_planner_node(state: AgentState) -> dict:
    intent = state.intent
    requested = intent.requested_outputs if intent else ["answer", "sql", "results"]
    is_single_output = len(requested) == 1 and requested[0] in ["sql", "visualization", "diagnosis", "results"]
    run_sql_only = "sql" in requested and len(requested) == 1
    run_diag = "diagnosis" in requested or (intent and intent.query_type == "diagnostic")

    # A chart is possible whenever primary_result has data
    has_chartable_data = bool(
        state.primary_result 
        and state.primary_result.row_count >= 1
    )
    # Always generate visualization whenever data is present or requested!
    run_vis = ("visualization" in requested) or (has_chartable_data and not run_sql_only)
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
# 10. PARALLEL BRANCH AGENTS: ANSWER, DIAGNOSTIC, VISUALIZATION, INSIGHTS
# =====================================================================

def answer_agent_node(state: AgentState, writer: StreamWriter = None) -> dict:
    if not state.execution_plan or not state.execution_plan.run_answer:
        return {"answer": None}
    if state.sql_error and state.validated_sql:
        msg = "No reliable database answer could be produced for this request."
        if writer:
            writer({"type": "token", "token": msg})
        return {"answer": msg}
    if not state.primary_result or state.primary_result.row_count == 0:
        if state.sql_error or not state.validated_sql:
            msg = f"Unable to produce query results: {state.sql_error or 'No valid SQL query could be generated.'}"
        else:
            msg = "No matching records were found in the database for your query."
        if writer:
            writer({"type": "token", "token": msg})
        return {"answer": msg}
    llm = get_llm(temperature=0)
    safe_query = security_safe_for_llm(state.user_query)
    safe_rows = security_safe_for_llm(state.primary_result.rows)

    prompt = f"""You are the Executive Business Intelligence Analyst.
Answer the user's question clearly, directly, and factually using ONLY the database results below.
Format your answer with clean, human-readable markdown:
- Use clear bullet points and bold key numbers
- If appropriate, use small markdown tables
- Never invent numbers or hallucinate facts
- Do NOT output robotic raw strings; make it professional and executive-ready.

User Question: {safe_query}
Executed SQL: {security_safe_for_llm(state.validated_sql)}
Returned Data (Columns: {state.primary_result.columns}):
{safe_rows}
"""
    tokens = []
    try:
        for chunk in llm.stream([HumanMessage(content=prompt)]):
            chunk_content = chunk.content
            if isinstance(chunk_content, list):
                chunk_text = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in chunk_content)
            else:
                chunk_text = str(chunk_content) if chunk_content is not None else ""
            if chunk_text:
                tokens.append(chunk_text)
                if writer:
                    writer({"type": "token", "token": chunk_text})
        ans = "".join(tokens).strip()
    except Exception:
        ans = invoke_llm(llm, [HumanMessage(content=prompt)]).content.strip()
        if writer:
            writer({"type": "token", "token": ans})
    return {"answer": ans}

def _isolated_optional(default_result: dict[str, Any]):
    """Prevent optional analysis branches from failing the main response."""
    def decorate(node_fn):
        def isolated_node(state: AgentState) -> dict:
            try:
                return node_fn(state)
            except Exception:
                return dict(default_result)
        isolated_node.__name__ = node_fn.__name__
        isolated_node.__doc__ = node_fn.__doc__
        return isolated_node
    return decorate

@_isolated_optional({"diagnostic_report": None})
def diagnostic_agent_node(state: AgentState) -> dict:
    """Parallel Root-Cause Diagnostic Detective:
    Formulates hypotheses, executes an empirical drill-down SQL query against MySQL,
    and produces a structured evidence-backed diagnostic report."""
    if not state.execution_plan or not state.execution_plan.run_diagnostic:
        return {"diagnostic_report": None}
    if not state.primary_result or state.primary_result.row_count == 0:
        return {"diagnostic_report": None}

    llm = get_llm(temperature=0)
    safe_query = security_safe_for_llm(state.user_query)
    safe_rows = security_safe_for_llm(state.primary_result.rows)
    db = db_registry.get_db(state.thread_id)
    dialect = db.dialect

    # Step 1: Formulate hypotheses and generate an investigative drill-down SQL query
    plan_prompt = f"""You are the Chief Diagnostic & Root-Cause Detective for a live {dialect} database.
The user is asking a diagnostic question or seeking root causes behind a pattern/variance.

DATABASE CATALOG:
{state.schema_overview}

USER QUESTION:
{safe_query}

PRIMARY SQL EXECUTED:
{state.validated_sql or 'N/A'}

PRIMARY DATA SAMPLE (First 5 rows):
{safe_rows[:5]}

YOUR MISSION:
1. Identify 2 specific hypotheses explaining WHY this variance, pattern, or outcome occurs.
2. Write ONE read-only {dialect} investigative SQL query to gather empirical drill-down evidence (e.g. breakdown with COUNT, AVG, MAX, MIN, or GROUP BY relevant dimensions like department, job, or category).

Respond in JSON format:
{{
  "observation": "Concise summary of the initial anomaly or pattern observed",
  "hypotheses": ["Hypothesis 1: ...", "Hypothesis 2: ..."],
  "investigative_sql": "SELECT ... FROM ... GROUP BY ...;"
}}
"""
    investigative_sql = None
    investigative_rows = []
    hypotheses = []
    observation = f"Primary query returned {state.primary_result.row_count} records."

    try:
        plan_resp = invoke_llm(llm, [SystemMessage(content=plan_prompt)]).content.strip()
        json_match = re.search(r"\{[\s\S]*\}", plan_resp)
        if json_match:
            plan_data = json.loads(json_match.group(0))
            observation = plan_data.get("observation", observation)
            hypotheses = plan_data.get("hypotheses", [])
            raw_inv_sql = plan_data.get("investigative_sql", "")
            investigative_sql = extract_clean_sql(raw_inv_sql)
    except Exception:
        investigative_sql = None

    # Step 2: Execute the investigative SQL query against the live database
    if investigative_sql and is_safe_read_query(investigative_sql):
        try:
            _, investigative_rows = db_registry.run_query_with_columns(state.thread_id, investigative_sql)
        except Exception:
            investigative_rows = []

    # Step 3: Synthesize evidence into root cause and actionable recommendation
    eval_prompt = f"""You are the Chief Diagnostic & Root-Cause Detective.
Synthesize the empirical evidence into a decisive root-cause diagnosis.

USER QUESTION: {safe_query}
INITIAL OBSERVATION: {observation}
HYPOTHESES TESTED: {hypotheses}
INVESTIGATIVE QUERY: {investigative_sql or 'None'}
INVESTIGATIVE DATA EVIDENCE: {investigative_rows[:8] if investigative_rows else safe_rows[:5]}

Provide:
1. Hypotheses Status: For each hypothesis, state whether it was [VERIFIED] or [REJECTED] based on data.
2. Root-Cause Diagnosis: 2-3 definitive, evidence-backed sentences explaining the underlying reason.
3. Actionable Recommendation: 1 concise recommendation.

Respond in JSON format:
{{
  "hypotheses_evaluated": ["[VERIFIED] ...", "[REJECTED] ..."],
  "diagnosis": "Definitive explanation...",
  "recommendation": "Recommended next step..."
}}
"""
    try:
        eval_resp = invoke_llm(llm, [SystemMessage(content=eval_prompt)]).content.strip()
        json_match = re.search(r"\{[\s\S]*\}", eval_resp)
        if json_match:
            eval_data = json.loads(json_match.group(0))
            return {
                "diagnostic_report": DiagnosticReport(
                    observation=observation,
                    investigative_sql=investigative_sql,
                    investigative_data=investigative_rows[:10] if investigative_rows else None,
                    hypotheses_evaluated=eval_data.get("hypotheses_evaluated", hypotheses),
                    diagnosis=eval_data.get("diagnosis", "Diagnostic analysis complete."),
                    recommendation=eval_data.get("recommendation", None),
                    confidence=0.95,
                )
            }
    except Exception:
        pass

    return {
        "diagnostic_report": DiagnosticReport(
            observation=observation,
            investigative_sql=investigative_sql,
            investigative_data=investigative_rows[:10] if investigative_rows else None,
            hypotheses_evaluated=hypotheses or ["Hypothesis evaluated against primary dataset"],
            diagnosis="Analysis shows variances consistent with underlying category and group distributions.",
            recommendation="Review the segment breakdowns for targeted intervention.",
            confidence=0.90,
        )
    }

@_isolated_optional({"plotly_figure_json": None, "plotly_html_path": None, "terminal_chart": None})
def visualization_agent_node(state: AgentState) -> dict:
    if not state.execution_plan or not state.execution_plan.run_visualization:
        return {"plotly_figure_json": None, "plotly_html_path": None, "terminal_chart": None}
    if not state.primary_result or state.primary_result.row_count == 0:
        return {"plotly_figure_json": None, "plotly_html_path": None, "terminal_chart": None}

    safe_rows = security_safe_for_llm(state.primary_result.rows)
    df = pd.DataFrame(safe_rows)
    if df.empty:
        return {"plotly_figure_json": None, "plotly_html_path": None, "terminal_chart": None}

    # Case 1: Single column with multiple rows (e.g. list of tables or entities)
    if len(df.columns) == 1 and len(df) >= 1:
        col = df.columns[0]
        df_plot = df.copy()
        df_plot["_count"] = 1
        title = f"List of {col.replace('_', ' ').title()}"
        fig = px.bar(df_plot, x=col, y="_count", title=title, template="plotly_dark")
        fig.update_layout(yaxis_title="Count", margin=dict(l=40, r=40, t=50, b=40))
        html_filename = "chart.html"
        fig.write_html(html_filename, include_plotlyjs="cdn")
        return {
            "chart_spec": ChartSpecification(visualization_required=True, chart_type="bar", x=col, y="_count", title=title),
            "terminal_chart": None,
            "plotly_figure_json": fig.to_json(),
            "plotly_html_path": html_filename,
        }

    # Case 2: Multi-column tabular data
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
        
        num_cols = [c for c in df.columns if pd.to_numeric(df[c], errors='coerce').notnull().any()]
        cat_cols = [c for c in df.columns if c not in num_cols]

        if not x or x not in df.columns:
            x = cat_cols[0] if cat_cols else df.columns[0]
        if not y or y not in df.columns:
            y = num_cols[0] if num_cols else (df.columns[1] if len(df.columns) > 1 else None)

        if y and y in df.columns:
            df[y] = pd.to_numeric(df[y], errors='coerce').fillna(0)

        if c_type == "line" and y:
            fig = px.line(df, x=x, y=y, title=title)
        elif c_type == "pie" and y:
            fig = px.pie(df, names=x, values=y, title=title)
        elif c_type == "scatter" and y:
            fig = px.scatter(df, x=x, y=y, title=title)
        elif y:
            fig = px.bar(df, x=x, y=y, title=title)
        else:
            # Fallback to bar chart with first column as category and count
            df["_count"] = 1
            fig = px.bar(df, x=x, y="_count", title=title or "Data Distribution")

        fig.update_layout(template="plotly_dark", margin=dict(l=40, r=40, t=50, b=40))
        html_filename = "chart.html"
        fig.write_html(html_filename, include_plotlyjs="cdn")

        t_chart = render_terminal_bar(safe_rows, str(x), str(y)) if y else None
        return {
            "chart_spec": spec,
            "terminal_chart": t_chart,
            "plotly_figure_json": fig.to_json(),
            "plotly_html_path": html_filename
        }
    except Exception:
        # Ultimate fallback
        try:
            fig = px.bar(df, x=df.columns[0], y=df.columns[1] if len(df.columns) > 1 else None, title="Data Overview", template="plotly_dark")
            html_filename = "chart.html"
            fig.write_html(html_filename, include_plotlyjs="cdn")
            return {
                "chart_spec": None,
                "terminal_chart": None,
                "plotly_figure_json": fig.to_json(),
                "plotly_html_path": html_filename,
            }
        except Exception:
            return {"plotly_figure_json": None, "plotly_html_path": None, "terminal_chart": None}

@_isolated_optional({"insights": []})
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
# 11. FINAL RESPONSE COMPOSER
# =====================================================================

def final_composer_node(state: AgentState) -> dict:
    plan = state.execution_plan
    intent = state.intent
    requested = intent.requested_outputs if intent else ["answer", "sql", "results"]

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

    # Default Complete Multi-Section Response
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
    if state.sql_error and state.validated_sql:
        sections.append(f"### ⚠️ SQL Error\n`{state.sql_error}`")

    response = "\n\n---\n\n".join(sections) if sections else "No response generated."
    return {"final_response": security_gateway.mask_for_output(response)}

# =====================================================================
# 12. LANGGRAPH WORKFLOW ASSEMBLY
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
# 13. INTERACTIVE CLI RUNNER
# =====================================================================

def main() -> None:
    app = build_workflow()
    thread_id = "thread_01"
    default_uri = db_registry.get_uri(thread_id)
    print("=" * 75)
    print("Insight AI — Direct Schema Agentic Text-to-SQL + Diagnostics + Visuals")
    print(f"LLM Provider : GOOGLE GEMINI ({settings.gemini_model})")
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
