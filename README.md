# 🧠 QueryMind AI
### *Autonomous Enterprise Data Analyst & Root-Cause Detective*

<p align="center">
  <img src="https://img.shields.io/badge/Model-Google%20Gemini%203.5-4285F4?style=for-the-badge&logo=google&logoColor=white" alt="Google Gemini" />
  <img src="https://img.shields.io/badge/Orchestrator-LangGraph-FF6F00?style=for-the-badge&logo=python&logoColor=white" alt="LangGraph" />
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/React-Vite%20UI-61DAFB?style=for-the-badge&logo=react&logoColor=black" alt="React" />
  <img src="https://img.shields.io/badge/Plotly-Universal%20Visuals-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" alt="Plotly" />
  <img src="https://img.shields.io/badge/Security-Fail--Safe%20Gateway-10B981?style=for-the-badge&logo=shield&logoColor=white" alt="Security" />
</p>

---

## 🚀 The Goal & Pitch

Traditional Text-to-SQL solutions stop at generating basic SQL strings. When queries return empty rows, unexplained trends, or subtle discrepancies, data teams are still forced to spend hours diagnosing root causes manually.

**QueryMind AI** redefines autonomous data intelligence:
1. **Understands Natural Intent**: Refines fuzzy goals into precise analytical objectives and requests clarifications whenever metrics are undefined.
2. **Generates & Self-Repairs Real SQL**: Direct Tier-1 generation (<2s) with schema-aware repair against live database engines.
3. **Word-by-Word Real-Time Streaming**: Delivers an executive business summary in real-time markdown token-by-token.
4. **Universal Interactive Plotly Visualizations**: Automatically generates proportional, interactive charts whenever tabular rows exist.
5. **Autonomous Root-Cause Diagnostic Detective**: Formulates empirical hypotheses, writes targeted drill-down SQL queries, tests evidence, and presents an actionable root-cause diagnosis.
6. **Zero-Downtime Docker Resiliency**: Operates seamlessly even when Docker containers (PostgreSQL / Redis) are offline by falling back to local SQLite and thread-safe in-memory sliding-window caches.

---

## 📸 Visual Showcase

### 1. Autonomous LangGraph State Machine
<p align="center">
  <img src="docs/images/architecture.png" alt="QueryMind AI Architecture" width="85%" />
</p>

### 2. Conversational Analysis with Streaming & Structured Tables
<p align="center">
  <img src="docs/images/chat_analysis.png" alt="QueryMind AI Chat Analysis" width="85%" />
</p>

### 3. Universal Interactive Charts & Root-Cause Diagnostics
<p align="center">
  <img src="docs/images/visualization_diagnostic.png" alt="QueryMind AI Visualization & Diagnostics" width="85%" />
</p>

---

## 🧩 Technology Stack & Skills Used

| Domain | Technology / Framework | Role & Capabilities |
| :--- | :--- | :--- |
| **Foundation Models** | `Google Gemini 3.5 Flash` & `Flash-Lite` | Ultra-low latency reasoning, SQL generation, and real-time token streaming. |
| **Agentic Workflow** | `LangGraph` & `LangChain` | Cyclic multi-agent graph with parallel fan-out branches (Answer, Visuals, Diagnostic, Insights). |
| **Database Introspection** | `SQLAlchemy` Live Inspector | Dynamic real-time schema cataloging directly from connection URIs (MySQL, PostgreSQL, SQLite). |
| **Interactive Visuals** | `Plotly Express` & `react-plotly.js` | Autoscaled, responsive charts (bar, line, scatter, pie) with PNG/SVG export. |
| **Frontend UI** | `React 18`, `TypeScript`, `Vite`, `marked` | ChatGPT-style unboxed layout, multi-theme selector (Dark, Midnight, Light), auto-scrolling. |
| **Backend API** | `FastAPI`, `Uvicorn`, `Server-Sent Events (SSE)` | Async token and workflow event streaming directly to the browser. |
| **Security Gateway** | Fail-Closed Gateway & PII Redactor | Sliding-window rate limiter, regex PII masking, AST SQL validation, and prompt injection defense. |
| **High-Performance Cache**| `Redis` (with In-Memory Fallback) | Sub-millisecond rate limiting and semantic answer caching with zero crash guarantee. |

---

## ⚡ Zero-Downtime Docker Fail-Safe Architecture

QueryMind AI is engineered to be **indestructible** against infrastructure failures:
- **Redis Offline?**: Automatically fails over to a thread-safe sliding-window rate limiter and in-memory TTL dictionary.
- **PostgreSQL Offline?**: Automatically falls back to local SQLite working memory for thread persistence and context retrieval.
- **The Application Never Crashes**: All features, analytical flows, and SQL pipelines continue uninterrupted.

---

## 🛠️ Quickstart & Setup Guide

### 1. Prerequisites
- Python 3.11+
- Node.js 18+ & npm
- MySQL / PostgreSQL / SQLite database

### 2. Configure Environment (`.env`)
Create or verify your `.env` file in the project root:
```env
# Primary LLM Provider Configuration
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash
GEMINI_FALLBACK_MODEL=gemini-3.5-flash-lite

# Database Connection (Default fallback)
DEFAULT_DATABASE_URI=mysql+pymysql://root:password@localhost:3306/company

# Optional Redis Cache (Auto-falls back to memory if container is offline)
REDIS_HOST=localhost
REDIS_PORT=6379
```

### 3. Backend Setup
```bash
# Activate virtual environment
.\venv\Scripts\activate   # Windows
source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirement.txt

# Start the FastAPI server
uvicorn api:app --host 0.0.0.0 --port 8000
```

### 4. Frontend Setup
```bash
cd frontend

# Install packages
npm install

# Start Vite dev server
npm run dev
```

Visit the application at: **`http://localhost:5173`**

---

## 🐳 Docker Commands (Optional Infrastructure)

### Redis (Semantic Caching & Rate Limiting)
```bash
# Build and run Redis in the background
docker compose -f docker-compose.redis.yml up -d --build

# Stop Redis container
docker compose -f docker-compose.redis.yml down
```

### PostgreSQL (Persistent Long-Term Memory)
```bash
# Build and start PostgreSQL container
docker compose up -d

# Stop PostgreSQL container
docker compose down

# If you ever encounter a stale network error:
docker rm -f text_to_sql
docker compose up -d
```

---

## 💡 Example Analytical Queries to Try

Once connected to your database in the UI, test these prompts:

1. **Schema Discovery**:
   - `"Show all tables in this database with their record counts"`
2. **Aggregations & Rankings**:
   - `"Who are the highest paid employees and what departments are they in?"`
   - `"What is the total salary expenditure by department?"`
3. **Comparisons & Proportions**:
   - `"Compare workforce headcount across departments as a chart"`
4. **Autonomous Root-Cause Diagnostics**:
   - `"Why does the research department have a high average salary compared to sales?"`
   - `"Explain why certain employees have no commission recorded"`
5. **Ambiguity Clarification Trigger**:
   - `"Who is the best employee?"` *(QueryMind AI will clarify whether you mean salary, commission, or tenure)*

---

## 🔒 Security & Governance Boundary
- **Read-Only Enforcement**: Blocks `DROP`, `ALTER`, `TRUNCATE`, `INSERT`, `UPDATE`, and `DELETE`.
- **AST SQL Sanitization**: Strips multi-statement injections and disallows database catalog manipulation.
- **Data Privacy**: Automatic regex redaction of credit cards, passwords, API tokens, and credentials.
- **Fail-Safe Rate Limiter**: Configurable request quotas across IP and thread identifiers.

---

<p align="center">
  <b>QueryMind AI</b> • Built for Data-Driven Teams
</p>
