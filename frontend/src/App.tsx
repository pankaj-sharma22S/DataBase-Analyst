import { useEffect, useRef, useState } from "react";
import Plot from "react-plotly.js";
import { marked } from "marked";

const API = "http://localhost:8000";

type Status = "pending" | "running" | "completed" | "failed";
type WorkflowStep = { id: string; label: string; status: Status };
type Thread = { thread_id: string; name: string; created_at?: string | null };
type Result = { columns: string[]; rows: Record<string, unknown>[]; row_count: number };
type DiagnosticReport = {
  observation?: string;
  investigative_sql?: string | null;
  investigative_data?: Record<string, unknown>[] | null;
  hypotheses_evaluated?: string[];
  diagnosis?: string;
  recommendation?: string | null;
  confidence?: number;
};
type Analysis = {
  user_query?: string;
  schema_overview?: string;
  answer?: string | null;
  generated_sql?: string | null;
  validated_sql?: string | null;
  primary_result?: Result | null;
  plotly_figure?: { data?: unknown[]; layout?: Record<string, unknown> } | null;
  plotly_html_path?: string | null;
  diagnostic_report?: DiagnosticReport | null;
  insights?: string[];
  final_response?: string | null;
  sql_error?: string | null;
};

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  pending?: boolean;
  analysis?: Analysis;
  steps?: WorkflowStep[];
};

type ThemeMode = "dark" | "light" | "midnight";

const isGreeting = (value: string) => /^(hi|hello|hey|good morning|good afternoon|good evening)[!. ,]*$/i.test(value.trim());

const defaultSteps = (): WorkflowStep[] => [
  { id: "intent_analyzer", label: "Intent Analysis", status: "pending" },
  { id: "decomposition_node", label: "Schema Retrieval", status: "pending" },
  { id: "sql_agent", label: "SQL Generation & Execution", status: "pending" },
  { id: "execution_planner", label: "Result Planning", status: "pending" },
  { id: "answer_agent", label: "Answer", status: "pending" },
  { id: "diagnostic_agent", label: "Diagnostic Analysis", status: "pending" },
  { id: "visualization_agent", label: "Visualization", status: "pending" },
  { id: "insights_agent", label: "Insights", status: "pending" },
  { id: "final_composer", label: "Final Analysis", status: "pending" },
];

function MarkdownText({ text }: { text: string }) {
  if (!text) return null;
  const html = marked.parse(text, { gfm: true, breaks: true }) as string;
  return <div className="markdown-prose" dangerouslySetInnerHTML={{ __html: html }} />;
}

export default function App() {
  const [threads, setThreads] = useState<Thread[]>([]);
  const [selected, setSelected] = useState<Thread | null>(null);
  const [query, setQuery] = useState("");
  const [running, setRunning] = useState(false);
  const [schema, setSchema] = useState("");
  const [databaseReady, setDatabaseReady] = useState<Record<string, boolean>>({});
  const [showDatabaseDialog, setShowDatabaseDialog] = useState(false);
  const [databaseUri, setDatabaseUri] = useState("");
  const [connecting, setConnecting] = useState(false);
  const [messagesByThread, setMessagesByThread] = useState<Record<string, ChatMessage[]>>({});
  
  // Multi-theme mode
  const [theme, setTheme] = useState<ThemeMode>(() => {
    return (localStorage.getItem("insight_ai_theme") as ThemeMode) || "dark";
  });

  // Sidebar toggle
  const [sidebarOpen, setSidebarOpen] = useState(true);

  // Auto-scroll ref
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [showScrollBottom, setShowScrollBottom] = useState(false);

  const chatMessages = selected ? messagesByThread[selected.thread_id] || [] : [];

  // Theme application
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("insight_ai_theme", theme);
  }, [theme]);

  // Load initial threads
  const loadThreads = async () => {
    try {
      const response = await fetch(`${API}/api/threads`);
      if (response.ok) {
        const data = await response.json();
        setThreads(data);
        if (data.length > 0 && !selected) {
          void selectThread(data[0]);
        }
      }
    } catch {
      // Backend starting up
    }
  };

  useEffect(() => {
    void loadThreads();
  }, []);

  // Automatic scrolling to bottom when new messages/tokens arrive
  const scrollToBottom = (behavior: ScrollBehavior = "smooth") => {
    messagesEndRef.current?.scrollIntoView({ behavior });
  };

  useEffect(() => {
    scrollToBottom(running ? "auto" : "smooth");
  }, [chatMessages, running]);

  // Handle scroll listener to show scroll-to-bottom pill if scrolled up
  const handleScroll = () => {
    if (!scrollAreaRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = scrollAreaRef.current;
    const isNearBottom = scrollHeight - scrollTop - clientHeight < 150;
    setShowScrollBottom(!isNearBottom);
  };

  const createThread = async () => {
    const response = await fetch(`${API}/api/threads`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    const thread = (await response.json()) as Thread;
    setThreads((old) => [thread, ...old.filter((item) => item.thread_id !== thread.thread_id)]);
    setDatabaseReady((old) => ({ ...old, [thread.thread_id]: false }));
    await selectThread(thread, false);
  };

  const selectThread = async (thread: Thread, knownReady?: boolean) => {
    setSelected(thread);
    const [schemaResponse, historyResponse] = await Promise.all([
      fetch(`${API}/api/schema/${thread.thread_id}`),
      fetch(`${API}/api/threads/${thread.thread_id}/history`),
    ]);
    if (schemaResponse.ok) {
      setSchema((await schemaResponse.json()).schema_overview || "");
      setDatabaseReady((old) => ({ ...old, [thread.thread_id]: knownReady ?? true }));
    } else {
      setDatabaseReady((old) => ({ ...old, [thread.thread_id]: false }));
    }
    if (historyResponse.ok) {
      const records = (await historyResponse.json()) as Analysis[];
      const restored: ChatMessage[] = [];
      records.forEach((record, index) => {
        const text = record.user_query || "";
        restored.push({ id: `${thread.thread_id}-${index}-user`, role: "user", content: text });
        restored.push({
          id: `${thread.thread_id}-${index}-assistant`,
          role: "assistant",
          content: record.answer || record.final_response || "Analysis completed.",
          analysis: record,
          steps: defaultSteps().map((s) => ({ ...s, status: "completed" })),
        });
      });
      setMessagesByThread((old) => ({ ...old, [thread.thread_id]: restored }));
    }
  };

  const submitDatabaseConnection = async () => {
    if (!selected || !databaseUri.trim() || connecting) return;
    setConnecting(true);
    try {
      const response = await fetch(`${API}/api/database/connect`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ thread_id: selected.thread_id, database_uri: databaseUri.trim() }),
      });
      if (!response.ok) {
        let errDetail = "Database connection failed";
        try {
          const errJson = await response.json();
          if (errJson.detail) errDetail = errJson.detail;
        } catch {
          // fallback
        }
        throw new Error(errDetail);
      }
      setDatabaseReady((old) => ({ ...old, [selected.thread_id]: true }));
      setDatabaseUri("");
      setShowDatabaseDialog(false);
      const schemaResponse = await fetch(`${API}/api/schema/${selected.thread_id}`);
      if (schemaResponse.ok) setSchema((await schemaResponse.json()).schema_overview || "");
    } catch (e) {
      alert(e instanceof Error ? e.message : "Connection failed");
    } finally {
      setConnecting(false);
    }
  };

  const runAnalysis = async (userPrompt?: string) => {
    const textToRun = userPrompt || query;
    if (!selected || !textToRun.trim() || running) return;
    const submittedQuery = textToRun.trim();

    if (!databaseReady[selected.thread_id] && !isGreeting(submittedQuery)) {
      setShowDatabaseDialog(true);
      return;
    }

    const userMessage: ChatMessage = {
      id: `${selected.thread_id}-${Date.now()}-user`,
      role: "user",
      content: submittedQuery,
    };
    const assistantMessage: ChatMessage = {
      id: `${selected.thread_id}-${Date.now()}-assistant`,
      role: "assistant",
      content: "",
      pending: true,
      steps: defaultSteps(),
    };

    setMessagesByThread((old) => ({
      ...old,
      [selected.thread_id]: [...(old[selected.thread_id] || []), userMessage, assistantMessage],
    }));
    setQuery("");

    if (isGreeting(submittedQuery)) {
      const greeting = "Hello! How can I help you analyze your database today?";
      setMessagesByThread((old) => ({
        ...old,
        [selected.thread_id]: (old[selected.thread_id] || []).map((msg, idx, list) =>
          idx === list.length - 1 ? { ...msg, content: greeting, pending: false } : msg
        ),
      }));
      return;
    }

    setRunning(true);
    try {
      const response = await fetch(
        `${API}/api/analyze/${selected.thread_id}/stream?query=${encodeURIComponent(submittedQuery)}`
      );
      if (!response.ok || !response.body) throw new Error("Analysis stream unavailable");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const chunk = await reader.read();
        if (chunk.done) break;
        buffer += decoder.decode(chunk.value, { stream: true });
        const packets = buffer.split("\n\n");
        buffer = packets.pop() || "";

        for (const packet of packets) {
          const line = packet.split("\n").find((item) => item.startsWith("data: "));
          if (!line) continue;
          const event = JSON.parse(line.slice(6)) as {
            type: string;
            step?: string;
            status?: Status;
            data?: Analysis;
            message?: string;
            token?: string;
          };

          if (event.type === "token" && event.token) {
            setMessagesByThread((old) => ({
              ...old,
              [selected.thread_id]: (old[selected.thread_id] || []).map((message, index, list) => {
                if (index !== list.length - 1) return message;
                return {
                  ...message,
                  content: message.content + event.token,
                  pending: false,
                };
              }),
            }));
          } else if (event.type === "workflow" && event.step && event.status) {
            const cleanData = event.data
              ? Object.fromEntries(Object.entries(event.data).filter(([_, v]) => v !== null && v !== undefined))
              : {};
            setMessagesByThread((old) => ({
              ...old,
              [selected.thread_id]: (old[selected.thread_id] || []).map((message, index, list) => {
                if (index !== list.length - 1) return message;
                const updatedAnalysis = { ...(message.analysis || {}), ...cleanData };
                return {
                  ...message,
                  steps: (message.steps || defaultSteps()).map((step) =>
                    step.id === event.step ? { ...step, status: event.status! } : step
                  ),
                  analysis: updatedAnalysis,
                };
              }),
            }));
          } else if (event.type === "result" && event.data) {
            const responseText = event.data.answer || event.data.final_response || "Analysis completed.";
            setMessagesByThread((old) => ({
              ...old,
              [selected.thread_id]: (old[selected.thread_id] || []).map((message, index, list) => {
                if (index !== list.length - 1) return message;
                return {
                  ...message,
                  content: message.content.trim() ? message.content : responseText,
                  pending: false,
                  analysis: event.data,
                };
              }),
            }));
          } else if (event.type === "error") {
            const errMsg = event.message || "Analysis failed";
            setMessagesByThread((old) => ({
              ...old,
              [selected.thread_id]: (old[selected.thread_id] || []).map((item, index, list) =>
                index === list.length - 1 ? { ...item, content: errMsg, pending: false } : item
              ),
            }));
          }
        }
      }
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : "Analysis failed";
      setMessagesByThread((old) => ({
        ...old,
        [selected.thread_id]: (old[selected.thread_id] || []).map((item, index, list) =>
          index === list.length - 1 ? { ...item, content: errMsg, pending: false } : item
        ),
      }));
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="app-shell">
      {/* Sidebar - Collapsible with threads & theme controls */}
      <aside className={`sidebar ${sidebarOpen ? "open" : "closed"}`}>
        <div className="brand">
          <div className="brand-left">
            <div className="brand-mark">✧</div>
            <strong>QueryMind AI</strong>
          </div>
        </div>

        <div className="sidebar-actions">
          <button className="new-chat-btn" onClick={createThread}>
            <span>＋</span> New Analysis
          </button>
          <button className="connect-db-btn" onClick={() => setShowDatabaseDialog(true)}>
            <span>▱</span> Connect Database
          </button>
        </div>

        <div className="section-label">Recent Chats</div>
        <div className="thread-list">
          {threads.map((t) => (
            <button
              key={t.thread_id}
              className={`thread-item ${selected?.thread_id === t.thread_id ? "active" : ""}`}
              onClick={() => void selectThread(t)}
            >
              <span>💬 {t.name || t.thread_id}</span>
            </button>
          ))}
        </div>

        <div className="sidebar-footer">
          <div className="theme-selector">
            <button
              className={`theme-pill ${theme === "dark" ? "active" : ""}`}
              onClick={() => setTheme("dark")}
            >
              Dark
            </button>
            <button
              className={`theme-pill ${theme === "midnight" ? "active" : ""}`}
              onClick={() => setTheme("midnight")}
            >
              Midnight
            </button>
            <button
              className={`theme-pill ${theme === "light" ? "active" : ""}`}
              onClick={() => setTheme("light")}
            >
              Light
            </button>
          </div>

          <div className="db-status-bar">
            <span
              className={`status-dot ${
                selected && databaseReady[selected.thread_id] ? "" : "disconnected"
              }`}
            />
            <span>
              {selected && databaseReady[selected.thread_id]
                ? "Database Live"
                : "No Database"}
            </span>
          </div>
        </div>
      </aside>

      {/* Main ChatGPT-Style Conversational Canvas */}
      <main className="main-chat-container">
        {/* Top Minimalist Header */}
        <header className="chat-top-bar">
          <button
            className="sidebar-toggle-btn"
            onClick={() => setSidebarOpen((prev) => !prev)}
            title="Toggle Sidebar"
          >
            ☰ {sidebarOpen ? "Close sidebar" : "Open sidebar"}
          </button>
          <div className="current-thread-title">
            {selected?.name || "Select or start a chat"}
          </div>
          <div style={{ width: 80 }} />
        </header>

        {/* Scrollable Chat Viewport */}
        <div className="messages-scroll-area" ref={scrollAreaRef} onScroll={handleScroll}>
          {chatMessages.length === 0 ? (
            <div className="empty-chat-screen">
              <div className="empty-logo-circle">✧</div>
              <h2>What would you like to analyze?</h2>
              <p>
                Ask questions about your database in natural language. Instant queries,
                interactive charts, and root-cause diagnostics will generate sequentially.
              </p>
              <div className="quick-prompts-grid">
                <button
                  className="quick-prompt-card"
                  onClick={() => void runAnalysis("Show all tables in this database")}
                >
                  <strong>📊 List database tables</strong>
                  <div>View all cataloged tables and row counts</div>
                </button>
                <button
                  className="quick-prompt-card"
                  onClick={() => void runAnalysis("Show total employees by department")}
                >
                  <strong>👥 Employee distribution</strong>
                  <div>Compare workforce headcount across departments</div>
                </button>
                <button
                  className="quick-prompt-card"
                  onClick={() => void runAnalysis("Who are the highest paid employees?")}
                >
                  <strong>💰 Compensation rank</strong>
                  <div>Find top salary earners and their job titles</div>
                </button>
                <button
                  className="quick-prompt-card"
                  onClick={() => void runAnalysis("What are the department salary totals?")}
                >
                  <strong>📈 Salary expenditure</strong>
                  <div>Total budget breakdown by department</div>
                </button>
              </div>
            </div>
          ) : (
            <div className="chat-conversation-flow">
              {chatMessages.map((msg, idx) => (
                <div
                  className={`chat-turn ${msg.role}`}
                  key={msg.id || `${msg.role}-${idx}`}
                >
                  {msg.role === "assistant" && (
                    <div className="turn-avatar ai-avatar">✧</div>
                  )}

                  <div className="turn-content">
                    {msg.role === "user" ? (
                      <div className="user-bubble">
                        <MarkdownText text={msg.content} />
                      </div>
                    ) : (
                      <>
                        {/* 1. Downward streaming / written markdown answer */}
                        {msg.content ? (
                          <MarkdownText text={msg.content} />
                        ) : msg.pending ? (
                          <div className="stream-loading">
                            <span>Analyzing database...</span>
                            <span className="typing-dots">
                              <i />
                              <i />
                              <i />
                            </span>
                          </div>
                        ) : null}

                        {/* 2. Sequential Downward Snippets (Query, Graph, Diagnosis) */}
                        <SequentialArtifacts
                          analysis={msg.analysis || null}
                          messageId={msg.id}
                        />
                      </>
                    )}
                  </div>

                  {msg.role === "user" && (
                    <div className="turn-avatar user-avatar">DA</div>
                  )}
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Scroll To Bottom Button when scrolled up */}
        {showScrollBottom && (
          <button className="scroll-bottom-pill" onClick={() => scrollToBottom("smooth")}>
            ↓ Scroll to bottom
          </button>
        )}

        {/* Floating Center Capsule Input */}
        <div className="input-dock-container">
          <div className="input-capsule">
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  void runAnalysis();
                }
              }}
              placeholder="Ask a question about your database..."
              disabled={running}
            />
            <button
              className="send-action-btn"
              onClick={() => void runAnalysis()}
              disabled={running || !query.trim()}
              title="Send message"
            >
              {running ? "…" : "↑"}
            </button>
          </div>
        </div>
      </main>

      {/* Database Connection Modal Dialog */}
      {showDatabaseDialog && (
        <div className="modal-backdrop">
          <div className="modal-window">
            <div className="modal-header">
              <h3>Connect Database</h3>
              <button
                className="modal-close-btn"
                onClick={() => setShowDatabaseDialog(false)}
              >
                ✕
              </button>
            </div>
            <div className="modal-body">
              <p>
                Enter the SQLAlchemy connection URI for this thread (e.g. MySQL, PostgreSQL, SQLite).
              </p>
              <input
                type="text"
                value={databaseUri}
                onChange={(e) => setDatabaseUri(e.target.value)}
                placeholder="mysql+pymysql://user:pass@localhost:3306/dbname"
                autoFocus
              />
            </div>
            <div className="modal-footer">
              <button
                className="btn-secondary"
                onClick={() => setShowDatabaseDialog(false)}
              >
                Cancel
              </button>
              <button
                className="btn-primary"
                onClick={submitDatabaseConnection}
                disabled={connecting || !databaseUri.trim()}
              >
                {connecting ? "Connecting..." : "Connect"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Sequential generation downward: Snippet Query -> Snippet Graph -> Diagnostic
function SequentialArtifacts({
  analysis,
  messageId,
}: {
  analysis: Analysis | null;
  messageId: string;
}) {
  if (!analysis) return null;

  const sql = analysis.validated_sql || analysis.generated_sql;
  const fig = analysis.plotly_figure;
  const diag = analysis.diagnostic_report;
  const result = analysis.primary_result;

  return (
    <div className="sequential-snippets">
      {/* 1. Snippet Query */}
      {sql && (
        <div className="snippet-card">
          <div className="snippet-card-header">
            <span className="snippet-badge">
              <span>&lt;/&gt;</span> Executed SQL Query
            </span>
            <button
              className="snippet-action-btn"
              onClick={() => void navigator.clipboard.writeText(sql)}
            >
              Copy
            </button>
          </div>
          <pre className="sql-snippet-body">
            <code>{sql};</code>
          </pre>
        </div>
      )}

      {/* 2. Snippet Graph (proportional and properly sized) */}
      {fig && fig.data && (
        <div className="snippet-card">
          <div className="snippet-card-header">
            <span className="snippet-badge">
              <span>▥</span> Interactive Visualization
            </span>
            <button
              className="snippet-action-btn"
              onClick={() => window.open(`${API}/api/chart`, "_blank")}
            >
              Open Full
            </button>
          </div>
          <div className="chart-snippet-shell">
            <Plot
              key={`plot-${messageId}`}
              data={JSON.parse(JSON.stringify(fig.data || []))}
              layout={{
                ...(fig.layout || {}),
                template: document.documentElement.getAttribute("data-theme") === "light" ? "plotly_white" : "plotly_dark",
                autosize: true,
                paper_bgcolor: "transparent",
                plot_bgcolor: "transparent",
                font: {
                  color: document.documentElement.getAttribute("data-theme") === "light" ? "#333333" : "#d0d7de",
                },
                margin: { l: 45, r: 25, t: 40, b: 40 },
                height: 380, // Harmonious, proportionate height
              }}
              useResizeHandler
              style={{ width: "100%", height: "380px" }}
              config={{ responsive: true, displaylogo: false }}
            />
          </div>
        </div>
      )}

      {/* 3. Diagnostic Report Snippet */}
      {diag && diag.diagnosis && (
        <div className="snippet-card">
          <div className="snippet-card-header">
            <span className="snippet-badge">
              <span>⌁</span> Diagnostic Insight
            </span>
            <span className="diag-summary-pill">
              ✓ {Math.round((diag.confidence || 0.95) * 100)}% Confidence
            </span>
          </div>
          <div className="diag-snippet-body">
            <p className="diag-text">{diag.diagnosis}</p>
            {diag.recommendation && (
              <div className="diag-rec">
                <strong>Action:</strong> {diag.recommendation}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
