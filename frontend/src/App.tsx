import { useEffect, useRef, useState } from "react";
import Plot from "react-plotly.js";

const API = "http://localhost:8000";

type Status = "pending" | "running" | "completed" | "failed";
type WorkflowStep = { id: string; label: string; status: Status };
type Thread = { thread_id: string; name: string; created_at?: string | null };
type ChatMessage = { id: string; role: "user" | "assistant"; content: string; pending?: boolean; analysis?: Analysis; steps?: WorkflowStep[] };
type Result = { columns: string[]; rows: Record<string, unknown>[]; row_count: number };
type Analysis = {
  user_query?: string;
  schema_overview?: string;
  answer?: string | null;
  generated_sql?: string | null;
  validated_sql?: string | null;
  primary_result?: Result | null;
  plotly_figure?: { data?: unknown[]; layout?: Record<string, unknown> } | null;
  plotly_html_path?: string | null;
  diagnostic_report?: { observation?: string; diagnosis?: string; hypotheses_evaluated?: string[] } | null;
  insights?: string[];
  final_response?: string | null;
  sql_error?: string | null;
};

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

function App() {
  const [threads, setThreads] = useState<Thread[]>([]);
  const [selected, setSelected] = useState<Thread | null>(null);
  const [query, setQuery] = useState("");
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [steps, setSteps] = useState(defaultSteps);
  const [activeTab, setActiveTab] = useState("answer");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const [schema, setSchema] = useState("");
  const [databaseReady, setDatabaseReady] = useState<Record<string, boolean>>({});
  const [showDatabaseDialog, setShowDatabaseDialog] = useState(false);
  const [databaseUri, setDatabaseUri] = useState("");
  const [connecting, setConnecting] = useState(false);
  const [messagesByThread, setMessagesByThread] = useState<Record<string, ChatMessage[]>>({});
  const chatMessages = selected ? messagesByThread[selected.thread_id] || [] : [];

  const loadThreads = async () => {
    const response = await fetch(`${API}/api/threads`);
    if (response.ok) setThreads(await response.json());
  };

  useEffect(() => { void loadThreads(); }, []);

  const createThread = async () => {
    const response = await fetch(`${API}/api/threads`, { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
    const thread = await response.json() as Thread;
    setThreads((old) => [thread, ...old.filter((item) => item.thread_id !== thread.thread_id)]);
    setDatabaseReady((old) => ({ ...old, [thread.thread_id]: false }));
    await selectThread(thread, false);
    setShowDatabaseDialog(true);
  };

  const selectThread = async (thread: Thread, knownReady?: boolean) => {
    setSelected(thread);
    setAnalysis(null);
    setSteps(defaultSteps());
    setError("");
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
      const records = await historyResponse.json() as Analysis[];
      const restored: ChatMessage[] = [];
      records.forEach((record, index) => {
        const text = record.user_query || "";
        restored.push({ id: `${thread.thread_id}-${index}-user`, role: "user", content: text });
        restored.push({ id: `${thread.thread_id}-${index}-assistant`, role: "assistant", content: record.answer || record.final_response || "Analysis completed.", analysis: record, steps: defaultSteps().map((step) => ({ ...step, status: "completed" })) });
      });
      setMessagesByThread((old) => ({ ...old, [thread.thread_id]: restored }));
      const assistants = restored.filter((message) => message.role === "assistant");
      const latest = assistants[assistants.length - 1];
      if (latest?.analysis) setAnalysis(latest.analysis);
    }
  };

  const connectDatabase = async () => {
    if (selected) setShowDatabaseDialog(true);
  };

  const submitDatabaseConnection = async () => {
    if (!selected || !databaseUri.trim() || connecting) return;
    setConnecting(true);
    setError("");
    try {
      const response = await fetch(`${API}/api/database/connect`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ thread_id: selected.thread_id, database_uri: databaseUri.trim() }),
      });
      if (!response.ok) throw new Error("Connection failed");
      setDatabaseReady((old) => ({ ...old, [selected.thread_id]: true }));
      setDatabaseUri("");
      setShowDatabaseDialog(false);
      const schemaResponse = await fetch(`${API}/api/schema/${selected.thread_id}`);
      if (schemaResponse.ok) setSchema((await schemaResponse.json()).schema_overview || "");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Connection failed");
    } finally {
      setConnecting(false);
    }
  };

  const runAnalysis = async () => {
    if (!selected || !query.trim() || running) return;
    const submittedQuery = query.trim();
    if (!databaseReady[selected.thread_id] && !isGreeting(submittedQuery)) {
      setError("Connect a database to this thread before asking a data question.");
      setShowDatabaseDialog(true);
      return;
    }
    const userMessage: ChatMessage = { id: `${selected.thread_id}-${Date.now()}-user`, role: "user", content: submittedQuery };
    const assistantMessage: ChatMessage = { id: `${selected.thread_id}-${Date.now()}-assistant`, role: "assistant", content: "Working through the analysis…", pending: true, steps: defaultSteps() };
    setMessagesByThread((old) => ({ ...old, [selected.thread_id]: [...(old[selected.thread_id] || []), userMessage, assistantMessage] }));
    setQuery("");
    if (isGreeting(submittedQuery)) {
      const greeting = "Hello! How can I help you analyze your data?";
      setMessagesByThread((old) => ({ ...old, [selected.thread_id]: (old[selected.thread_id] || []).map((message, index, list) => index === list.length - 1 ? { ...message, content: greeting, pending: false } : message) }));
      return;
    }
    setRunning(true); setError(""); setAnalysis(null); setSteps(defaultSteps()); setActiveTab("answer");
    try {
      const response = await fetch(`${API}/api/analyze/${selected.thread_id}/stream?query=${encodeURIComponent(submittedQuery)}`);
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
          const event = JSON.parse(line.slice(6)) as { type: string; step?: string; status?: Status; data?: Analysis; message?: string };
          if (event.type === "workflow" && event.step && event.status) {
            setSteps((old) => old.map((step) => step.id === event.step ? { ...step, status: event.status! } : step));
            setMessagesByThread((old) => ({ ...old, [selected.thread_id]: (old[selected.thread_id] || []).map((message, index, list) => index === list.length - 1 ? { ...message, steps: (message.steps || defaultSteps()).map((step) => step.id === event.step ? { ...step, status: event.status! } : step), analysis: event.data ? { ...(message.analysis || {}), ...event.data } : message.analysis } : message) }));
            if (event.data) setAnalysis((old) => ({ ...(old || {}), ...event.data }));
          } else if (event.type === "result" && event.data) {
            setAnalysis(event.data); setActiveTab(event.data.plotly_figure ? "visualization" : "answer");
            const responseText = event.data.answer || event.data.final_response || "Analysis completed.";
            setMessagesByThread((old) => ({ ...old, [selected.thread_id]: (old[selected.thread_id] || []).map((message, index, list) => index === list.length - 1 ? { ...message, content: responseText, pending: false, analysis: event.data, steps } : message) }));
          } else if (event.type === "error") {
            const message = event.message || "Analysis failed";
            setError(message);
            setMessagesByThread((old) => ({ ...old, [selected.thread_id]: (old[selected.thread_id] || []).map((item, index, list) => index === list.length - 1 ? { ...item, content: message, pending: false, steps } : item) }));
          }
        }
      }
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : "Analysis failed";
      setError(message);
      setMessagesByThread((old) => ({ ...old, [selected.thread_id]: (old[selected.thread_id] || []).map((item, index, list) => index === list.length - 1 ? { ...item, content: message, pending: false, steps } : item) }));
    } finally { setRunning(false); }
  };

  const availableTabs = [
    { id: "answer", label: "Answer", icon: "✧", show: Boolean(analysis?.answer || analysis?.final_response) },
    { id: "sql", label: "SQL Query", icon: "</>", show: Boolean(analysis?.validated_sql) },
    { id: "results", label: "Results", icon: "▦", show: Boolean(analysis?.primary_result?.columns?.length) },
    { id: "visualization", label: "Visualization", icon: "▥", show: Boolean(analysis?.plotly_figure) },
    { id: "diagnostic", label: "Diagnostic", icon: "⌁", show: Boolean(analysis?.diagnostic_report?.diagnosis) },
    { id: "insights", label: "Insights", icon: "✦", show: Boolean(analysis?.insights?.length) },
  ].filter((tab) => tab.show);

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark">✧</div><div><strong>Insight AI</strong><span>AI Data Analyst</span></div></div>
      <div className="sidebar-actions"><button className="primary wide" onClick={createThread}>＋ New Thread</button><button className="secondary wide" onClick={connectDatabase}>▱ &nbsp; Add Database</button></div>
      <div className="section-title">RECENT THREADS</div>
      <div className="thread-list">{threads.map((thread) => <button className={`thread ${selected?.thread_id === thread.thread_id ? "selected" : ""}`} key={thread.thread_id} onClick={() => void selectThread(thread)}><i /> <span>{thread.name}<small>{thread.created_at ? "recent" : "ready"}</small></span></button>)}</div>
      <div className="sidebar-bottom"><div className="section-title">DATABASE</div><div className="database-card"><span className="status-dot" /> <span>Connected</span><small>{selected ? "Thread database" : "No database selected"}</small></div><div className="profile"><div className="avatar">DA</div><div><strong>Data Analyst</strong><small>analyst@insight.ai</small></div><span className="profile-icons">☼ &nbsp; ⚙</span></div></div>
    </aside>
    <main className="main-panel">
      {!selected ? <EmptyState onCreate={createThread} onConnect={createThread} /> : <>
        <header className="analysis-header"><div><div className="eyebrow">THREAD / {selected.thread_id}</div><h1>{selected.name}</h1></div></header>
        {chatMessages.length > 0 ? <ChatMessages messages={chatMessages} /> : <div className="schema-card"><div className="eyebrow">SCHEMA EXPLORER</div><h2>Connected data source</h2><SchemaExplorer overview={schema} /></div>}
        {running && <div className="stream-note"><span className="status-dot" />Insight AI is analyzing this thread…</div>}
        <section className="query-bar composer"><input value={query} onChange={(event) => setQuery(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter") void runAnalysis(); }} placeholder="Ask a question about your data..." disabled={running} /><button className="primary" onClick={() => void runAnalysis()} disabled={running || !query.trim()}>{running ? "Analyzing…" : "Send"} <span>→</span></button></section>
        {error && <div className="error-card"><strong>Analysis failed</strong><span>{error}</span><button className="secondary" onClick={() => void runAnalysis()}>Retry</button></div>}
      </>}
    </main>
  </div>;
}

function EmptyState({ onCreate, onConnect }: { onCreate: () => void; onConnect: () => void }) { return <div className="empty-state"><div className="empty-icon">▱</div><h1>Select or create a thread</h1><p>Threads keep your analyses organized.<br />Each thread has its own database connection and conversation history.</p><div><button className="primary" onClick={onCreate}>New Thread</button><button className="secondary" onClick={onConnect}>Connect Database</button></div></div>; }

function DatabaseDialog({ value, connecting, required, onChange, onSubmit, onClose }: { value: string; connecting: boolean; required: boolean; onChange: (value: string) => void; onSubmit: () => void; onClose: () => void }) {
  return <div className="dialog-backdrop" role="presentation"><div className="database-dialog" role="dialog" aria-modal="true" aria-labelledby="database-dialog-title"><div className="content-heading"><div><div className="eyebrow">THREAD DATABASE</div><h2 id="database-dialog-title">Connect this conversation</h2></div>{!required && <button className="dialog-close" onClick={onClose} aria-label="Close">×</button>}</div><p>Enter a database URL for this thread. It is sent only to the existing backend connection endpoint and is cleared from this UI after connection.</p><input autoFocus type="password" value={value} onChange={(event) => onChange(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter") onSubmit(); }} placeholder="Database connection URL" autoComplete="off" /><div className="dialog-actions"><button className="secondary" onClick={onClose} disabled={required || connecting}>Cancel</button><button className="primary" onClick={onSubmit} disabled={!value.trim() || connecting}>{connecting ? "Connecting…" : "Connect"}</button></div></div></div>;
}

function Workflow({ steps }: { steps: WorkflowStep[] }) {
  const phases = [
    { id: "analysis", label: "Analysis", source: ["intent_analyzer"] },
    { id: "schema", label: "Schema", source: ["decomposition_node"] },
    { id: "sql", label: "SQL Generation", source: ["sql_agent"] },
    { id: "execution", label: "Execution", source: ["execution_planner"] },
    { id: "result", label: "Result", source: ["final_composer"] },
  ];
  const statusFor = (source: string[]) => {
    const matching = steps.filter((step) => source.includes(step.id));
    if (matching.some((step) => step.status === "failed")) return "failed";
    if (matching.some((step) => step.status === "running")) return "running";
    if (matching.length && matching.every((step) => step.status === "completed")) return "completed";
    return "pending";
  };
  return <div className="workflow compact-workflow"><div className="eyebrow">PROGRESS</div>{phases.map((phase) => { const status = statusFor(phase.source); return <div className={`workflow-step ${status}`} key={phase.id}><span className="step-mark">{status === "completed" ? "✓" : status === "running" ? "•" : status === "failed" ? "!" : ""}</span><div><strong>{phase.label}</strong><small>{status}</small></div></div>; })}</div>;
}

function Answer({ analysis }: { analysis: Analysis | null }) { const text = analysis?.answer || analysis?.final_response || "Waiting for the final analysis…"; return <section className="tab-content"><div className="eyebrow">ANSWER</div><h2>Analysis result</h2><MarkdownText text={text} /></section>; }
function ChatMessages({ messages }: { messages: ChatMessage[] }) { return <section className="chat-panel">{messages.map((message, index) => <div className={`chat-row ${message.role}`} key={message.id || `${message.role}-${index}`}><div className="chat-avatar">{message.role === "user" ? "DA" : "✧"}</div><div className="chat-bubble"><div className="chat-role">{message.role === "user" ? "You" : "Insight AI"}</div>{(message.role === "user" || (!message.analysis && !message.pending)) && <MarkdownText text={message.content} />}{message.pending && <><MarkdownText text={message.content} /><span className="typing"><i /> <i /> <i /></span></>}{message.role === "assistant" && <MessageArtifacts analysis={message.analysis || null} steps={message.steps || defaultSteps()} pending={Boolean(message.pending)} />}</div></div>)}</section>; }
function MessageArtifacts({ analysis, steps, pending }: { analysis: Analysis | null; steps: WorkflowStep[]; pending: boolean }) {
  const [active, setActive] = useState("answer");
  if (!analysis && !pending) return null;
  const tabs = [
    { id: "answer", label: "Answer", show: true },
    { id: "sql", label: "Query", show: Boolean(analysis?.validated_sql || analysis?.generated_sql) },
    { id: "visualization", label: "Visualization", show: Boolean(analysis?.plotly_figure) },
    { id: "insights", label: "Insights", show: Boolean(analysis?.insights?.length) },
    { id: "results", label: "Results", show: Boolean(analysis?.primary_result) },
    { id: "diagnostic", label: "Diagnostic", show: Boolean(analysis?.diagnostic_report?.diagnosis) },
  ].filter((tab) => tab.show);
  const selected = tabs.some((tab) => tab.id === active) ? active : "answer";
  return <section className="message-artifacts"><Workflow steps={steps} /><div className="response-container"><nav className="tabs response-tabs">{tabs.map((tab) => <button className={selected === tab.id ? "active" : ""} key={tab.id} onClick={() => setActive(tab.id)}>{tab.label}</button>)}</nav>{selected === "answer" && <Answer analysis={analysis} />}{selected === "sql" && <Sql sql={analysis?.validated_sql || analysis?.generated_sql || ""} />}{selected === "visualization" && <Visualization figure={analysis?.plotly_figure || null} />}{selected === "insights" && <Insights values={analysis?.insights || []} />}{selected === "results" && <Results result={analysis?.primary_result || null} />}{selected === "diagnostic" && <Diagnostic report={analysis?.diagnostic_report || null} />}{pending && <div className="response-loading">Waiting for structured results…</div>}</div></section>;
}
function SchemaExplorer({ overview }: { overview: string }) { const tables = overview.split("\n").map((line) => { const match = line.match(/^Table `?([^`]+)`?:\s*(.*)$/); if (!match) return null; return { name: match[1], columns: match[2].split(/,\s*/).map((column) => column.replace(/\s*\([^)]*\)$/, "")) }; }).filter(Boolean) as { name: string; columns: string[] }[]; if (!tables.length) return <pre>{overview || "Schema will appear after a database connection."}</pre>; return <div className="schema-tree">{tables.map((table) => <div className="schema-table" key={table.name}><strong>▾ {table.name}</strong>{table.columns.map((column) => <span key={column}>├── {column}</span>)}</div>)}</div>; }
function MarkdownText({ text }: { text: string }) { return <div className="markdown plain-text">{text}</div>; }
function Sql({ sql }: { sql: string }) { return <section className="tab-content"><div className="content-heading"><div><div className="eyebrow">SQL QUERY</div><h2>Executed SQL</h2></div><button className="secondary" onClick={() => void navigator.clipboard.writeText(sql)}>Copy</button></div><pre className="sql-block"><code>{sql}</code></pre></section>; }

function Results({ result }: { result: Result | null }) { const [search, setSearch] = useState(""); const [page, setPage] = useState(0); const pageSize = 10; const rows = (result?.rows || []).filter((row) => JSON.stringify(row).toLowerCase().includes(search.toLowerCase())); const visible = rows.slice(page * pageSize, (page + 1) * pageSize); const download = () => { if (!result) return; const csv = [result.columns, ...result.rows.map((row) => result.columns.map((column) => JSON.stringify(row[column] ?? "")))].map((row) => row.join(",")).join("\n"); const url = URL.createObjectURL(new Blob([csv], { type: "text/csv" })); const anchor = document.createElement("a"); anchor.href = url; anchor.download = "insight-results.csv"; anchor.click(); URL.revokeObjectURL(url); }; return <section className="tab-content"><div className="content-heading"><div><div className="eyebrow">RESULTS</div><h2>{result?.row_count || 0} rows returned</h2></div><div><input className="small-input" placeholder="Search results" value={search} onChange={(event) => { setSearch(event.target.value); setPage(0); }} /><button className="secondary" onClick={download}>CSV</button></div></div><div className="table-wrap"><table><thead><tr>{(result?.columns || []).map((column) => <th key={column}>{column}</th>)}</tr></thead><tbody>{visible.map((row, index) => <tr key={index}>{(result?.columns || []).map((column) => <td key={column}>{row[column] == null ? <span className="null">NULL</span> : String(row[column])}</td>)}</tr>)}</tbody></table></div><div className="pagination"><span>Showing {rows.length ? page * pageSize + 1 : 0}–{Math.min((page + 1) * pageSize, rows.length)} of {rows.length}</span><button className="secondary" disabled={page === 0} onClick={() => setPage(page - 1)}>‹</button><button className="secondary" disabled={(page + 1) * pageSize >= rows.length} onClick={() => setPage(page + 1)}>›</button></div></section>; }

function Visualization({ figure }: { figure: Analysis["plotly_figure"] }) { const plotRef = useRef<any>(null); const download = async (format: "png" | "svg") => { if (!plotRef.current) return; const url = await plotRef.current.toImage({ format, height: 600, width: 1000 }); const anchor = document.createElement("a"); anchor.href = url; anchor.download = `insight-chart.${format}`; anchor.click(); }; if (!figure) return <section className="tab-content"><div className="eyebrow">VISUALIZATION</div><h2>Visualization unavailable</h2></section>; const rawTitle = figure.layout?.title; const title = typeof rawTitle === "object" && rawTitle !== null && "text" in rawTitle ? String((rawTitle as { text?: unknown }).text || "Interactive visualization") : String(rawTitle || "Interactive visualization"); return <section className="tab-content"><div className="content-heading"><div><div className="eyebrow">VISUALIZATION</div><h2>{title}</h2></div><div className="chart-actions"><button onClick={() => void download("png")}>▧ PNG</button><button onClick={() => void download("svg")}>⇩ SVG</button><button onClick={() => window.open(`${API}/api/chart`, "_blank")}>▱ HTML</button><button onClick={() => document.documentElement.requestFullscreen()}>↗</button></div></div><div className="chart-shell"><Plot ref={plotRef} data={(figure.data || []) as any[]} layout={{ ...(figure.layout || {}), template: "plotly_dark", autosize: true, paper_bgcolor: "#111722", plot_bgcolor: "#111722", font: { color: "#aebbd0" } }} useResizeHandler style={{ width: "100%", height: "520px" }} config={{ responsive: true, displaylogo: false }} /></div></section>; }
function Diagnostic({ report }: { report: Analysis["diagnostic_report"] }) { return <section className="tab-content"><div className="eyebrow">DIAGNOSTIC ANALYSIS</div><h2>Investigation and evidence</h2><div className="diagnostic-box"><strong>Observation</strong><p>{report?.observation}</p><strong>Investigation SQL</strong>{(report?.hypotheses_evaluated || []).map((item, index) => <div className="evidence" key={index}>✓ {item}</div>)}<strong>Diagnosis</strong><p>{report?.diagnosis}</p></div></section>; }
function Insights({ values }: { values: string[] }) { return <section className="tab-content"><div className="eyebrow">KEY INSIGHTS</div><h2>Key Insights</h2><div className="insight-grid">{values.map((value, index) => <div className="insight-card" key={index}><span>✦</span><p>{value}</p></div>)}</div></section>; }

export default App;
