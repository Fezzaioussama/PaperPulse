import { useState, useEffect, useMemo, useCallback } from "react";

// ──────────────────────────────────────────────────────────────────────────
// Topic presets → arXiv search queries. Categories (cat:cs.CL …) give far
// higher signal than raw keyword search, and we mix in keywords for the
// cross-cutting themes (agents, RAG, reasoning…) that aren't real categories.
// ──────────────────────────────────────────────────────────────────────────
const PRESETS = [
  { label: "LLMs",            query: "cat:cs.CL" },
  { label: "AI Agents",       query: 'cat:cs.AI AND (all:agent OR all:agentic OR all:"LLM agent")' },
  { label: "Reasoning",       query: 'all:reasoning AND (all:"chain-of-thought" OR all:"large language model")' },
  { label: "RAG",             query: 'all:"retrieval-augmented generation" OR all:RAG' },
  { label: "RLHF / Alignment",query: 'all:RLHF OR all:DPO OR all:"reinforcement learning from human feedback"' },
  { label: "Multimodal",      query: "cat:cs.CV AND all:multimodal" },
  { label: "Computer Vision", query: "cat:cs.CV" },
  { label: "Diffusion",       query: "cat:cs.CV AND all:diffusion" },
  { label: "Mech Interp",     query: 'all:"mechanistic interpretability"' },
  { label: "Efficiency / MoE",query: 'all:quantization OR all:"mixture of experts" OR all:MoE' },
  { label: "Robotics / VLA",  query: "cat:cs.RO AND all:learning" },
  { label: "AI Safety",       query: 'all:"AI safety" OR all:"AI alignment"' },
  { label: "Benchmarks",      query: "cat:cs.CL AND all:benchmark" },
  { label: "Machine Learning",query: "cat:cs.LG" },
];

const COUNT_OPTIONS = [10, 20, 30];
const DEFAULT_MODEL = import.meta.env.REACT_APP_OPENROUTER_MODEL || "anthropic/claude-3.5-sonnet";
const ENV_KEY = import.meta.env.REACT_APP_OPENROUTER_API_KEY || "";

const LS = {
  key: "openrouter_key",
  model: "openrouter_model",
  theme: "pb_theme",
  bookmarks: "pb_bookmarks",
  cache: "pb_cache",
};

const TODAY = new Date().toLocaleDateString("en-US", {
  weekday: "long", year: "numeric", month: "long", day: "numeric",
});

// ── localStorage helpers ──
const getLS = (k, fallback = null) => {
  try { const v = localStorage.getItem(k); return v == null ? fallback : v; }
  catch { return fallback; }
};
const getJSON = (k, fallback) => {
  try { const v = localStorage.getItem(k); return v == null ? fallback : JSON.parse(v); }
  catch { return fallback; }
};
const setJSON = (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} };

const getApiKey = () => getLS(LS.key) || ENV_KEY;
const getModel = () => getLS(LS.model) || DEFAULT_MODEL;

// ── JSON extraction from arbitrary model text ──
function extractJson(raw) {
  if (!raw) return null;
  let s = raw.replace(/```json/gi, "").replace(/```/g, "").trim();
  try { return JSON.parse(s); } catch (_) {}
  const start = s.indexOf("[") !== -1 ? s.indexOf("[") : s.indexOf("{");
  const lastArr = s.lastIndexOf("]");
  const lastObj = s.lastIndexOf("}");
  const end = Math.max(lastArr, lastObj);
  if (start !== -1 && end > start) {
    try { return JSON.parse(s.slice(start, end + 1)); } catch (_) {}
  }
  return null;
}

async function apiCall(prompt, maxTokens = 900) {
  const apiKey = getApiKey();
  if (!apiKey) throw new Error("OpenRouter API key not found. Open Settings (⚙) to add one, or set REACT_APP_OPENROUTER_API_KEY.");
  const model = getModel();
  if (!model) throw new Error("No OpenRouter model set. Open Settings (⚙) to choose one.");

  const res = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${apiKey}`,
      "HTTP-Referer": "http://localhost",
      "X-Title": "ArXiv Paper Briefings",
    },
    body: JSON.stringify({ model, max_tokens: maxTokens, messages: [{ role: "user", content: prompt }] }),
  });
  if (!res.ok) throw new Error(`API status ${res.status}`);
  const data = await res.json();
  if (data.error) throw new Error(data.error.message || "API error");
  let raw = "";
  for (const choice of data.choices || []) if (choice.message?.content) raw += choice.message.content + "\n";
  return raw;
}

// ── generic fetch with CORS-proxy fallbacks; returns response text ──
async function fetchWithProxies(url) {
  const tries = [
    url,
    `https://corsproxy.io/?${encodeURIComponent(url)}`,
    `https://api.allorigins.win/raw?url=${encodeURIComponent(url)}`,
  ];
  for (const u of tries) {
    try {
      const res = await fetch(u);
      if (res.ok) {
        const text = await res.text();
        if (text) return text;
      }
    } catch (_) { /* next */ }
  }
  return null;
}

function normId(url = "") {
  const m = url.match(/(\d{4}\.\d{4,5})(v\d+)?/);
  return m ? m[1] : url.toLowerCase().trim();
}

// ── arXiv ──
function parseArxivXml(xml) {
  const doc = new DOMParser().parseFromString(xml, "application/xml");
  const entries = Array.from(doc.getElementsByTagName("entry"));
  return entries.map(e => {
    const get = (tag) => {
      const el = e.getElementsByTagName(tag)[0];
      return el ? el.textContent.trim().replace(/\s+/g, " ") : "";
    };
    const authors = Array.from(e.getElementsByTagName("author"))
      .map(a => a.getElementsByTagName("name")[0]?.textContent?.trim()).filter(Boolean);
    const authorStr = authors.length > 2 ? `${authors[0]} et al.` : authors.join(", ") || "—";
    const published = get("published");
    const links = Array.from(e.getElementsByTagName("link"));
    const absLink = links.find(l => l.getAttribute("rel") === "alternate");
    return {
      title: get("title"),
      authors: authorStr,
      date: published ? published.slice(0, 10) : "",
      url: absLink ? absLink.getAttribute("href") : get("id"),
      abstract: get("summary"),
      source: "arxiv",
    };
  }).filter(p => p.title && p.abstract);
}

async function fetchArxiv(query, count) {
  const apiUrl = `https://export.arxiv.org/api/query?search_query=${encodeURIComponent(query)}&sortBy=submittedDate&sortOrder=descending&max_results=${count}`;
  const xml = await fetchWithProxies(apiUrl);
  if (xml && xml.includes("<entry")) {
    const parsed = parseArxivXml(xml);
    if (parsed.length) return parsed;
  }
  return [];
}

// ── Hugging Face Daily Papers (community-upvoted "what's hot today") ──
async function fetchHFDaily(count) {
  const text = await fetchWithProxies("https://huggingface.co/api/daily_papers");
  if (!text) return [];
  let items;
  try { items = JSON.parse(text); } catch { return []; }
  if (!Array.isArray(items)) return [];
  return items.slice(0, count).map(it => {
    const p = it.paper || it;
    const authors = (p.authors || []).map(a => a.name).filter(Boolean);
    const authorStr = authors.length > 2 ? `${authors[0]} et al.` : authors.join(", ") || "—";
    const id = p.id || "";
    return {
      title: (p.title || "").trim().replace(/\s+/g, " "),
      authors: authorStr,
      date: (it.publishedAt || p.publishedAt || "").slice(0, 10),
      url: id ? `https://arxiv.org/abs/${id}` : (p.url || ""),
      abstract: (p.summary || "").trim().replace(/\s+/g, " "),
      source: "hf",
      upvotes: p.upvotes ?? it.upvotes ?? 0,
    };
  }).filter(p => p.title && p.abstract);
}

function dedupe(papers) {
  const seen = new Set();
  const out = [];
  for (const p of papers) {
    const k = normId(p.url) || p.title.toLowerCase();
    if (seen.has(k)) continue;
    seen.add(k);
    out.push(p);
  }
  return out;
}

// ── per-paper enrichment: TL;DR, key points, method, results, impact, tags ──
async function summarizeOne(orig) {
  const sumPrompt = `You are an AI research analyst summarizing a single arXiv paper.

Title: ${orig.title}
Authors: ${orig.authors}
Abstract: ${orig.abstract}

Respond with ONLY a JSON object (no markdown fences, no text before or after):
{
  "tldr": "one punchy sentence capturing the core idea",
  "key_points": ["point 1", "point 2", "point 3", "point 4"],
  "method": "the approach/technique in 1-2 sentences",
  "results": "main findings or claimed improvements in 1 sentence",
  "impact": <integer 1-5, how significant this is for the AI field>,
  "why": "one sentence on why this matters to AI/LLM/agent researchers",
  "tags": ["3-5 short lowercase topic tags, e.g. agents, rag, benchmark, open-weights"]
}`;
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      const raw = await apiCall(sumPrompt, 1100);
      const s = extractJson(raw);
      if (s && (s.tldr || s.key_points)) {
        return {
          ...orig,
          tldr: s.tldr || orig.abstract.slice(0, 160) + "…",
          key_points: Array.isArray(s.key_points) ? s.key_points : [],
          method: s.method || "",
          results: s.results || "",
          impact: Math.max(1, Math.min(5, parseInt(s.impact, 10) || 3)),
          why: s.why || "",
          tags: Array.isArray(s.tags) ? s.tags.map(t => String(t).toLowerCase().trim()).filter(Boolean).slice(0, 5) : [],
        };
      }
    } catch (_) {}
    await new Promise(r => setTimeout(r, 700));
  }
  return {
    ...orig,
    tldr: orig.abstract.slice(0, 180) + (orig.abstract.length > 180 ? "…" : ""),
    key_points: [], method: "", results: "", impact: 3, why: "",
    tags: [], abstractFallback: true,
  };
}

// ── Markdown export ──
function buildMarkdown(papers, label) {
  const lines = [`# AI Paper Briefings — ${label}`, `_${TODAY}_`, ""];
  papers.forEach((p, i) => {
    lines.push(`## ${i + 1}. ${p.title}`);
    lines.push(`*${p.authors}${p.date ? " · " + p.date : ""}${p.source === "hf" ? " · 🤗 " + (p.upvotes || 0) + " upvotes" : ""}*`);
    lines.push("");
    lines.push(`**Impact:** ${"★".repeat(p.impact || 3)}${"☆".repeat(5 - (p.impact || 3))}`);
    if (p.tags?.length) lines.push(`**Tags:** ${p.tags.map(t => "#" + t).join(" ")}`);
    lines.push("");
    lines.push(`**TL;DR:** ${p.tldr}`);
    if (p.why) lines.push(`\n**Why it matters:** ${p.why}`);
    if (p.key_points?.length) { lines.push("\n**Key points:**"); p.key_points.forEach(k => lines.push(`- ${k}`)); }
    if (p.method) lines.push(`\n**Method:** ${p.method}`);
    if (p.results) lines.push(`\n**Results:** ${p.results}`);
    if (p.url) lines.push(`\n[Open paper](${p.url})`);
    lines.push("\n---\n");
  });
  return lines.join("\n");
}

// ──────────────────────────────────────────────────────────────────────────
function ImpactDots({ n = 3 }) {
  return (
    <span className="impact" title={`Impact ${n}/5`}>
      {[1, 2, 3, 4, 5].map(i => (
        <span key={i} className={`idot ${i <= n ? "on" : ""}`} />
      ))}
    </span>
  );
}

function PaperCard({ paper, index, saved, onToggleSave, onTagClick }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="card" style={{ animationDelay: `${(index % 6) * 50}ms` }}>
      <div className="card-head" onClick={() => setOpen(o => !o)}>
        <div className="card-num">{String(index + 1).padStart(2, "0")}</div>
        <div className="card-head-main">
          <div className="card-title">{paper.title}</div>
          <div className="card-meta">
            <span>{paper.authors}</span>
            {paper.date && <><span className="dot">·</span><span>{paper.date}</span></>}
            <span className={`src-badge ${paper.source}`}>
              {paper.source === "hf" ? `🤗 ${paper.upvotes || 0}` : "arXiv"}
            </span>
            <ImpactDots n={paper.impact} />
          </div>
          {paper.tags?.length > 0 && (
            <div className="tag-row">
              {paper.tags.map(t => (
                <button key={t} className="tag" onClick={e => { e.stopPropagation(); onTagClick(t); }}>#{t}</button>
              ))}
            </div>
          )}
        </div>
        <button
          className={`save-btn ${saved ? "on" : ""}`}
          title={saved ? "Remove bookmark" : "Bookmark"}
          onClick={e => { e.stopPropagation(); onToggleSave(paper); }}
        >{saved ? "★" : "☆"}</button>
        <div className={`chev ${open ? "up" : ""}`}>⌄</div>
      </div>

      <div className="card-tldr">{paper.tldr}</div>

      {open && (
        <div className="card-body">
          {paper.why && (
            <div className="field"><span className="field-label">Why it matters</span>
              <span className="field-val">{paper.why}</span></div>
          )}
          {paper.key_points?.length > 0 && (
            <div className="kp-block">
              <div className="kp-label">Key Points</div>
              <ul className="kp-list">{paper.key_points.map((k, i) => <li key={i}>{k}</li>)}</ul>
            </div>
          )}
          {paper.method && (
            <div className="field"><span className="field-label">Method</span>
              <span className="field-val">{paper.method}</span></div>
          )}
          {paper.results && (
            <div className="field"><span className="field-label">Results</span>
              <span className="field-val">{paper.results}</span></div>
          )}
          {paper.abstractFallback && paper.abstract && (
            <div className="field"><span className="field-label">Abstract</span>
              <span className="field-val">{paper.abstract}</span></div>
          )}
          {paper.url && (
            <a className="card-link" href={paper.url} target="_blank" rel="noreferrer">↗ Open paper</a>
          )}
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [selected, setSelected] = useState(["LLMs"]);
  const [customTopic, setCustomTopic] = useState("");
  const [useHF, setUseHF] = useState(true);
  const [count, setCount] = useState(20);

  const [status, setStatus] = useState("idle"); // idle | loading | done | error
  const [papers, setPapers] = useState([]);
  const [progress, setProgress] = useState({ done: 0, total: 0 });
  const [phase, setPhase] = useState("");
  const [errorMsg, setErrorMsg] = useState("");
  const [lastLabel, setLastLabel] = useState("");

  const [query, setQuery] = useState("");
  const [tagFilter, setTagFilter] = useState("");
  const [minImpact, setMinImpact] = useState(0);
  const [view, setView] = useState("all"); // all | saved

  const [bookmarks, setBookmarks] = useState(() => getJSON(LS.bookmarks, {}));
  const [theme, setTheme] = useState(() => getLS(LS.theme, "light"));
  const [showSettings, setShowSettings] = useState(false);
  const [keyInput, setKeyInput] = useState(() => getLS(LS.key, ""));
  const [modelInput, setModelInput] = useState(() => getModel());
  const [toast, setToast] = useState("");

  // restore last digest from cache (no auto-fetch — saves API credits)
  useEffect(() => {
    const cached = getJSON(LS.cache, null);
    if (cached?.papers?.length) {
      setPapers(cached.papers);
      setLastLabel(cached.label || "");
      setStatus("done");
    }
  }, []);

  // theme → body + persistence
  useEffect(() => {
    try { localStorage.setItem(LS.theme, theme); } catch {}
    document.body.style.background = theme === "dark" ? "#16140f" : "#faf7f0";
  }, [theme]);

  useEffect(() => { setJSON(LS.bookmarks, bookmarks); }, [bookmarks]);

  const flash = useCallback((msg) => { setToast(msg); setTimeout(() => setToast(""), 1800); }, []);

  const toggleTopic = (label) => {
    setSelected(prev => prev.includes(label) ? prev.filter(t => t !== label) : [...prev, label]);
  };

  const activeLabel = useMemo(() => {
    const parts = [...selected];
    if (customTopic.trim()) parts.push(`"${customTopic.trim()}"`);
    if (useHF) parts.push("HF Daily");
    return parts.length ? parts.join(" + ") : "—";
  }, [selected, customTopic, useHF]);

  const toggleSave = useCallback((paper) => {
    const k = normId(paper.url) || paper.title;
    setBookmarks(prev => {
      const next = { ...prev };
      if (next[k]) delete next[k]; else next[k] = paper;
      return next;
    });
  }, []);

  const isSaved = (paper) => !!bookmarks[normId(paper.url) || paper.title];

  const run = async () => {
    if (!getApiKey()) { setShowSettings(true); flash("Add an OpenRouter API key first"); return; }
    const queries = selected.map(l => PRESETS.find(p => p.label === l)?.query).filter(Boolean);
    if (customTopic.trim()) queries.push(`all:${customTopic.trim()}`);
    if (queries.length === 0 && !useHF) { flash("Pick at least one topic or enable HF Daily"); return; }

    setStatus("loading"); setPapers([]); setErrorMsg(""); setView("all");
    setProgress({ done: 0, total: 0 });

    try {
      setPhase("Fetching papers from arXiv" + (useHF ? " + Hugging Face…" : "…"));
      const perSource = Math.max(6, Math.ceil(count / Math.max(1, queries.length)));
      const buckets = await Promise.all([
        ...queries.map(q => fetchArxiv(q, perSource)),
        ...(useHF ? [fetchHFDaily(count)] : []),
      ]);

      let list = dedupe(buckets.flat());
      // HF (upvoted) first, then keep newest
      list.sort((a, b) => (b.source === "hf" ? 1 : 0) - (a.source === "hf" ? 1 : 0) || (b.date || "").localeCompare(a.date || ""));
      list = list.slice(0, count);
      if (list.length === 0) throw new Error("No papers found for this selection. Try different topics.");

      setProgress({ done: 0, total: list.length });
      setPhase("Summarizing + scoring each paper…");

      const collected = new Array(list.length).fill(null);
      let completed = 0, cursor = 0;
      const CONCURRENCY = 3;
      const worker = async () => {
        while (cursor < list.length) {
          const i = cursor++;
          collected[i] = await summarizeOne(list[i]);
          completed++;
          setProgress({ done: completed, total: list.length });
          setPapers(collected.filter(Boolean));
        }
      };
      await Promise.all(Array.from({ length: CONCURRENCY }, worker));

      const final = collected.filter(Boolean);
      setPapers(final);
      setLastLabel(activeLabel);
      setJSON(LS.cache, { papers: final, label: activeLabel, ts: Date.now() });
      setStatus("done"); setPhase("");
    } catch (err) {
      console.error(err);
      setErrorMsg((err.message || "Something went wrong") + ".");
      setStatus("error");
    }
  };

  const saveSettings = () => {
    try {
      if (keyInput.trim()) localStorage.setItem(LS.key, keyInput.trim());
      else localStorage.removeItem(LS.key);
      if (modelInput.trim()) localStorage.setItem(LS.model, modelInput.trim());
    } catch {}
    setShowSettings(false);
    flash("Settings saved");
  };

  // ── derived: which papers to show ──
  const sourcePapers = view === "saved" ? Object.values(bookmarks) : papers;
  const allTags = useMemo(() => {
    const s = new Set();
    sourcePapers.forEach(p => (p.tags || []).forEach(t => s.add(t)));
    return [...s].sort();
  }, [sourcePapers]);

  const shown = useMemo(() => {
    const q = query.trim().toLowerCase();
    return sourcePapers.filter(p => {
      if (minImpact && (p.impact || 0) < minImpact) return false;
      if (tagFilter && !(p.tags || []).includes(tagFilter)) return false;
      if (q) {
        const hay = `${p.title} ${p.abstract} ${p.tldr} ${(p.tags || []).join(" ")}`.toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });
  }, [sourcePapers, query, tagFilter, minImpact]);

  const exportMd = (download) => {
    const md = buildMarkdown(shown, view === "saved" ? "Bookmarks" : (lastLabel || activeLabel));
    if (download) {
      const blob = new Blob([md], { type: "text/markdown" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `ai-briefings-${new Date().toISOString().slice(0, 10)}.md`;
      a.click(); URL.revokeObjectURL(a.href);
      flash("Markdown downloaded");
    } else {
      navigator.clipboard?.writeText(md).then(() => flash("Copied as Markdown"), () => flash("Copy failed"));
    }
  };

  const pct = progress.total ? Math.round((progress.done / progress.total) * 100) : 0;
  const savedCount = Object.keys(bookmarks).length;

  return (
    <div className={`app ${theme === "dark" ? "dark" : ""}`}>
      <style>{CSS}</style>

      <div className="masthead">
        <div>
          <div className="kicker">arXiv + 🤗 · AI Research Digest</div>
          <div className="h1">Paper <em>Briefings</em></div>
        </div>
        <div className="mast-right">
          <div className="dateline">{TODAY}<br/>{count} papers / run</div>
          <div className="mast-tools">
            <button className="icon-btn" title="Toggle theme" onClick={() => setTheme(t => t === "dark" ? "light" : "dark")}>
              {theme === "dark" ? "☀" : "☾"}
            </button>
            <button className="icon-btn" title="Settings" onClick={() => setShowSettings(s => !s)}>⚙</button>
          </div>
        </div>
      </div>

      {showSettings && (
        <div className="settings">
          <div className="settings-grid">
            <label className="fld">
              <span className="ctl-label">OpenRouter API key</span>
              <input type="password" placeholder="sk-or-v1-…" value={keyInput} onChange={e => setKeyInput(e.target.value)} />
            </label>
            <label className="fld">
              <span className="ctl-label">Model</span>
              <input placeholder="anthropic/claude-3.5-sonnet" value={modelInput} onChange={e => setModelInput(e.target.value)} />
            </label>
          </div>
          <div className="settings-actions">
            <button className="run-btn" onClick={saveSettings}>Save</button>
            <span className="hint">Stored only in your browser (localStorage).</span>
          </div>
        </div>
      )}

      {/* topic chips */}
      <div className="topics">
        {PRESETS.map(p => (
          <button key={p.label}
            className={`chip ${selected.includes(p.label) ? "on" : ""}`}
            disabled={status === "loading"}
            onClick={() => toggleTopic(p.label)}>{p.label}</button>
        ))}
      </div>

      <div className="controls">
        <input className="custom-in" placeholder="+ custom topic…" value={customTopic}
          disabled={status === "loading"}
          onChange={e => setCustomTopic(e.target.value)}
          onKeyDown={e => e.key === "Enter" && status !== "loading" && run()} />

        <label className="toggle">
          <input type="checkbox" checked={useHF} disabled={status === "loading"}
            onChange={e => setUseHF(e.target.checked)} /> 🤗 HF Daily
        </label>

        <label className="toggle">
          <span className="ctl-label">Count</span>
          <select value={count} disabled={status === "loading"} onChange={e => setCount(+e.target.value)}>
            {COUNT_OPTIONS.map(c => <option key={c} value={c}>{c}</option>)}
          </select>
        </label>

        <button className="run-btn" onClick={run} disabled={status === "loading"}>
          {status === "loading" ? "Working…" : "↻ Analyze"}
        </button>
      </div>

      {status === "loading" && (
        <div className="progress-wrap">
          <div className="progress-top">
            <span className="progress-label">Summaries generated</span>
            <span className="progress-count">{progress.done}<small> / {progress.total || "…"}</small></span>
          </div>
          <div className="bar"><div className="bar-fill" style={{ width: `${pct}%` }} /></div>
          <div className="spin-row"><div className="mini-spin" /><span className="spin-text">{phase}</span></div>
        </div>
      )}

      {status === "error" && (
        <div className="error-box">
          <div>⚠ {errorMsg}</div>
          <button className="run-btn" onClick={run}>↻ Try Again</button>
        </div>
      )}

      {(papers.length > 0 || savedCount > 0) && (
        <>
          <div className="toolbar">
            <div className="tabs">
              <button className={`tab ${view === "all" ? "on" : ""}`} onClick={() => setView("all")}>
                Digest {papers.length ? `(${papers.length})` : ""}
              </button>
              <button className={`tab ${view === "saved" ? "on" : ""}`} onClick={() => setView("saved")}>
                ★ Saved ({savedCount})
              </button>
            </div>
            <div className="filters">
              <input className="search-in" placeholder="🔍 filter…" value={query} onChange={e => setQuery(e.target.value)} />
              <select value={minImpact} onChange={e => setMinImpact(+e.target.value)} title="Minimum impact">
                <option value={0}>impact: any</option>
                <option value={3}>impact ≥ 3</option>
                <option value={4}>impact ≥ 4</option>
                <option value={5}>impact = 5</option>
              </select>
              <button className="ghost-btn" onClick={() => exportMd(false)}>⧉ Copy MD</button>
              <button className="ghost-btn" onClick={() => exportMd(true)}>⭳ Export</button>
            </div>
          </div>

          {allTags.length > 0 && (
            <div className="tagbar">
              <button className={`tag ${!tagFilter ? "on" : ""}`} onClick={() => setTagFilter("")}>all</button>
              {allTags.map(t => (
                <button key={t} className={`tag ${tagFilter === t ? "on" : ""}`}
                  onClick={() => setTagFilter(f => f === t ? "" : t)}>#{t}</button>
              ))}
            </div>
          )}

          <div className="grid">
            <div className="count-banner">
              {shown.length} PAPER{shown.length !== 1 ? "S" : ""} · {(view === "saved" ? "BOOKMARKS" : (lastLabel || activeLabel)).toUpperCase()} · TAP A CARD FOR DETAIL
            </div>
            {shown.length === 0 && <div className="empty">Nothing matches your filters.</div>}
            {shown.map((p, i) => (
              <PaperCard key={(p.url || p.title) + i} paper={p} index={i}
                saved={isSaved(p)} onToggleSave={toggleSave} onTagClick={t => setTagFilter(t)} />
            ))}
          </div>
        </>
      )}

      {status === "idle" && papers.length === 0 && savedCount === 0 && (
        <div className="empty hero-empty">Pick topics above and hit <b>Analyze</b> to build today's briefing.</div>
      )}

      {toast && <div className="toast">{toast}</div>}
    </div>
  );
}

const CSS = `
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;1,9..144,400&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
* { box-sizing: border-box; margin: 0; padding: 0; }
.app {
  --bg:#faf7f0; --paper:#fffdf8; --ink:#1a1814; --sub:#6b6456;
  --line:#e5ddcd; --accent:#c0392b; --accent2:#2d6a4f; --hl:#f4efe2; --chip:#fffdf8;
  min-height:100vh; background:var(--bg); color:var(--ink);
  font-family:'Space Grotesk',sans-serif; padding-bottom:80px;
}
.app.dark {
  --bg:#16140f; --paper:#211e17; --ink:#ece6d8; --sub:#9a917f;
  --line:#36312611; --line:#3a342a; --accent:#e07a5f; --accent2:#81b29a; --hl:#27231b; --chip:#211e17;
}
.masthead { padding:36px 44px 24px; border-bottom:2px solid var(--ink);
  display:flex; justify-content:space-between; align-items:flex-end; gap:24px; }
.kicker { font-size:11px; letter-spacing:0.3em; text-transform:uppercase; color:var(--accent); margin-bottom:8px; font-weight:600; }
.h1 { font-family:'Fraunces',serif; font-size:42px; line-height:0.95; font-weight:600; letter-spacing:-0.02em; }
.h1 em { font-style:italic; color:var(--accent); }
.mast-right { display:flex; flex-direction:column; align-items:flex-end; gap:10px; }
.dateline { font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--sub); text-align:right; line-height:1.6; }
.mast-tools { display:flex; gap:8px; }
.icon-btn { width:34px; height:34px; border:1px solid var(--line); background:var(--paper); color:var(--ink);
  border-radius:4px; cursor:pointer; font-size:15px; transition:.15s; }
.icon-btn:hover { border-color:var(--ink); }

.settings { padding:20px 44px; border-bottom:1px solid var(--line); background:var(--hl); }
.settings-grid { display:flex; gap:16px; flex-wrap:wrap; }
.fld { display:flex; flex-direction:column; gap:6px; flex:1; min-width:240px; }
.settings-actions { margin-top:14px; display:flex; align-items:center; gap:14px; }
.hint { font-size:11px; color:var(--sub); }

.topics { padding:18px 44px 4px; display:flex; gap:8px; flex-wrap:wrap; }
.chip { font-family:'Space Grotesk',sans-serif; font-size:12px; font-weight:500; padding:6px 13px;
  border:1px solid var(--line); background:var(--chip); color:var(--sub); border-radius:99px; cursor:pointer; transition:.15s; }
.chip:hover { border-color:var(--ink); color:var(--ink); }
.chip.on { background:var(--ink); color:var(--bg); border-color:var(--ink); }
.chip:disabled { opacity:.5; cursor:not-allowed; }

.controls { padding:14px 44px 20px; display:flex; gap:14px; align-items:center; flex-wrap:wrap; border-bottom:1px solid var(--line); }
.ctl-label { font-size:11px; letter-spacing:0.15em; text-transform:uppercase; color:var(--sub); }
select, input { font-family:'Space Grotesk',sans-serif; font-size:13px; padding:9px 13px;
  border:1px solid var(--line); background:var(--paper); color:var(--ink); border-radius:2px; outline:none; }
select:focus, input:focus { border-color:var(--ink); }
.custom-in, .search-in { font-family:'JetBrains Mono',monospace; font-size:12px; width:190px; }
.custom-in::placeholder, .search-in::placeholder { color:var(--sub); }
.toggle { display:flex; align-items:center; gap:7px; font-size:13px; color:var(--ink); cursor:pointer; }
.toggle input[type=checkbox] { width:15px; height:15px; accent-color:var(--accent); padding:0; }
.run-btn { font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:12px; letter-spacing:0.08em;
  text-transform:uppercase; padding:10px 22px; background:var(--ink); color:var(--bg); border:none; cursor:pointer;
  border-radius:2px; transition:opacity .2s, transform .1s; }
.run-btn:hover { opacity:.85; } .run-btn:active { transform:scale(.97); }
.run-btn:disabled { opacity:.4; cursor:not-allowed; }

.progress-wrap { padding:28px 44px; }
.progress-top { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:12px; }
.progress-label { font-size:13px; color:var(--sub); }
.progress-count { font-family:'Fraunces',serif; font-size:28px; font-weight:600; }
.progress-count small { font-size:14px; color:var(--sub); }
.bar { height:4px; background:var(--line); border-radius:99px; overflow:hidden; }
.bar-fill { height:100%; background:var(--accent); border-radius:99px; transition:width .5s ease; }
.spin-row { display:flex; align-items:center; gap:10px; margin-top:16px; }
.mini-spin { width:14px; height:14px; border:2px solid var(--line); border-top-color:var(--accent);
  border-radius:50%; animation:spin .7s linear infinite; }
@keyframes spin { to { transform:rotate(360deg); } }
.spin-text { font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--sub); }

.toolbar { padding:18px 44px 10px; display:flex; justify-content:space-between; align-items:center; gap:16px; flex-wrap:wrap; }
.tabs { display:flex; gap:4px; }
.tab { font-size:12px; font-weight:600; padding:7px 14px; border:1px solid var(--line); background:var(--paper);
  color:var(--sub); border-radius:99px; cursor:pointer; }
.tab.on { background:var(--ink); color:var(--bg); border-color:var(--ink); }
.filters { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
.ghost-btn { font-size:12px; padding:8px 12px; border:1px solid var(--line); background:var(--paper);
  color:var(--ink); border-radius:2px; cursor:pointer; }
.ghost-btn:hover { border-color:var(--ink); }
.tagbar { padding:4px 44px 10px; display:flex; gap:6px; flex-wrap:wrap; }
.tag { font-family:'JetBrains Mono',monospace; font-size:10.5px; padding:3px 9px; border:1px solid var(--line);
  background:var(--paper); color:var(--sub); border-radius:99px; cursor:pointer; }
.tag:hover { border-color:var(--accent); color:var(--accent); }
.tag.on { background:var(--accent); color:#fff; border-color:var(--accent); }

.grid { padding:8px 44px 0; }
.count-banner { font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--sub);
  padding:12px 0; border-bottom:1px solid var(--line); margin-bottom:8px; letter-spacing:0.05em; }
.empty { padding:40px 0; text-align:center; color:var(--sub); font-size:14px; }
.hero-empty { padding:70px 44px; }
.card { background:var(--paper); border:1px solid var(--line); border-radius:4px; margin-bottom:12px;
  overflow:hidden; animation:rise .45s ease both; }
@keyframes rise { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:none; } }
.card-head { display:flex; align-items:flex-start; gap:16px; padding:18px 20px 12px; cursor:pointer; }
.card-num { font-family:'Fraunces',serif; font-size:20px; font-weight:600; color:var(--accent); min-width:32px; line-height:1.3; }
.card-head-main { flex:1; }
.card-title { font-family:'Fraunces',serif; font-size:18px; line-height:1.25; font-weight:600; margin-bottom:6px; }
.card-meta { font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--sub);
  display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
.dot { opacity:.5; }
.src-badge { font-size:9.5px; padding:1px 7px; border-radius:99px; border:1px solid var(--line); letter-spacing:.04em; }
.src-badge.hf { color:var(--accent2); border-color:var(--accent2); }
.src-badge.arxiv { color:var(--sub); }
.impact { display:inline-flex; gap:2px; align-items:center; }
.idot { width:7px; height:7px; border-radius:50%; background:var(--line); }
.idot.on { background:var(--accent); }
.tag-row { display:flex; gap:5px; flex-wrap:wrap; margin-top:8px; }
.save-btn { background:none; border:none; cursor:pointer; font-size:20px; color:var(--sub); line-height:1; padding:0 2px; }
.save-btn.on { color:var(--accent); }
.chev { font-size:22px; color:var(--sub); transition:transform .25s; line-height:1; user-select:none; }
.chev.up { transform:rotate(180deg); }
.card-tldr { padding:0 20px 16px 68px; font-size:14px; line-height:1.55; color:var(--ink); }
.card-body { padding:18px 20px 20px 68px; border-top:1px solid var(--line); background:var(--hl); animation:rise .3s ease both; }
.kp-label, .field-label { font-size:10px; letter-spacing:0.18em; text-transform:uppercase; color:var(--accent2);
  font-weight:600; margin-bottom:8px; display:block; }
.kp-list { list-style:none; display:flex; flex-direction:column; gap:7px; margin-bottom:16px; }
.kp-list li { font-size:13.5px; line-height:1.5; padding-left:18px; position:relative; color:var(--ink); }
.kp-list li::before { content:'—'; position:absolute; left:0; color:var(--accent); }
.field { margin-bottom:12px; }
.field-val { font-size:13px; line-height:1.5; color:var(--ink); display:block; }
.card-link { display:inline-block; margin-top:6px; font-family:'JetBrains Mono',monospace; font-size:12px;
  color:var(--accent); text-decoration:none; border-bottom:1px solid var(--accent); padding-bottom:1px; }
.error-box { margin:40px 44px; padding:24px; border:1px solid var(--accent); background:var(--hl); border-radius:4px; color:var(--accent); }
.error-box button { margin-top:14px; }
.toast { position:fixed; bottom:28px; left:50%; transform:translateX(-50%); background:var(--ink); color:var(--bg);
  font-size:13px; padding:11px 20px; border-radius:99px; box-shadow:0 6px 24px rgba(0,0,0,.18); animation:rise .25s ease both; z-index:50; }
@media (max-width:640px) {
  .masthead,.topics,.controls,.progress-wrap,.grid,.toolbar,.tagbar,.settings { padding-left:20px; padding-right:20px; }
  .h1 { font-size:30px; }
  .card-tldr,.card-body { padding-left:20px; }
}
`;
