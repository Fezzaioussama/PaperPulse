import { useState, useEffect } from "react";

const TOPICS = [
  "Large Language Models",
  "Code generation",
  "Java / software migration",
  "Retrieval-Augmented Generation",
  "Reinforcement Learning",
  "Computer Vision",
  "Multimodal AI",
  "AI agents",
];

const TODAY = new Date().toLocaleDateString("en-US", {
  weekday: "long", year: "numeric", month: "long", day: "numeric",
});

const TARGET_COUNT = 20;
const BATCH_SIZE = 5; // papers summarized per API call

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

async function apiCall(prompt, useSearch, maxTokens = 8000) {
  const apiKey = process.env.REACT_APP_OPENROUTER_API_KEY || localStorage.getItem("openrouter_key");
  if (!apiKey) throw new Error("OpenRouter API key not found. Set REACT_APP_OPENROUTER_API_KEY env or store in localStorage.");
  
  const body = {
    model: "anthropic/claude-3.5-sonnet",
    max_tokens: maxTokens,
    messages: [{ role: "user", content: prompt }],
  };

  const res = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${apiKey}`,
      "HTTP-Referer": "http://localhost",
      "X-Title": "ArXiv Paper Briefings",
    },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`API status ${res.status}`);
  const data = await res.json();
  if (data.error) throw new Error(data.error.message || "API error");
  let raw = "";
  for (const choice of data.choices || []) if (choice.message?.content) raw += choice.message.content + "\n";
  return raw;
}

// Parse arXiv Atom XML into our paper objects.
function parseArxivXml(xml) {
  const doc = new DOMParser().parseFromString(xml, "application/xml");
  const entries = Array.from(doc.getElementsByTagName("entry"));
  return entries.map(e => {
    const get = (tag) => {
      const el = e.getElementsByTagName(tag)[0];
      return el ? el.textContent.trim().replace(/\s+/g, " ") : "";
    };
    const authors = Array.from(e.getElementsByTagName("author"))
      .map(a => a.getElementsByTagName("name")[0]?.textContent?.trim())
      .filter(Boolean);
    const authorStr = authors.length > 2 ? `${authors[0]} et al.`
      : authors.join(", ") || "—";
    const published = get("published");
    const date = published ? published.slice(0, 10) : "";
    let link = "";
    const links = Array.from(e.getElementsByTagName("link"));
    const absLink = links.find(l => l.getAttribute("rel") === "alternate");
    link = absLink ? absLink.getAttribute("href") : get("id");
    return {
      title: get("title"),
      authors: authorStr,
      date,
      url: link,
      abstract: get("summary"),
    };
  }).filter(p => p.title && p.abstract);
}

// Last-resort discovery: ask Claude (with web_search) to fetch the arXiv
// listing + abstracts. Slower but always reachable since the Anthropic API
// is the one network call this artifact is guaranteed to make.
async function fetchArxivViaClaude(topic, count) {
  const prompt = `Use web search to find the ${count} most recent arXiv papers about "${topic}".
For EACH paper, get the title, first author "et al.", publication date (YYYY-MM-DD), arXiv URL, and the FULL ABSTRACT text from the paper's arXiv page.

Respond with ONLY a JSON array. No prose, no markdown fences. Schema:
[{"title":"...", "authors":"X et al.", "date":"YYYY-MM-DD", "url":"https://arxiv.org/abs/...", "abstract":"the full abstract text"}]

Return as many as you can up to ${count}.`;
  const raw = await apiCall(prompt, true, 8000);
  const arr = extractJson(raw);
  if (!Array.isArray(arr)) throw new Error("Claude fallback returned non-array");
  return arr
    .filter(p => p && p.title && p.abstract)
    .slice(0, count)
    .map(p => ({
      title: p.title,
      authors: p.authors || "—",
      date: p.date || "",
      url: p.url || "",
      abstract: p.abstract,
    }));
}

// Robust arXiv fetcher: tries the API directly, then two CORS-proxy mirrors,
// and finally falls back to the Anthropic API + web search.
async function fetchArxiv(topic, count) {
  const apiUrl = `https://export.arxiv.org/api/query?search_query=${encodeURIComponent("all:" + topic)}&sortBy=submittedDate&sortOrder=descending&max_results=${count}`;
  const tries = [
    { label: "direct", url: apiUrl },
    { label: "corsproxy.io", url: `https://corsproxy.io/?${encodeURIComponent(apiUrl)}` },
    { label: "allorigins", url: `https://api.allorigins.win/raw?url=${encodeURIComponent(apiUrl)}` },
  ];
  for (const t of tries) {
    try {
      const res = await fetch(t.url);
      if (!res.ok) continue;
      const xml = await res.text();
      if (xml && xml.includes("<entry")) {
        const parsed = parseArxivXml(xml);
        if (parsed.length > 0) return parsed;
      }
    } catch (e) {
      // try the next source
    }
  }
  // All XML sources failed — fall back to Claude
  return await fetchArxivViaClaude(topic, count);
}

function PaperCard({ paper, index }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="card" style={{ animationDelay: `${(index % 5) * 60}ms` }}>
      <div className="card-head" onClick={() => setOpen(o => !o)}>
        <div className="card-num">{String(index + 1).padStart(2, "0")}</div>
        <div className="card-head-main">
          <div className="card-title">{paper.title}</div>
          <div className="card-meta">
            <span>{paper.authors}</span>
            {paper.date && <span className="dot">·</span>}
            {paper.date && <span>{paper.date}</span>}
          </div>
        </div>
        <div className={`chev ${open ? "up" : ""}`}>⌄</div>
      </div>

      <div className="card-tldr">{paper.tldr}</div>

      {open && (
        <div className="card-body">
          {paper.key_points && paper.key_points.length > 0 && (
            <div className="kp-block">
              <div className="kp-label">Key Points</div>
              <ul className="kp-list">
                {paper.key_points.map((k, i) => <li key={i}>{k}</li>)}
              </ul>
            </div>
          )}
          {paper.method && (
            <div className="field">
              <span className="field-label">Method</span>
              <span className="field-val">{paper.method}</span>
            </div>
          )}
          {paper.results && (
            <div className="field">
              <span className="field-label">Results</span>
              <span className="field-val">{paper.results}</span>
            </div>
          )}
          {paper.abstractFallback && paper.abstract && (
            <div className="field">
              <span className="field-label">Abstract</span>
              <span className="field-val">{paper.abstract}</span>
            </div>
          )}
          {paper.url && (
            <a className="card-link" href={paper.url} target="_blank" rel="noreferrer">
              ↗ Open paper
            </a>
          )}
        </div>
      )}
    </div>
  );
}

export default function ArxivBatchAnalyzer() {
  const [topic, setTopic] = useState(TOPICS[0]);
  const [customTopic, setCustomTopic] = useState("");
  const [status, setStatus] = useState("idle"); // idle | loading | done | error
  const [papers, setPapers] = useState([]);
  const [progress, setProgress] = useState({ done: 0, total: TARGET_COUNT });
  const [phase, setPhase] = useState("");
  const [errorMsg, setErrorMsg] = useState("");

  const activeTopic = customTopic.trim() || topic;

  const run = async () => {
    setStatus("loading");
    setPapers([]);
    setErrorMsg("");
    setProgress({ done: 0, total: TARGET_COUNT });

    try {
      // ── PHASE 1: pull real papers + abstracts straight from the arXiv API ──
      // This removes all LLM/search flakiness from discovery: we get the actual
      // title, authors, date AND abstract for every paper.
      setPhase("Fetching recent papers (trying arXiv, proxies, then Claude)…");
      const list = await fetchArxiv(activeTopic, TARGET_COUNT);
      if (list.length === 0) throw new Error("No papers found on arXiv for this topic");

      setProgress({ done: 0, total: list.length });

      // ── PHASE 2: summarize ONE paper per call from its real abstract ──
      // One small, search-free request per paper => clean JSON that reliably
      // parses. Runs a few in parallel for speed, with per-paper retry.
      setPhase("Summarizing each paper…");
      const collected = new Array(list.length).fill(null);
      let completed = 0;

      const summarizeOne = async (orig) => {
        const sumPrompt = `You are summarizing a single arXiv paper. Here is its real metadata and abstract:

Title: ${orig.title}
Authors: ${orig.authors}
Abstract: ${orig.abstract}

Write a detailed summary. Respond with ONLY a JSON object (no markdown fences, no text before or after):
{
  "tldr": "one punchy sentence capturing the core idea",
  "key_points": ["point 1", "point 2", "point 3", "point 4"],
  "method": "the approach/technique in 1-2 sentences",
  "results": "main findings or claimed improvements in 1 sentence"
}`;
        // up to 3 tries per paper
        for (let attempt = 1; attempt <= 3; attempt++) {
          try {
            const raw = await apiCall(sumPrompt, false, 900);
            const s = extractJson(raw);
            if (s && (s.tldr || s.key_points)) {
              return {
                ...orig,
                tldr: s.tldr || orig.abstract.slice(0, 160) + "…",
                key_points: Array.isArray(s.key_points) ? s.key_points : [],
                method: s.method || "",
                results: s.results || "",
              };
            }
          } catch (_) {}
          await new Promise(r => setTimeout(r, 800));
        }
        // graceful fallback: use the real abstract so the card is never empty
        return {
          ...orig,
          tldr: orig.abstract.slice(0, 180) + (orig.abstract.length > 180 ? "…" : ""),
          key_points: [],
          method: "",
          results: "",
          abstractFallback: true,
        };
      };

      // process with limited concurrency (3 at a time)
      const CONCURRENCY = 3;
      let cursor = 0;
      const worker = async () => {
        while (cursor < list.length) {
          const myIdx = cursor++;
          const summary = await summarizeOne(list[myIdx]);
          collected[myIdx] = summary;
          completed++;
          setProgress({ done: completed, total: list.length });
          setPapers(collected.filter(Boolean));
        }
      };
      await Promise.all(Array.from({ length: CONCURRENCY }, worker));

      setPapers(collected.filter(Boolean));
      setStatus("done");
      setPhase("");
    } catch (err) {
      console.error(err);
      setErrorMsg((err.message || "Something went wrong") + ".");
      setStatus("error");
    }
  };

  useEffect(() => { run(); /* eslint-disable-next-line */ }, []);

  const pct = Math.round((progress.done / progress.total) * 100);

  return (
    <div className="app">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;1,9..144,400&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        :root {
          --bg: #faf7f0; --paper: #fffdf8; --ink: #1a1814; --sub: #6b6456;
          --line: #e5ddcd; --accent: #c0392b; --accent2: #2d6a4f;
          --hl: #f4efe2;
        }
        body { background: var(--bg); }
        .app {
          min-height: 100vh; background: var(--bg); color: var(--ink);
          font-family: 'Space Grotesk', sans-serif; padding-bottom: 60px;
        }
        .masthead {
          padding: 36px 44px 24px; border-bottom: 2px solid var(--ink);
          display: flex; justify-content: space-between; align-items: flex-end; gap: 24px;
        }
        .kicker {
          font-size: 11px; letter-spacing: 0.3em; text-transform: uppercase;
          color: var(--accent); margin-bottom: 8px; font-weight: 600;
        }
        .h1 {
          font-family: 'Fraunces', serif; font-size: 42px; line-height: 0.95;
          font-weight: 600; letter-spacing: -0.02em;
        }
        .h1 em { font-style: italic; color: var(--accent); }
        .dateline {
          font-family: 'JetBrains Mono', monospace; font-size: 11px;
          color: var(--sub); text-align: right; line-height: 1.6;
        }
        .controls {
          padding: 20px 44px; display: flex; gap: 12px; align-items: center;
          flex-wrap: wrap; border-bottom: 1px solid var(--line);
        }
        .ctl-label {
          font-size: 11px; letter-spacing: 0.15em; text-transform: uppercase; color: var(--sub);
        }
        select, input {
          font-family: 'Space Grotesk', sans-serif; font-size: 13px;
          padding: 9px 13px; border: 1px solid var(--ink); background: var(--paper);
          color: var(--ink); border-radius: 2px; outline: none;
        }
        input { font-family: 'JetBrains Mono', monospace; font-size: 12px; width: 200px; }
        input::placeholder { color: var(--sub); }
        .run-btn {
          font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 12px;
          letter-spacing: 0.08em; text-transform: uppercase; padding: 10px 22px;
          background: var(--ink); color: var(--bg); border: none; cursor: pointer;
          border-radius: 2px; transition: opacity .2s, transform .1s;
        }
        .run-btn:hover { opacity: .85; } .run-btn:active { transform: scale(.97); }
        .run-btn:disabled { opacity: .4; cursor: not-allowed; }

        .progress-wrap { padding: 28px 44px; }
        .progress-top {
          display: flex; justify-content: space-between; align-items: baseline;
          margin-bottom: 12px;
        }
        .progress-label { font-size: 13px; color: var(--sub); }
        .progress-count {
          font-family: 'Fraunces', serif; font-size: 28px; font-weight: 600;
        }
        .progress-count small { font-size: 14px; color: var(--sub); }
        .bar { height: 4px; background: var(--line); border-radius: 99px; overflow: hidden; }
        .bar-fill {
          height: 100%; background: var(--accent); border-radius: 99px;
          transition: width .5s ease;
        }
        .spin-row { display: flex; align-items: center; gap: 10px; margin-top: 16px; }
        .mini-spin {
          width: 14px; height: 14px; border: 2px solid var(--line);
          border-top-color: var(--accent); border-radius: 50%;
          animation: spin .7s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .spin-text {
          font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--sub);
        }

        .grid { padding: 8px 44px 0; }
        .count-banner {
          font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--sub);
          padding: 12px 0; border-bottom: 1px solid var(--line); margin-bottom: 8px;
          letter-spacing: 0.05em;
        }
        .card {
          background: var(--paper); border: 1px solid var(--line); border-radius: 4px;
          margin-bottom: 12px; overflow: hidden;
          animation: rise .45s ease both;
        }
        @keyframes rise { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: none; } }
        .card-head {
          display: flex; align-items: flex-start; gap: 16px; padding: 18px 20px 12px;
          cursor: pointer;
        }
        .card-num {
          font-family: 'Fraunces', serif; font-size: 20px; font-weight: 600;
          color: var(--accent); min-width: 32px; line-height: 1.3;
        }
        .card-head-main { flex: 1; }
        .card-title {
          font-family: 'Fraunces', serif; font-size: 18px; line-height: 1.25;
          font-weight: 600; margin-bottom: 6px;
        }
        .card-meta {
          font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--sub);
          display: flex; gap: 6px; flex-wrap: wrap;
        }
        .dot { opacity: .5; }
        .chev {
          font-size: 22px; color: var(--sub); transition: transform .25s; line-height: 1;
          user-select: none;
        }
        .chev.up { transform: rotate(180deg); }
        .card-tldr {
          padding: 0 20px 16px 68px; font-size: 14px; line-height: 1.55; color: var(--ink);
        }
        .card-body {
          padding: 18px 20px 20px 68px; border-top: 1px solid var(--line);
          background: var(--hl);
          animation: rise .3s ease both;
        }
        .kp-label, .field-label {
          font-size: 10px; letter-spacing: 0.18em; text-transform: uppercase;
          color: var(--accent2); font-weight: 600; margin-bottom: 8px; display: block;
        }
        .kp-list { list-style: none; display: flex; flex-direction: column; gap: 7px; margin-bottom: 16px; }
        .kp-list li {
          font-size: 13.5px; line-height: 1.5; padding-left: 18px; position: relative; color: var(--ink);
        }
        .kp-list li::before { content: '—'; position: absolute; left: 0; color: var(--accent); }
        .field { margin-bottom: 12px; }
        .field-val { font-size: 13px; line-height: 1.5; color: var(--ink); display: block; }
        .card-link {
          display: inline-block; margin-top: 6px; font-family: 'JetBrains Mono', monospace;
          font-size: 12px; color: var(--accent); text-decoration: none;
          border-bottom: 1px solid var(--accent); padding-bottom: 1px;
        }
        .error-box {
          margin: 40px 44px; padding: 24px; border: 1px solid var(--accent);
          background: #fdf0ee; border-radius: 4px; color: var(--accent);
        }
        .error-box button { margin-top: 14px; }
        @media (max-width: 640px) {
          .masthead, .controls, .progress-wrap, .grid { padding-left: 20px; padding-right: 20px; }
          .h1 { font-size: 30px; }
          .card-tldr, .card-body { padding-left: 20px; }
        }
      `}</style>

      <div className="masthead">
        <div>
          <div className="kicker">arXiv Digest · Batch Analyzer</div>
          <div className="h1">Paper <em>Briefings</em></div>
        </div>
        <div className="dateline">{TODAY}<br/>{TARGET_COUNT} papers / run</div>
      </div>

      <div className="controls">
        <span className="ctl-label">Topic</span>
        <select value={topic} onChange={e => setTopic(e.target.value)} disabled={status === "loading"}>
          {TOPICS.map(t => <option key={t}>{t}</option>)}
        </select>
        <input
          placeholder="or custom topic…"
          value={customTopic}
          onChange={e => setCustomTopic(e.target.value)}
          onKeyDown={e => e.key === "Enter" && status !== "loading" && run()}
          disabled={status === "loading"}
        />
        <button className="run-btn" onClick={run} disabled={status === "loading"}>
          {status === "loading" ? "Working…" : "↻ Analyze"}
        </button>
      </div>

      {status === "loading" && (
        <div className="progress-wrap">
          <div className="progress-top">
            <span className="progress-label">Summaries generated</span>
            <span className="progress-count">{progress.done}<small> / {progress.total}</small></span>
          </div>
          <div className="bar"><div className="bar-fill" style={{ width: `${pct}%` }} /></div>
          <div className="spin-row">
            <div className="mini-spin" />
            <span className="spin-text">{phase}</span>
          </div>
        </div>
      )}

      {status === "error" && (
        <div className="error-box">
          <div>⚠ {errorMsg}</div>
          <button className="run-btn" onClick={run}>↻ Try Again</button>
        </div>
      )}

      {papers.length > 0 && (
        <div className="grid">
          <div className="count-banner">
            {papers.length} PAPER{papers.length !== 1 ? "S" : ""} ON “{activeTopic.toUpperCase()}” · TAP A CARD FOR KEY POINTS
          </div>
          {papers.map((p, i) => <PaperCard key={i} paper={p} index={i} />)}
        </div>
      )}
    </div>
  );
}
