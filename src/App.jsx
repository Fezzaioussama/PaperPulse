import { useState, useEffect, useMemo, useCallback, useRef } from "react";

// ──────────────────────────────────────────────────────────────────────────
// Topic presets → arXiv search queries. Categories (cat:cs.CL …) give far
// higher signal than raw keyword search, and we mix in keywords for the
// cross-cutting themes (agents, RAG, reasoning…) that aren't real categories.
// ──────────────────────────────────────────────────────────────────────────
const PRESETS = [
  {
    label: "LLMs",
    query: "cat:cs.CL",
    categories: ["cs.CL"],
    terms: ["large language model", "language model", "llm", "instruction tuning", "transformer"],
  },
  {
    label: "AI Agents",
    query: 'cat:cs.AI AND (all:agent OR all:agentic OR all:"LLM agent")',
    categories: ["cs.AI", "cs.CL"],
    terms: ["agent", "agentic", "llm agent", "tool use", "multi-agent", "planning", "memory"],
  },
  {
    label: "Reasoning",
    query: 'all:reasoning AND (all:"chain-of-thought" OR all:"large language model")',
    categories: ["cs.CL", "cs.AI"],
    terms: ["reasoning", "chain of thought", "chain-of-thought", "cot", "verifier", "math", "proof"],
  },
  {
    label: "RAG",
    query: 'all:"retrieval-augmented generation" OR all:RAG',
    categories: ["cs.CL", "cs.IR"],
    terms: ["retrieval augmented generation", "rag", "retrieval", "grounding", "knowledge base", "reranking"],
  },
  {
    label: "RLHF / Alignment",
    query: 'all:RLHF OR all:DPO OR all:"reinforcement learning from human feedback"',
    categories: ["cs.CL", "cs.AI", "cs.LG"],
    terms: ["rlhf", "dpo", "preference optimization", "alignment", "human feedback", "reward model"],
  },
  {
    label: "Multimodal",
    query: "cat:cs.CV AND all:multimodal",
    categories: ["cs.CV", "cs.CL"],
    terms: ["multimodal", "vision language", "vlm", "image text", "video language"],
  },
  {
    label: "Computer Vision",
    query: "cat:cs.CV",
    categories: ["cs.CV"],
    terms: ["computer vision", "image", "video", "object detection", "segmentation", "visual recognition"],
  },
  {
    label: "Diffusion",
    query: "cat:cs.CV AND all:diffusion",
    categories: ["cs.CV", "cs.LG"],
    terms: ["diffusion", "denoising", "score matching", "text to image", "image generation"],
  },
  {
    label: "Mech Interp",
    query: 'all:"mechanistic interpretability"',
    categories: ["cs.LG", "cs.CL"],
    terms: ["mechanistic interpretability", "interpretability", "circuit", "activation", "feature attribution", "representation"],
  },
  {
    label: "Efficiency / MoE",
    query: 'all:quantization OR all:"mixture of experts" OR all:MoE',
    categories: ["cs.LG", "cs.CL"],
    terms: ["quantization", "mixture of experts", "moe", "sparsity", "distillation", "inference efficiency"],
  },
  {
    label: "Robotics / VLA",
    query: "cat:cs.RO AND all:learning",
    categories: ["cs.RO"],
    terms: ["robotics", "vision language action", "vla", "embodied ai", "robot learning", "manipulation"],
  },
  {
    label: "AI Safety",
    query: 'all:"AI safety" OR all:"AI alignment"',
    categories: ["cs.AI", "cs.CL"],
    terms: ["ai safety", "alignment", "jailbreak", "red teaming", "robustness", "misuse", "risk"],
  },
  {
    label: "Benchmarks",
    query: "cat:cs.CL AND all:benchmark",
    categories: ["cs.CL", "cs.LG"],
    terms: ["benchmark", "evaluation", "dataset", "leaderboard", "test set", "task suite"],
  },
  {
    label: "Machine Learning",
    query: "cat:cs.LG",
    categories: ["cs.LG", "stat.ML"],
    terms: ["machine learning", "learning algorithm", "generalization", "optimization", "training"],
  },
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
const APP_NAME = "PaperPulse";
const APP_TAGLINE = "AI research briefings for fast paper triage.";

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

// ── chat-completions call taking a full message array (system/user/assistant) ──
async function chatCall(messages, maxTokens = 8000) {
  const apiKey = getApiKey();
  if (!apiKey) throw new Error("OpenRouter API key not found. Open Settings to add one, or set REACT_APP_OPENROUTER_API_KEY.");
  const model = getModel();
  if (!model) throw new Error("No OpenRouter model set. Open Settings to choose one.");

  const res = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${apiKey}`,
      "HTTP-Referer": "http://localhost",
      "X-Title": APP_NAME,
    },
    body: JSON.stringify({ model, max_tokens: maxTokens, messages }),
  });
  if (!res.ok) throw new Error(`API status ${res.status}`);
  const data = await res.json();
  if (data.error) throw new Error(data.error.message || "API error");
  let raw = "";
  for (const choice of data.choices || []) if (choice.message?.content) raw += choice.message.content + "\n";
  return raw;
}

// single-prompt convenience wrapper used by the per-paper summarizer
const apiCall = (prompt, maxTokens = 8000) =>
  chatCall([{ role: "user", content: prompt }], maxTokens);

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

// ── strip an HTML document down to readable body text ──
function htmlToText(html) {
  try {
    const doc = new DOMParser().parseFromString(html, "text/html");
    doc.querySelectorAll("script, style, nav, header, footer, .ltx_bibliography, .ltx_page_footer").forEach(el => el.remove());
    const main = doc.querySelector("article") || doc.querySelector("main") || doc.body;
    return (main?.textContent || "").replace(/\s+/g, " ").trim();
  } catch { return ""; }
}

// ── best-effort full-text fetch for an arXiv paper (HTML render, else ""). ──
// arXiv serves an HTML version for most recent papers; ar5iv is the fallback.
const FULLTEXT_LIMIT = 16000;
async function fetchPaperText(paper) {
  const id = normId(paper.url);
  if (!/^\d{4}\.\d{4,5}$/.test(id)) return "";
  const urls = [
    `https://arxiv.org/html/${id}`,
    `https://ar5iv.labs.arxiv.org/html/${id}`,
  ];
  for (const u of urls) {
    const html = await fetchWithProxies(u);
    if (!html || !/<\/?(article|main|body)/i.test(html)) continue;
    const text = htmlToText(html);
    if (text && text.length > 500) return text.slice(0, FULLTEXT_LIMIT);
  }
  return "";
}

const STOPWORDS = new Set([
  "the", "and", "for", "with", "from", "that", "this", "into", "over", "using",
  "based", "paper", "article", "research", "study", "approach", "method", "methods",
  "system", "systems", "model", "models", "new", "large", "learning",
]);

const QUALITY_SIGNALS = [
  "state of the art",
  "outperform",
  "benchmark",
  "dataset",
  "open source",
  "open weight",
  "scalable",
  "efficient",
  "evaluation",
  "framework",
  "generalization",
  "theory",
];

function normalizeForMatch(value = "") {
  return String(value)
    .toLowerCase()
    .replace(/['"`]/g, "")
    .replace(/[^a-z0-9+#.]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function uniqNormalized(values) {
  const seen = new Set();
  const out = [];
  for (const value of values) {
    const raw = String(value || "").trim();
    const key = normalizeForMatch(raw);
    if (!key || seen.has(key)) continue;
    seen.add(key);
    out.push(raw.toLowerCase());
  }
  return out;
}

function topicTokens(value = "") {
  return normalizeForMatch(value)
    .split(" ")
    .filter(t => t.length >= 3 && !STOPWORDS.has(t));
}

function getCustomTopicParts(customTopic = "") {
  const raw = customTopic.trim();
  if (!raw) return { phrases: [], tokens: [] };
  const chunks = raw.split(/[;,]/).map(part => part.trim()).filter(Boolean);
  const phraseSeeds = chunks.length > 1 ? chunks : [raw];
  return {
    phrases: uniqNormalized(phraseSeeds).filter(p => normalizeForMatch(p).length >= 3),
    tokens: uniqNormalized(topicTokens(raw)),
  };
}

function quoteArxivPhrase(value = "") {
  return String(value).trim().replace(/"/g, "");
}

function buildCustomQuery(customTopic = "") {
  const { phrases, tokens } = getCustomTopicParts(customTopic);
  const phraseQueries = phrases
    .filter(p => normalizeForMatch(p).includes(" "))
    .slice(0, 4)
    .map(p => `all:"${quoteArxivPhrase(p)}"`);
  const tokenQueries = tokens.slice(0, 6).map(t => `all:${t}`);
  const parts = [...new Set([...phraseQueries, ...tokenQueries])];
  if (parts.length === 0) return "";
  return parts.length === 1 ? parts[0] : `(${parts.join(" OR ")})`;
}

function buildTopicProfile(selectedLabels, customTopic) {
  const presets = selectedLabels
    .map(label => PRESETS.find(p => p.label === label))
    .filter(Boolean);
  const custom = getCustomTopicParts(customTopic);
  const presetTerms = presets.flatMap(p => p.terms || []);
  const labelTokens = presets.flatMap(p => topicTokens(p.label));
  const phrases = uniqNormalized([
    ...presetTerms.filter(t => normalizeForMatch(t).includes(" ")),
    ...custom.phrases.filter(t => normalizeForMatch(t).includes(" ")),
  ]);
  const tokens = uniqNormalized([
    ...presetTerms.flatMap(topicTokens),
    ...labelTokens,
    ...custom.tokens,
  ]);
  const categories = uniqNormalized(presets.flatMap(p => p.categories || []));
  return { phrases, tokens, categories };
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function tokenVariants(token) {
  const variants = [token];
  if (token.endsWith("ies") && token.length > 4) {
    variants.push(`${token.slice(0, -3)}y`);
  } else if (token.endsWith("s") && token.length > 4 && !token.endsWith("ss") && !token.endsWith("is")) {
    variants.push(token.slice(0, -1));
  } else if (token.length > 2) {
    variants.push(`${token}s`);
  }
  return variants;
}

function hasMatch(haystack, value) {
  const needle = normalizeForMatch(value);
  if (!needle) return false;
  if (needle.includes(" ")) return haystack.includes(needle);
  return tokenVariants(needle).some(token =>
    new RegExp(`(^|\\s)${escapeRegExp(token)}(?=\\s|$)`).test(haystack)
  );
}

function recencyScore(date) {
  if (!date) return 0;
  const ts = new Date(`${date}T00:00:00Z`).getTime();
  if (!Number.isFinite(ts)) return 0;
  const days = Math.max(0, (Date.now() - ts) / 86400000);
  if (days <= 7) return 5;
  if (days <= 30) return 3;
  if (days <= 90) return 1.5;
  return 0;
}

function scorePaperForTopic(paper, profile) {
  const title = normalizeForMatch(paper.title);
  const abstract = normalizeForMatch(paper.abstract);
  const combined = `${title} ${abstract}`;
  const paperCategories = (paper.categories || []).map(c => String(c).toLowerCase());
  const hasTopicProfile = profile.phrases.length > 0 || profile.tokens.length > 0 || profile.categories.length > 0;

  let topical = 0;
  let categoryMatches = 0;
  for (const category of profile.categories) {
    if (paperCategories.includes(category)) categoryMatches++;
  }
  topical += Math.min(14, categoryMatches * 8);

  let phraseMatches = 0;
  for (const phrase of profile.phrases) {
    if (hasMatch(title, phrase)) {
      topical += 18;
      phraseMatches++;
    } else if (hasMatch(abstract, phrase)) {
      topical += 8;
      phraseMatches++;
    }
  }

  let tokenScore = 0;
  let tokenMatches = 0;
  for (const token of profile.tokens) {
    if (hasMatch(title, token)) {
      tokenScore += 5;
      tokenMatches++;
    } else if (hasMatch(abstract, token)) {
      tokenScore += 2;
      tokenMatches++;
    }
  }
  topical += Math.min(32, tokenScore);
  if (profile.tokens.length > 0) {
    topical += Math.min(12, (tokenMatches / Math.min(profile.tokens.length, 8)) * 12);
  }

  const topicalHits = categoryMatches + phraseMatches + tokenMatches;
  let quality = recencyScore(paper.date);
  quality += paper.source === "hf" ? 1 : 0;
  quality += Math.min(8, Math.log2((paper.upvotes || 0) + 1) * 1.8);
  quality += Math.min(4, QUALITY_SIGNALS.filter(signal => hasMatch(combined, signal)).length);

  if (hasTopicProfile && topicalHits === 0) quality -= 20;
  return Math.round((topical + quality) * 10) / 10;
}

function comparePaperFallback(a, b) {
  return (b.upvotes || 0) - (a.upvotes || 0)
    || (b.date || "").localeCompare(a.date || "")
    || (a.title || "").localeCompare(b.title || "");
}

function rankPapersForTopic(papers, profile) {
  return papers
    .map(p => ({ ...p, relevanceScore: scorePaperForTopic(p, profile) }))
    .sort((a, b) => (b.relevanceScore || 0) - (a.relevanceScore || 0) || comparePaperFallback(a, b));
}

function finalPaperScore(paper) {
  const impact = Math.max(1, Math.min(5, paper.impact || 3));
  return (paper.relevanceScore || 0) + impact * 7 - (paper.abstractFallback ? 3 : 0);
}

function sortFinalPapers(papers) {
  return [...papers].sort((a, b) => finalPaperScore(b) - finalPaperScore(a) || comparePaperFallback(a, b));
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
    const categories = Array.from(e.getElementsByTagName("category"))
      .map(c => c.getAttribute("term"))
      .filter(Boolean);
    return {
      title: get("title"),
      authors: authorStr,
      date: published ? published.slice(0, 10) : "",
      url: absLink ? absLink.getAttribute("href") : get("id"),
      abstract: get("summary"),
      categories,
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
    const categories = (p.categories || p.tags || [])
      .map(c => typeof c === "string" ? c : (c.term || c.name || c.label))
      .filter(Boolean);
    return {
      title: (p.title || "").trim().replace(/\s+/g, " "),
      authors: authorStr,
      date: (it.publishedAt || p.publishedAt || "").slice(0, 10),
      url: id ? `https://arxiv.org/abs/${id}` : (p.url || ""),
      abstract: (p.summary || "").trim().replace(/\s+/g, " "),
      categories,
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
  const lines = [`# ${APP_NAME} - ${label}`, `_${APP_TAGLINE} ${TODAY}_`, ""];
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
    <span className="impact" title={`Impact ${n}/5`} aria-label={`Impact ${n} out of 5`}>
      {[1, 2, 3, 4, 5].map(i => (
        <span key={i} className={`idot ${i <= n ? "on" : ""}`} aria-hidden="true" />
      ))}
    </span>
  );
}

function PaperCard({ paper, index, saved, onToggleSave, onTagClick, onChat }) {
  const [open, setOpen] = useState(false);
  const toggleOpen = () => setOpen(o => !o);
  const onCardKeyDown = (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      toggleOpen();
    }
  };

  return (
    <article className="card" style={{ animationDelay: `${(index % 6) * 50}ms` }}>
      <div
        className="card-head"
        role="button"
        tabIndex={0}
        aria-expanded={open}
        onClick={toggleOpen}
        onKeyDown={onCardKeyDown}
      >
        <div className="card-num">{String(index + 1).padStart(2, "0")}</div>
        <div className="card-head-main">
          <div className="card-title">{paper.title}</div>
          <div className="card-meta">
            <span>{paper.authors}</span>
            {paper.date && <><span className="dot">·</span><span>{paper.date}</span></>}
            <span className={`src-badge ${paper.source}`}>
              {paper.source === "hf" ? `HF ${paper.upvotes || 0}` : "arXiv"}
            </span>
            <ImpactDots n={paper.impact} />
          </div>
          {paper.tags?.length > 0 && (
            <div className="tag-row">
              {paper.tags.map(t => (
                <button key={t} type="button" className="tag" onClick={e => { e.stopPropagation(); onTagClick(t); }}>#{t}</button>
              ))}
            </div>
          )}
        </div>
        <button
          type="button"
          className="chat-btn"
          title="Chat with this paper"
          aria-label={`Chat about ${paper.title}`}
          onClick={e => { e.stopPropagation(); onChat(paper); }}
        >💬</button>
        <button
          type="button"
          className={`save-btn ${saved ? "on" : ""}`}
          title={saved ? "Remove bookmark" : "Bookmark"}
          aria-label={saved ? `Remove bookmark for ${paper.title}` : `Bookmark ${paper.title}`}
          onClick={e => { e.stopPropagation(); onToggleSave(paper); }}
        >{saved ? "★" : "☆"}</button>
        <div className={`chev ${open ? "up" : ""}`} aria-hidden="true">⌄</div>
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
          <div className="card-actions">
            <button type="button" className="chat-cta" onClick={() => onChat(paper)}>
              💬 Chat with this paper
            </button>
            {paper.url && (
              <a className="card-link" href={paper.url} target="_blank" rel="noreferrer">↗ Open paper</a>
            )}
          </div>
        </div>
      )}
    </article>
  );
}

// ── assemble everything we know about a paper into chat-grounding context ──
function buildPaperContext(paper, fullText) {
  return [
    `Title: ${paper.title}`,
    `Authors: ${paper.authors}`,
    paper.date && `Published: ${paper.date}`,
    paper.categories?.length && `Categories: ${paper.categories.join(", ")}`,
    `Abstract: ${paper.abstract}`,
    paper.tldr && `TL;DR: ${paper.tldr}`,
    paper.key_points?.length && `Key points:\n- ${paper.key_points.join("\n- ")}`,
    paper.method && `Method: ${paper.method}`,
    paper.results && `Results: ${paper.results}`,
    fullText && `Full text (may be truncated):\n${fullText}`,
  ].filter(Boolean).join("\n\n");
}

const CHAT_SUGGESTIONS = [
  "Explain this paper like I'm new to the field",
  "What is the key contribution?",
  "Walk me through the method step by step",
  "What are the limitations or weaknesses?",
  "How does this compare to prior work?",
];

function ChatModal({ paper, onClose }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [fullText, setFullText] = useState("");
  const [reading, setReading] = useState(true);
  const bodyRef = useRef(null);
  const inputRef = useRef(null);

  // close on Escape
  useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  // ingest the paper's full text once; abstract + summary are the fallback context
  useEffect(() => {
    let cancelled = false;
    setReading(true);
    fetchPaperText(paper).then(text => {
      if (!cancelled) { setFullText(text); setReading(false); }
    });
    inputRef.current?.focus();
    return () => { cancelled = true; };
  }, [paper]);

  // autoscroll to the newest message
  useEffect(() => {
    bodyRef.current?.scrollTo({ top: bodyRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, sending]);

  const systemPrompt = useMemo(() => (
    "You are a knowledgeable research assistant helping a reader understand ONE specific arXiv paper through conversation. " +
    "Answer questions accurately and ground every claim in the paper content below. " +
    "If something isn't covered by the provided content, say so plainly instead of guessing. " +
    "Prefer clear, plain language and concrete examples; keep answers focused.\n\n" +
    `=== PAPER CONTENT ===\n${buildPaperContext(paper, fullText)}`
  ), [paper, fullText]);

  const send = useCallback(async (text) => {
    const content = (text ?? input).trim();
    if (!content || sending) return;
    const next = [...messages, { role: "user", content }];
    setMessages(next);
    setInput("");
    setSending(true);
    try {
      const reply = await chatCall([{ role: "system", content: systemPrompt }, ...next], 900);
      setMessages(m => [...m, { role: "assistant", content: reply.trim() || "(no response)" }]);
    } catch (e) {
      setMessages(m => [...m, { role: "assistant", content: "⚠️ " + (e.message || "Failed to get a response.") }]);
    } finally {
      setSending(false);
    }
  }, [input, sending, messages, systemPrompt]);

  return (
    <div
      className="chat-overlay"
      role="dialog"
      aria-modal="true"
      aria-label={`Chat about ${paper.title}`}
      onClick={onClose}
    >
      <div className="chat-modal" onClick={e => e.stopPropagation()}>
        <header className="chat-head">
          <div className="chat-head-main">
            <div className="chat-kicker">Chat with paper</div>
            <div className="chat-title">{paper.title}</div>
            <div className="chat-status">
              {reading
                ? <><span className="mini-spin" aria-hidden="true" /> Reading the paper…</>
                : fullText ? "Full text loaded — ask anything" : "Using abstract + summary as context"}
            </div>
          </div>
          <button type="button" className="chat-close" aria-label="Close chat" onClick={onClose}>✕</button>
        </header>

        <div className="chat-body" ref={bodyRef}>
          {messages.length === 0 && (
            <div className="chat-intro">
              <p className="chat-intro-lead">Ask anything about this paper. Try one of these:</p>
              <div className="chat-suggest">
                {CHAT_SUGGESTIONS.map(s => (
                  <button key={s} type="button" disabled={sending} onClick={() => send(s)}>{s}</button>
                ))}
              </div>
            </div>
          )}
          {messages.map((m, i) => (
            <div key={i} className={`chat-msg ${m.role}`}>
              <div className="chat-bubble">{m.content}</div>
            </div>
          ))}
          {sending && (
            <div className="chat-msg assistant">
              <div className="chat-bubble typing"><span /><span /><span /></div>
            </div>
          )}
        </div>

        <form className="chat-input-row" onSubmit={e => { e.preventDefault(); send(); }}>
          <input
            ref={inputRef}
            className="chat-input"
            placeholder="Ask about the method, results, limitations…"
            value={input}
            disabled={sending}
            onChange={e => setInput(e.target.value)}
          />
          <button type="submit" className="chat-send" disabled={sending || !input.trim()}>Send</button>
        </form>
      </div>
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
  const [chatPaper, setChatPaper] = useState(null);

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
    document.body.style.background = theme === "dark" ? "#111111" : "#f5f7fb";
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
    const customQuery = buildCustomQuery(customTopic);
    const topicProfile = buildTopicProfile(selected, customTopic);
    if (customQuery) queries.push(customQuery);
    if (queries.length === 0 && !useHF) { flash("Pick at least one topic or enable HF Daily"); return; }

    setStatus("loading"); setPapers([]); setErrorMsg(""); setView("all");
    setProgress({ done: 0, total: 0 });

    try {
      setPhase("Fetching papers from arXiv" + (useHF ? " + Hugging Face…" : "…"));
      const candidateTarget = Math.max(count * 3, count + 20);
      const perSource = Math.max(12, Math.ceil(candidateTarget / Math.max(1, queries.length)));
      const buckets = await Promise.all([
        ...queries.map(q => fetchArxiv(q, perSource)),
        ...(useHF ? [fetchHFDaily(candidateTarget)] : []),
      ]);

      let list = rankPapersForTopic(dedupe(buckets.flat()), topicProfile).slice(0, count);
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
          setPapers(sortFinalPapers(collected.filter(Boolean)));
        }
      };
      await Promise.all(Array.from({ length: CONCURRENCY }, worker));

      const final = sortFinalPapers(collected.filter(Boolean));
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
      a.download = `paperpulse-${new Date().toISOString().slice(0, 10)}.md`;
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

      <main className="app-shell">
        <header className="masthead">
          <div className="brand-block">
            <div className="kicker">AI research briefing workspace</div>
            <h1 className="brand-name">{APP_NAME}</h1>
            <p className="brand-copy">Triage fresh arXiv and Hugging Face papers by topic, impact, tags, and saved notes.</p>
          </div>
          <div className="mast-right">
            <div className="run-context" aria-label="Briefing context">
              <span>Today</span>
              <strong>{TODAY}</strong>
              <span>{count} papers per run</span>
              <span>{savedCount} saved</span>
            </div>
            <div className="mast-tools">
              <button
                type="button"
                className="icon-btn"
                title="Toggle theme"
                aria-label="Toggle theme"
                onClick={() => setTheme(t => t === "dark" ? "light" : "dark")}
              >
                {theme === "dark" ? "Light" : "Dark"}
              </button>
              <button
                type="button"
                className="icon-btn"
                title="Settings"
                aria-label="Settings"
                aria-expanded={showSettings}
                onClick={() => setShowSettings(s => !s)}
              >
                Settings
              </button>
            </div>
          </div>
        </header>

        {showSettings && (
          <section className="settings" aria-label="Settings">
            <div className="settings-grid">
              <label className="fld">
                <span className="ctl-label">OpenRouter API key</span>
                <input type="password" placeholder="sk-or-v1-..." value={keyInput} onChange={e => setKeyInput(e.target.value)} />
              </label>
              <label className="fld">
                <span className="ctl-label">Model</span>
                <input placeholder="anthropic/claude-3.5-sonnet" value={modelInput} onChange={e => setModelInput(e.target.value)} />
              </label>
            </div>
            <div className="settings-actions">
              <button type="button" className="run-btn" onClick={saveSettings}>Save Settings</button>
              <span className="hint">Stored only in this browser.</span>
            </div>
          </section>
        )}

        <section className="topic-panel" aria-label="Briefing setup">
          <div className="section-head">
            <div>
              <span className="section-label">Topics</span>
              <p className="section-copy">Choose presets, add a custom query, then run the briefing.</p>
            </div>
            <span className="active-brief">Current: {activeLabel}</span>
          </div>

          <div className="topics">
            {PRESETS.map(p => (
              <button
                key={p.label}
                type="button"
                className={`chip ${selected.includes(p.label) ? "on" : ""}`}
                disabled={status === "loading"}
                onClick={() => toggleTopic(p.label)}
              >
                {p.label}
              </button>
            ))}
          </div>

          <div className="controls">
            <label className="fld compact">
              <span className="ctl-label">Custom topic</span>
              <input
                className="custom-in"
                placeholder="agent memory, sparse attention..."
                value={customTopic}
                disabled={status === "loading"}
                onChange={e => setCustomTopic(e.target.value)}
                onKeyDown={e => e.key === "Enter" && status !== "loading" && run()}
              />
            </label>

            <label className="toggle">
              <input
                type="checkbox"
                checked={useHF}
                disabled={status === "loading"}
                onChange={e => setUseHF(e.target.checked)}
              />
              <span>HF Daily source</span>
            </label>

            <label className="fld count-field">
              <span className="ctl-label">Count</span>
              <select value={count} disabled={status === "loading"} onChange={e => setCount(+e.target.value)}>
                {COUNT_OPTIONS.map(c => <option key={c} value={c}>{c}</option>)}
              </select>
            </label>

            <button type="button" className="run-btn primary-action" onClick={run} disabled={status === "loading"}>
              {status === "loading" ? "Working..." : "Analyze"}
            </button>
          </div>
        </section>

        {status === "loading" && (
          <section className="progress-wrap" aria-label="Generation progress">
            <div className="progress-top">
              <span className="progress-label">Summaries generated</span>
              <span className="progress-count">{progress.done}<small> / {progress.total || "..."}</small></span>
            </div>
            <div className="bar" aria-hidden="true"><div className="bar-fill" style={{ width: `${pct}%` }} /></div>
            <div className="spin-row"><div className="mini-spin" aria-hidden="true" /><span className="spin-text">{phase}</span></div>
          </section>
        )}

        {status === "error" && (
          <section className="error-box" role="alert">
            <div>{errorMsg}</div>
            <button type="button" className="run-btn" onClick={run}>Try Again</button>
          </section>
        )}

        {(papers.length > 0 || savedCount > 0) && (
          <section className="results" aria-label="Briefing results">
            <div className="toolbar">
              <div className="tabs" role="tablist" aria-label="Result view">
                <button type="button" className={`tab ${view === "all" ? "on" : ""}`} onClick={() => setView("all")}>
                  Digest {papers.length ? `(${papers.length})` : ""}
                </button>
                <button type="button" className={`tab ${view === "saved" ? "on" : ""}`} onClick={() => setView("saved")}>
                  Saved ({savedCount})
                </button>
              </div>
              <div className="filters">
                <input
                  className="search-in"
                  aria-label="Filter papers"
                  placeholder="Filter papers..."
                  value={query}
                  onChange={e => setQuery(e.target.value)}
                />
                <select value={minImpact} onChange={e => setMinImpact(+e.target.value)} title="Minimum impact" aria-label="Minimum impact">
                  <option value={0}>Any impact</option>
                  <option value={3}>Impact 3+</option>
                  <option value={4}>Impact 4+</option>
                  <option value={5}>Impact 5</option>
                </select>
                <button type="button" className="ghost-btn" onClick={() => exportMd(false)}>Copy MD</button>
                <button type="button" className="ghost-btn" onClick={() => exportMd(true)}>Export MD</button>
              </div>
            </div>

            {allTags.length > 0 && (
              <div className="tagbar" aria-label="Filter by tag">
                <button type="button" className={`tag ${!tagFilter ? "on" : ""}`} onClick={() => setTagFilter("")}>all</button>
                {allTags.map(t => (
                  <button
                    key={t}
                    type="button"
                    className={`tag ${tagFilter === t ? "on" : ""}`}
                    onClick={() => setTagFilter(f => f === t ? "" : t)}
                  >
                    #{t}
                  </button>
                ))}
              </div>
            )}

            <div className="grid">
              <div className="count-banner">
                <strong>{shown.length}</strong> paper{shown.length !== 1 ? "s" : ""} shown
                <span>{view === "saved" ? "Bookmarks" : (lastLabel || activeLabel)}</span>
                <span>Open a card for method, results, and links</span>
              </div>
              {shown.length === 0 && <div className="empty">Nothing matches your filters.</div>}
              {shown.map((p, i) => (
                <PaperCard key={(p.url || p.title) + i} paper={p} index={i}
                  saved={isSaved(p)} onToggleSave={toggleSave} onTagClick={t => setTagFilter(t)}
                  onChat={setChatPaper} />
              ))}
            </div>
          </section>
        )}

        {status === "idle" && papers.length === 0 && savedCount === 0 && (
          <section className="empty hero-empty">
            <strong>No briefing yet.</strong>
            <span>Start with a topic preset, keep HF Daily on for community signal, and run Analyze.</span>
          </section>
        )}
      </main>

      {chatPaper && <ChatModal paper={chatPaper} onClose={() => setChatPaper(null)} />}
      {toast && <div className="toast">{toast}</div>}
    </div>
  );
}

const CSS = `
* { box-sizing: border-box; }
html, body, #root { width:100%; overflow-x:hidden; }
body { margin: 0; }
button, input, select { font: inherit; }
button { appearance: none; }
.app {
  --bg:#f5f7fb;
  --surface:#ffffff;
  --surface-2:#f9fbff;
  --surface-3:#eef4fb;
  --ink:#172033;
  --muted:#667085;
  --muted-2:#8793a4;
  --line:#d8e1ec;
  --line-strong:#b9c7d8;
  --accent:#1769aa;
  --accent-strong:#0f4f85;
  --accent-soft:#e5f2fb;
  --success:#1f7a55;
  --success-soft:#e6f6ee;
  --danger:#b42318;
  --danger-soft:#fff0ed;
  --focus:rgba(23,105,170,.2);
  --shadow:0 18px 46px rgba(27,44,67,.08);
  --radius:8px;
  --radius-sm:6px;
  --font-sans:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  --font-mono:"SFMono-Regular", Consolas, "Liberation Mono", monospace;
  min-height:100vh;
  background:var(--bg);
  color:var(--ink);
  font-family:var(--font-sans);
  overflow-x:hidden;
}
.app.dark {
  color-scheme:dark;
  --bg:#111111;
  --surface:#181818;
  --surface-2:#202020;
  --surface-3:#262626;
  --ink:#f4f6f8;
  --muted:#b8bec8;
  --muted-2:#8f98a6;
  --line:#333333;
  --line-strong:#4a4a4a;
  --accent:#70b8ff;
  --accent-strong:#acd8ff;
  --accent-soft:#172c3b;
  --success:#7bcfa6;
  --success-soft:#153324;
  --danger:#ff9c8f;
  --danger-soft:#331c19;
  --focus:rgba(112,184,255,.26);
  --shadow:0 18px 46px rgba(0,0,0,.32);
}
.app-shell {
  width:calc(100% - 40px);
  max-width:1180px;
  margin:0 auto;
  padding:30px 0 80px;
}
.masthead {
  display:grid;
  grid-template-columns:minmax(0, 1fr) auto;
  gap:24px;
  align-items:end;
  padding:24px 0 22px;
  border-bottom:1px solid var(--line);
}
.brand-block { max-width:720px; }
.kicker {
  color:var(--accent);
  font-size:13px;
  font-weight:800;
  text-transform:uppercase;
}
.brand-name {
  margin:8px 0 10px;
  font-size:56px;
  line-height:1;
  font-weight:850;
}
.brand-copy {
  max-width:640px;
  color:var(--muted);
  font-size:16px;
  line-height:1.55;
  overflow-wrap:anywhere;
}
.mast-right {
  display:flex;
  flex-direction:column;
  align-items:flex-end;
  gap:14px;
}
.run-context {
  display:grid;
  gap:3px;
  text-align:right;
  color:var(--muted);
  font-size:12px;
  line-height:1.35;
}
.run-context strong {
  color:var(--ink);
  font-size:14px;
  font-weight:750;
}
.mast-tools, .settings-actions, .filters, .tabs, .tagbar, .tag-row {
  display:flex;
  flex-wrap:wrap;
  align-items:center;
}
.mast-tools { gap:8px; justify-content:flex-end; }
.icon-btn, .ghost-btn, .tab, .chip, .tag, .run-btn {
  display:inline-flex;
  align-items:center;
  justify-content:center;
  border-radius:var(--radius-sm);
  border:1px solid var(--line);
  cursor:pointer;
  transition:background .16s ease, border-color .16s ease, color .16s ease, transform .1s ease, box-shadow .16s ease;
}
.icon-btn {
  min-height:38px;
  padding:8px 12px;
  background:var(--surface);
  color:var(--ink);
  font-size:13px;
  font-weight:700;
}
.icon-btn:hover, .ghost-btn:hover, .tab:hover, .chip:hover, .tag:hover {
  border-color:var(--accent);
  color:var(--accent);
}
button:focus-visible,
input:focus-visible,
select:focus-visible,
.card-head:focus-visible,
a:focus-visible {
  outline:3px solid var(--focus);
  outline-offset:2px;
}
.settings, .topic-panel, .progress-wrap, .error-box, .hero-empty {
  border:1px solid var(--line);
  border-radius:var(--radius);
  background:var(--surface);
  box-shadow:var(--shadow);
}
.settings {
  margin-top:18px;
  padding:20px;
}
.settings-grid {
  display:grid;
  grid-template-columns:repeat(2, minmax(0, 1fr));
  gap:16px;
}
.settings-actions {
  margin-top:16px;
  gap:12px;
}
.hint, .section-copy, .active-brief, .progress-label, .spin-text, .count-banner, .empty {
  color:var(--muted);
}
.hint { font-size:12px; }
.topic-panel {
  margin-top:18px;
  padding:20px;
}
.section-head {
  display:flex;
  justify-content:space-between;
  gap:20px;
  align-items:flex-start;
  margin-bottom:16px;
}
.section-label, .ctl-label, .kp-label, .field-label {
  display:block;
  color:var(--muted);
  font-size:12px;
  font-weight:800;
  text-transform:uppercase;
}
.section-copy {
  margin:4px 0 0;
  font-size:14px;
  overflow-wrap:anywhere;
}
.active-brief {
  max-width:420px;
  padding:7px 10px;
  border-radius:var(--radius-sm);
  background:var(--surface-3);
  font-size:12px;
  line-height:1.4;
  text-align:right;
  overflow-wrap:anywhere;
}
.topics {
  display:flex;
  flex-wrap:wrap;
  gap:8px;
  margin-bottom:18px;
}
.chip {
  max-width:100%;
  padding:8px 12px;
  background:var(--surface-2);
  color:var(--muted);
  font-size:13px;
  font-weight:700;
  white-space:normal;
}
.chip.on {
  background:var(--accent);
  border-color:var(--accent);
  color:#fff;
}
.chip:disabled, .run-btn:disabled {
  cursor:not-allowed;
  opacity:.52;
}
.controls {
  display:grid;
  grid-template-columns:minmax(240px, 1fr) auto 92px auto;
  gap:12px;
  align-items:end;
}
.controls > *, .filters > * { min-width:0; }
.fld {
  display:flex;
  flex-direction:column;
  gap:7px;
  min-width:0;
}
.count-field { min-width:92px; }
input, select {
  width:100%;
  min-height:42px;
  border:1px solid var(--line);
  border-radius:var(--radius-sm);
  background:var(--surface);
  color:var(--ink);
  padding:9px 12px;
  outline:none;
}
input::placeholder { color:var(--muted-2); }
input:disabled, select:disabled {
  cursor:not-allowed;
  opacity:.65;
}
.toggle {
  min-height:42px;
  display:flex;
  align-items:center;
  gap:9px;
  color:var(--ink);
  font-size:14px;
  font-weight:700;
  cursor:pointer;
}
.toggle input[type=checkbox] {
  width:18px;
  height:18px;
  min-height:18px;
  accent-color:var(--accent);
  padding:0;
}
.run-btn {
  min-height:42px;
  padding:10px 18px;
  background:var(--ink);
  border-color:var(--ink);
  color:var(--bg);
  font-size:13px;
  font-weight:800;
}
.primary-action {
  min-width:132px;
  background:var(--accent);
  border-color:var(--accent);
  color:#fff;
}
.run-btn:hover:not(:disabled) {
  transform:translateY(-1px);
  box-shadow:0 10px 22px rgba(23,105,170,.18);
}
.progress-wrap {
  margin-top:18px;
  padding:20px;
}
.progress-top {
  display:flex;
  justify-content:space-between;
  align-items:baseline;
  gap:16px;
  margin-bottom:12px;
}
.progress-count {
  font-size:30px;
  font-weight:850;
}
.progress-count small {
  color:var(--muted);
  font-size:14px;
  font-weight:650;
}
.bar {
  height:8px;
  overflow:hidden;
  border-radius:999px;
  background:var(--surface-3);
}
.bar-fill {
  height:100%;
  border-radius:999px;
  background:var(--accent);
  transition:width .45s ease;
}
.spin-row {
  display:flex;
  align-items:center;
  gap:10px;
  margin-top:14px;
}
.mini-spin {
  width:16px;
  height:16px;
  border:2px solid var(--line-strong);
  border-top-color:var(--accent);
  border-radius:50%;
  animation:spin .8s linear infinite;
}
@keyframes spin { to { transform:rotate(360deg); } }
.results { margin-top:20px; }
.toolbar {
  display:flex;
  justify-content:space-between;
  align-items:center;
  gap:14px;
  padding-bottom:12px;
  border-bottom:1px solid var(--line);
}
.tabs { gap:6px; }
.tab {
  padding:8px 13px;
  background:var(--surface);
  color:var(--muted);
  font-size:13px;
  font-weight:800;
}
.tab.on {
  background:var(--ink);
  border-color:var(--ink);
  color:var(--bg);
}
.filters {
  justify-content:flex-end;
  gap:8px;
}
.search-in { width:220px; }
.filters select { width:135px; }
.ghost-btn {
  min-height:38px;
  padding:8px 12px;
  background:var(--surface);
  color:var(--ink);
  font-size:13px;
  font-weight:750;
}
.tagbar {
  gap:6px;
  padding:14px 0 4px;
}
.tag {
  padding:5px 9px;
  background:var(--surface-2);
  color:var(--muted);
  font-family:var(--font-mono);
  font-size:11px;
}
.tag.on {
  background:var(--accent-soft);
  border-color:var(--accent);
  color:var(--accent-strong);
}
.grid { padding-top:12px; }
.count-banner {
  display:flex;
  flex-wrap:wrap;
  gap:8px 12px;
  align-items:center;
  margin-bottom:12px;
  padding:10px 0;
  border-bottom:1px solid var(--line);
  font-family:var(--font-mono);
  font-size:12px;
}
.count-banner strong { color:var(--ink); }
.count-banner span {
  padding-left:12px;
  border-left:1px solid var(--line);
}
.empty {
  padding:42px 20px;
  text-align:center;
  font-size:14px;
}
.hero-empty {
  display:grid;
  gap:6px;
  margin-top:18px;
  padding:42px 20px;
}
.hero-empty strong {
  color:var(--ink);
  font-size:17px;
}
.card {
  overflow:hidden;
  margin-bottom:12px;
  border:1px solid var(--line);
  border-radius:var(--radius);
  background:var(--surface);
  animation:rise .38s ease both;
}
@keyframes rise {
  from { opacity:0; transform:translateY(10px); }
  to { opacity:1; transform:none; }
}
.card-head {
  display:flex;
  align-items:flex-start;
  gap:14px;
  padding:18px 18px 12px;
  cursor:pointer;
}
.card-head:hover { background:var(--surface-2); }
.card-num {
  display:grid;
  place-items:center;
  min-width:40px;
  height:32px;
  border-radius:var(--radius-sm);
  background:var(--accent-soft);
  color:var(--accent-strong);
  font-family:var(--font-mono);
  font-size:13px;
  font-weight:800;
}
.card-head-main {
  min-width:0;
  flex:1;
}
.card-title {
  margin-bottom:7px;
  font-size:18px;
  line-height:1.35;
  font-weight:800;
  overflow-wrap:anywhere;
}
.card-meta {
  display:flex;
  flex-wrap:wrap;
  align-items:center;
  gap:7px;
  color:var(--muted);
  font-family:var(--font-mono);
  font-size:11px;
  line-height:1.45;
}
.dot { opacity:.45; }
.src-badge {
  padding:2px 7px;
  border:1px solid var(--line);
  border-radius:999px;
  font-size:10px;
  font-weight:800;
}
.src-badge.hf {
  border-color:var(--success);
  color:var(--success);
}
.src-badge.arxiv { color:var(--muted); }
.impact {
  display:inline-flex;
  gap:3px;
  align-items:center;
}
.idot {
  width:7px;
  height:7px;
  border-radius:50%;
  background:var(--line-strong);
}
.idot.on { background:var(--accent); }
.tag-row {
  gap:6px;
  margin-top:9px;
}
.save-btn {
  width:34px;
  min-width:34px;
  height:34px;
  border:1px solid transparent;
  border-radius:var(--radius-sm);
  background:transparent;
  color:var(--muted);
  cursor:pointer;
  font-size:22px;
  line-height:1;
}
.save-btn:hover {
  border-color:var(--line);
  background:var(--surface-3);
}
.save-btn.on { color:var(--accent); }
.chev {
  color:var(--muted);
  font-size:22px;
  line-height:1;
  transition:transform .22s ease;
  user-select:none;
}
.chev.up { transform:rotate(180deg); }
.card-tldr {
  padding:0 18px 18px 72px;
  color:var(--ink);
  font-size:14.5px;
  line-height:1.6;
  overflow-wrap:anywhere;
}
.card-body {
  padding:18px 18px 20px 72px;
  border-top:1px solid var(--line);
  background:var(--surface-2);
  animation:rise .28s ease both;
}
.kp-label, .field-label {
  margin-bottom:8px;
  color:var(--success);
}
.kp-list {
  display:flex;
  flex-direction:column;
  gap:8px;
  margin:0 0 16px;
  padding:0;
  list-style:none;
}
.kp-list li {
  position:relative;
  padding-left:18px;
  color:var(--ink);
  font-size:13.5px;
  line-height:1.55;
}
.kp-list li::before {
  content:'-';
  position:absolute;
  left:0;
  color:var(--accent);
  font-weight:900;
}
.field { margin-bottom:14px; }
.field-val {
  display:block;
  color:var(--ink);
  font-size:13.5px;
  line-height:1.55;
  overflow-wrap:anywhere;
}
.card-link {
  display:inline-flex;
  margin-top:4px;
  color:var(--accent-strong);
  font-size:13px;
  font-weight:800;
  text-decoration:none;
}
.card-link:hover { text-decoration:underline; }
.error-box {
  margin-top:18px;
  padding:20px;
  border-color:var(--danger);
  background:var(--danger-soft);
  color:var(--danger);
}
.error-box button { margin-top:14px; }
.toast {
  position:fixed;
  bottom:28px;
  left:50%;
  z-index:50;
  transform:translateX(-50%);
  max-width:calc(100% - 32px);
  padding:11px 18px;
  border-radius:999px;
  background:var(--ink);
  color:var(--bg);
  box-shadow:0 10px 28px rgba(0,0,0,.18);
  font-size:13px;
  animation:rise .22s ease both;
}
/* ── chat with paper ── */
.chat-btn {
  width:34px;
  min-width:34px;
  height:34px;
  border:1px solid transparent;
  border-radius:var(--radius-sm);
  background:transparent;
  color:var(--muted);
  cursor:pointer;
  font-size:16px;
  line-height:1;
}
.chat-btn:hover {
  border-color:var(--line);
  background:var(--surface-3);
  color:var(--accent);
}
.card-actions {
  display:flex;
  flex-wrap:wrap;
  align-items:center;
  gap:14px;
  margin-top:4px;
}
.chat-cta {
  display:inline-flex;
  align-items:center;
  gap:6px;
  padding:8px 14px;
  border-radius:var(--radius-sm);
  border:1px solid var(--accent);
  background:var(--accent-soft);
  color:var(--accent-strong);
  font-size:13px;
  font-weight:800;
  cursor:pointer;
  transition:transform .1s ease, box-shadow .16s ease;
}
.chat-cta:hover {
  transform:translateY(-1px);
  box-shadow:0 10px 22px rgba(23,105,170,.18);
}
.chat-overlay {
  position:fixed;
  inset:0;
  z-index:80;
  display:flex;
  align-items:center;
  justify-content:center;
  padding:20px;
  background:rgba(10,18,30,.55);
  backdrop-filter:blur(3px);
  animation:rise .18s ease both;
}
.chat-modal {
  display:flex;
  flex-direction:column;
  width:100%;
  max-width:720px;
  height:min(86vh, 760px);
  border:1px solid var(--line);
  border-radius:var(--radius);
  background:var(--surface);
  box-shadow:var(--shadow);
  overflow:hidden;
}
.chat-head {
  display:flex;
  align-items:flex-start;
  gap:14px;
  padding:18px 20px;
  border-bottom:1px solid var(--line);
  background:var(--surface-2);
}
.chat-head-main { flex:1; min-width:0; }
.chat-kicker {
  color:var(--accent);
  font-size:11px;
  font-weight:800;
  text-transform:uppercase;
  letter-spacing:.04em;
}
.chat-title {
  margin:4px 0 6px;
  font-size:16px;
  font-weight:800;
  line-height:1.35;
  overflow-wrap:anywhere;
}
.chat-status {
  display:flex;
  align-items:center;
  gap:8px;
  color:var(--muted);
  font-family:var(--font-mono);
  font-size:12px;
}
.chat-close {
  width:34px;
  min-width:34px;
  height:34px;
  border:1px solid var(--line);
  border-radius:var(--radius-sm);
  background:var(--surface);
  color:var(--muted);
  cursor:pointer;
  font-size:15px;
  line-height:1;
}
.chat-close:hover { border-color:var(--accent); color:var(--accent); }
.chat-body {
  flex:1;
  overflow-y:auto;
  display:flex;
  flex-direction:column;
  gap:14px;
  padding:18px 20px;
}
.chat-intro-lead {
  margin:0 0 12px;
  color:var(--muted);
  font-size:14px;
}
.chat-suggest {
  display:flex;
  flex-direction:column;
  align-items:flex-start;
  gap:8px;
}
.chat-suggest button {
  max-width:100%;
  text-align:left;
  padding:9px 13px;
  border-radius:var(--radius-sm);
  border:1px solid var(--line);
  background:var(--surface-2);
  color:var(--ink);
  font-size:13px;
  font-weight:600;
  cursor:pointer;
  transition:border-color .16s ease, color .16s ease;
}
.chat-suggest button:hover:not(:disabled) { border-color:var(--accent); color:var(--accent); }
.chat-suggest button:disabled { opacity:.5; cursor:not-allowed; }
.chat-msg { display:flex; }
.chat-msg.user { justify-content:flex-end; }
.chat-msg.assistant { justify-content:flex-start; }
.chat-bubble {
  max-width:80%;
  padding:11px 14px;
  border-radius:14px;
  font-size:14px;
  line-height:1.6;
  white-space:pre-wrap;
  overflow-wrap:anywhere;
}
.chat-msg.user .chat-bubble {
  background:var(--accent);
  color:#fff;
  border-bottom-right-radius:4px;
}
.chat-msg.assistant .chat-bubble {
  background:var(--surface-3);
  color:var(--ink);
  border:1px solid var(--line);
  border-bottom-left-radius:4px;
}
.chat-bubble.typing {
  display:flex;
  gap:5px;
  align-items:center;
}
.chat-bubble.typing span {
  width:7px;
  height:7px;
  border-radius:50%;
  background:var(--muted);
  animation:blink 1.2s infinite ease-in-out;
}
.chat-bubble.typing span:nth-child(2) { animation-delay:.2s; }
.chat-bubble.typing span:nth-child(3) { animation-delay:.4s; }
@keyframes blink { 0%, 80%, 100% { opacity:.25; } 40% { opacity:1; } }
.chat-input-row {
  display:flex;
  gap:10px;
  padding:14px 20px;
  border-top:1px solid var(--line);
  background:var(--surface-2);
}
.chat-input { flex:1; }
.chat-send {
  min-height:42px;
  padding:10px 20px;
  border-radius:var(--radius-sm);
  border:1px solid var(--accent);
  background:var(--accent);
  color:#fff;
  font-size:13px;
  font-weight:800;
  cursor:pointer;
}
.chat-send:disabled { opacity:.5; cursor:not-allowed; }

@media (max-width:640px) {
  .chat-overlay { padding:0; }
  .chat-modal {
    height:100vh;
    max-width:none;
    border:0;
    border-radius:0;
  }
  .chat-bubble { max-width:90%; }
}

@media (max-width:900px) {
  .masthead {
    grid-template-columns:1fr;
    align-items:start;
  }
  .mast-right {
    width:100%;
    align-items:flex-start;
  }
  .run-context { text-align:left; }
  .mast-tools { justify-content:flex-start; }
  .controls {
    grid-template-columns:1fr 1fr;
  }
  .primary-action { grid-column:1 / -1; }
  .toolbar {
    align-items:flex-start;
    flex-direction:column;
  }
  .filters { justify-content:flex-start; }
}
@media (max-width:640px) {
  .app-shell {
    width:calc(100% - 28px);
    padding-top:18px;
  }
  .brand-name { font-size:40px; }
  .brand-copy { font-size:15px; }
  .settings-grid, .controls {
    grid-template-columns:1fr;
  }
  .section-head {
    flex-direction:column;
  }
  .topics {
    display:grid;
    grid-template-columns:repeat(2, minmax(0, 1fr));
  }
  .chip { width:100%; }
  .active-brief {
    width:100%;
    max-width:none;
    text-align:left;
  }
  .filters, .tabs, .search-in, .filters select, .ghost-btn {
    width:100%;
  }
  .tab, .ghost-btn { justify-content:center; }
  .count-banner span {
    width:100%;
    padding-left:0;
    border-left:0;
  }
  .card-head {
    gap:10px;
    padding:16px 14px 10px;
  }
  .card-num {
    min-width:34px;
    height:30px;
  }
  .card-title { font-size:16px; }
  .card-tldr, .card-body {
    padding-left:14px;
    padding-right:14px;
  }
}
@media (prefers-reduced-motion:reduce) {
  *, *::before, *::after {
    animation-duration:.01ms !important;
    animation-iteration-count:1 !important;
    scroll-behavior:auto !important;
    transition-duration:.01ms !important;
  }
}
`;
