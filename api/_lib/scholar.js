import { fetchArxiv } from "./arxiv.js";
import { compactText, clipText, normalizeForMatch, normId } from "./text.js";

// ── Multi-source scholarly search: arXiv + Semantic Scholar + OpenAlex.
//    All three are real public APIs; each source degrades to [] on failure
//    (rate limits, downtime) so a literature search never dies with one
//    provider. Results are normalized to the app's paper shape and deduped
//    across sources by arXiv id → DOI → title. ──

const S2_BASE = "https://api.semanticscholar.org/graph/v1";
const OPENALEX_BASE = "https://api.openalex.org";
const UA = "PaperPulse/1.0 (research agent; https://paperpulse.vercel.app)";
const TIMEOUT = 12000;

const authorsLabel = (names) => {
  const list = names.filter(Boolean);
  if (!list.length) return "Unknown";
  return list.length > 2 ? `${list[0]} et al.` : list.join(", ");
};

// OpenAlex ships abstracts as an inverted index {word: [positions]}.
export function invertAbstract(inv) {
  if (!inv || typeof inv !== "object") return "";
  const words = [];
  for (const [word, positions] of Object.entries(inv)) {
    if (!Array.isArray(positions)) continue;
    for (const pos of positions) {
      if (Number.isInteger(pos) && pos >= 0 && pos < 5000) words[pos] = word;
    }
  }
  return compactText(words.filter(Boolean).join(" "));
}

export function arxivIdFromDoi(doi = "") {
  const m = String(doi).toLowerCase().match(/10\.48550\/arxiv\.(\d{4}\.\d{4,5})/);
  return m ? m[1] : "";
}

// Normalize one Semantic Scholar record to the app's paper shape.
export function fromSemanticScholar(rec) {
  if (!rec || typeof rec.title !== "string" || !rec.title.trim()) return null;
  const arxivId = rec.externalIds?.ArXiv || "";
  const doi = (rec.externalIds?.DOI || "").toLowerCase();
  return {
    title: compactText(rec.title),
    authors: authorsLabel((rec.authors || []).map((a) => a?.name)),
    date: rec.publicationDate || (rec.year ? `${rec.year}` : ""),
    url: arxivId ? `https://arxiv.org/abs/${arxivId}` : (rec.url || (doi ? `https://doi.org/${doi}` : "")),
    abstract: clipText(compactText(rec.abstract || ""), 2000),
    categories: [],
    source: "semanticscholar",
    doi,
    citations: Number.isInteger(rec.citationCount) ? rec.citationCount : null,
  };
}

// Normalize one OpenAlex work to the app's paper shape.
export function fromOpenAlex(rec) {
  const title = rec?.display_name || rec?.title;
  if (!rec || typeof title !== "string" || !title.trim()) return null;
  const doi = String(rec.doi || "").replace(/^https?:\/\/doi\.org\//i, "").toLowerCase();
  const arxivId = arxivIdFromDoi(doi);
  return {
    title: compactText(title),
    authors: authorsLabel((rec.authorships || []).map((a) => a?.author?.display_name)),
    date: rec.publication_date || "",
    url: arxivId
      ? `https://arxiv.org/abs/${arxivId}`
      : (rec.open_access?.oa_url || (doi ? `https://doi.org/${doi}` : String(rec.id || ""))),
    abstract: clipText(invertAbstract(rec.abstract_inverted_index), 2000),
    categories: [],
    source: "openalex",
    doi,
    citations: Number.isInteger(rec.cited_by_count) ? rec.cited_by_count : null,
  };
}

async function getJson(url, headers = {}) {
  const res = await fetch(url, {
    headers: { "User-Agent": UA, ...headers },
    signal: AbortSignal.timeout(TIMEOUT),
  });
  if (!res.ok) throw new Error(`status ${res.status}`);
  return res.json();
}

// Semantic Scholar paper search. The anonymous pool is aggressively
// rate-limited (429); an optional S2_API_KEY env raises the limits.
export async function fetchSemanticScholar(query, count = 10) {
  try {
    const fields = "title,abstract,year,publicationDate,citationCount,externalIds,url,authors";
    const url = `${S2_BASE}/paper/search?query=${encodeURIComponent(query)}&limit=${Math.min(count, 20)}&fields=${fields}`;
    const key = (process.env.S2_API_KEY || "").trim();
    const data = await getJson(url, key ? { "x-api-key": key } : {});
    return (Array.isArray(data?.data) ? data.data : []).map(fromSemanticScholar).filter(Boolean);
  } catch (_) {
    return [];
  }
}

export async function fetchOpenAlex(query, count = 10) {
  try {
    const select = "id,doi,title,display_name,publication_date,cited_by_count,authorships,open_access,abstract_inverted_index";
    const url = `${OPENALEX_BASE}/works?search=${encodeURIComponent(query)}&per-page=${Math.min(count, 25)}&select=${select}`;
    const data = await getJson(url);
    return (Array.isArray(data?.results) ? data.results : []).map(fromOpenAlex).filter(Boolean);
  } catch (_) {
    return [];
  }
}

// Cross-source identity: arXiv id → DOI → normalized title.
export function scholarKey(p) {
  const fromUrl = String(p.url || "").match(/(\d{4}\.\d{4,5})(v\d+)?/);
  if (fromUrl) return `arxiv:${fromUrl[1]}`;
  if (p.doi) return `doi:${p.doi}`;
  return `title:${normalizeForMatch(p.title).slice(0, 90)}`;
}

// Merge papers from several sources: first occurrence wins, later duplicates
// fill in missing fields (abstract, citations, doi) rather than being lost.
export function mergeScholarly(lists) {
  const byKey = new Map();
  for (const p of lists.flat()) {
    if (!p) continue;
    const key = scholarKey(p);
    const prev = byKey.get(key);
    if (!prev) {
      byKey.set(key, { ...p });
      continue;
    }
    if (!prev.abstract && p.abstract) prev.abstract = p.abstract;
    if (!prev.doi && p.doi) prev.doi = p.doi;
    if (!prev.date && p.date) prev.date = p.date;
    if (p.citations != null && (prev.citations == null || p.citations > prev.citations)) {
      prev.citations = p.citations;
    }
    if (!/arxiv\.org/.test(prev.url || "") && /arxiv\.org/.test(p.url || "")) prev.url = p.url;
  }
  return [...byKey.values()];
}

// One literature sweep across all three providers (each failure-isolated).
// `plainQuery` goes to S2/OpenAlex; `arxivQuery` may use arXiv field syntax.
export async function searchScholarly({ plainQuery, arxivQuery, perSource = 10 }) {
  const [arx, s2, oa] = await Promise.all([
    fetchArxiv(arxivQuery || `all:"${compactText(plainQuery).replace(/"/g, "")}"`, perSource).catch(() => []),
    fetchSemanticScholar(plainQuery, perSource),
    fetchOpenAlex(plainQuery, perSource),
  ]);
  return {
    papers: mergeScholarly([arx, s2, oa]),
    counts: { arxiv: arx.length, semanticscholar: s2.length, openalex: oa.length },
  };
}

// Novelty probe for a hypothesis: find the closest existing work by title
// search. API-based, no LLM — results are evidence, not generation.
export async function probeNovelty(statement, count = 5) {
  const [s2, oa] = await Promise.all([
    fetchSemanticScholar(statement, count),
    fetchOpenAlex(statement, count),
  ]);
  return mergeScholarly([s2, oa])
    .slice(0, count)
    .map((p) => ({ title: p.title, url: p.url, date: p.date, citations: p.citations }));
}

export const paperAge = (p) => {
  const y = Number(String(p.date || "").slice(0, 4));
  return Number.isFinite(y) && y > 1900 ? new Date().getFullYear() - y : null;
};

// Tag what a reading list can see at a glance: highly cited older work is
// foundational, recent work is the frontier.
export function tagPaper(p) {
  const age = paperAge(p);
  if ((p.citations ?? 0) >= 100 && age != null && age >= 2) return "foundational";
  if (age != null && age <= 1) return "recent";
  return "";
}

export { normId };
