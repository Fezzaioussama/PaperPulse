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
    venue: compactText(typeof rec.venue === "string" ? rec.venue : "") || null,
    isOpenAccess: typeof rec.isOpenAccess === "boolean" ? rec.isOpenAccess : null,
    pdfUrl: typeof rec.openAccessPdf?.url === "string" ? rec.openAccessPdf.url : null,
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
    venue: compactText(rec.primary_location?.source?.display_name || "") || null,
    isOpenAccess: typeof rec.open_access?.is_oa === "boolean" ? rec.open_access.is_oa : null,
    pdfUrl: typeof rec.open_access?.oa_url === "string" && rec.open_access.oa_url ? rec.open_access.oa_url : null,
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
// `filters` (optional, from normalizeDiscoverFilters) maps to native S2
// search params; absent filters keep the original request byte-identical.
export async function fetchSemanticScholar(query, count = 10, filters = null) {
  try {
    const fields = "title,abstract,year,publicationDate,citationCount,externalIds,url,authors,venue,isOpenAccess,openAccessPdf";
    const params = [`query=${encodeURIComponent(query)}`, `limit=${Math.min(count, 20)}`, `fields=${fields}`];
    if (filters?.from || filters?.to) params.push(`publicationDateOrYear=${filters.from || ""}:${filters.to || ""}`);
    if (filters?.openAccess) params.push("openAccessPdf"); // bare flag param
    if (filters?.venue) params.push(`venue=${encodeURIComponent(filters.venue)}`);
    const url = `${S2_BASE}/paper/search?${params.join("&")}`;
    const key = (process.env.S2_API_KEY || "").trim();
    const data = await getJson(url, key ? { "x-api-key": key } : {});
    return (Array.isArray(data?.data) ? data.data : []).map(fromSemanticScholar).filter(Boolean);
  } catch (_) {
    return [];
  }
}

export async function fetchOpenAlex(query, count = 10, filters = null) {
  try {
    const select = "id,doi,title,display_name,publication_date,cited_by_count,authorships,open_access,abstract_inverted_index,primary_location";
    const params = [`search=${encodeURIComponent(query)}`, `per-page=${Math.min(count, 25)}`, `select=${select}`];
    const clauses = [];
    if (filters?.from) clauses.push(`from_publication_date:${filters.from}`);
    if (filters?.to) clauses.push(`to_publication_date:${filters.to}`);
    if (filters?.openAccess) clauses.push("is_oa:true");
    // Native author filter: the normalized shape truncates >2 authors to
    // "First et al.", so post-filtering cannot enforce authorship — the
    // provider has the full list and must do it. Commas/colons would break
    // OpenAlex's filter syntax, so they become spaces.
    if (filters?.author) clauses.push(`raw_author_name.search:${encodeURIComponent(compactText(filters.author.replace(/[,:]/g, " ")))}`);
    if (clauses.length) params.push(`filter=${clauses.join(",")}`);
    if (filters?.sort === "newest") params.push("sort=publication_date:desc");
    else if (filters?.sort === "citations") params.push("sort=cited_by_count:desc");
    const url = `${OPENALEX_BASE}/works?${params.join("&")}`;
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
    if (!prev.venue && p.venue) prev.venue = p.venue;
    if (prev.isOpenAccess == null && p.isOpenAccess != null) prev.isOpenAccess = p.isOpenAccess;
    if (!prev.pdfUrl && p.pdfUrl) prev.pdfUrl = p.pdfUrl;
    if (p.citations != null && (prev.citations == null || p.citations > prev.citations)) {
      prev.citations = p.citations;
    }
    if (!/arxiv\.org/.test(prev.url || "") && /arxiv\.org/.test(p.url || "")) prev.url = p.url;
  }
  return [...byKey.values()];
}

// ── Discover filters: normalize → provider params → post-filter → sort.
//    All pure except the fetches; invalid fields are DROPPED, never errors. ──

const DISCOVER_DATE_RE = /^\d{4}-\d{2}-\d{2}$/;
const DISCOVER_SORTS = ["relevance", "newest", "citations"];
const DISCOVER_CATEGORY_RE = /^[a-zA-Z.-]{2,40}$/;

// Clamp/validate the client's discover filters. Junk fields are dropped
// (a filter is best-effort, not a contract); perSource always resolves.
export function normalizeDiscoverFilters(raw) {
  const f = raw && typeof raw === "object" && !Array.isArray(raw) ? raw : {};
  const out = {};
  if (typeof f.from === "string" && DISCOVER_DATE_RE.test(f.from.trim())) out.from = f.from.trim();
  if (typeof f.to === "string" && DISCOVER_DATE_RE.test(f.to.trim())) out.to = f.to.trim();
  if (DISCOVER_SORTS.includes(f.sort)) out.sort = f.sort;
  if (f.openAccess === true) out.openAccess = true;
  const author = compactText(typeof f.author === "string" ? f.author : "").slice(0, 120);
  if (author) out.author = author;
  const venue = compactText(typeof f.venue === "string" ? f.venue : "").slice(0, 120);
  if (venue) out.venue = venue;
  const category = typeof f.category === "string" ? f.category.trim() : "";
  if (DISCOVER_CATEGORY_RE.test(category)) out.category = category;
  const per = Number.parseInt(f.perSource, 10);
  out.perSource = Number.isFinite(per) && per > 0 ? Math.min(per, 25) : 10;
  return out;
}

// Build an arXiv API query from a plain query + discover filters. A query
// that already uses arXiv field syntax is passed through verbatim.
export function buildArxivDiscoverQuery(query, filters = {}) {
  const q = compactText(query);
  const clauses = [/\b(all|ti|abs|cat):/i.test(q) ? q : `all:"${q.replace(/"/g, "")}"`];
  if (filters.category) clauses.push(`cat:${filters.category}`);
  if (filters.author) clauses.push(`au:"${filters.author.replace(/"/g, "")}"`);
  if (filters.from || filters.to) {
    const from = filters.from ? filters.from.replace(/-/g, "") : "19910101";
    const to = filters.to ? filters.to.replace(/-/g, "") : "20991231";
    clauses.push(`submittedDate:[${from}0000 TO ${to}2359]`);
  }
  return clauses.join(" AND ");
}

// A paper date can be "YYYY", "YYYY-MM", or "YYYY-MM-DD"; return the
// [earliest, latest] full dates it could mean, or null when unparseable.
function dateBounds(date) {
  const m = compactText(date).match(/^(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?/);
  if (!m) return null;
  return [`${m[1]}-${m[2] || "01"}-${m[3] || "01"}`, `${m[1]}-${m[2] || "12"}-${m[3] || "31"}`];
}

// Post-filter the merged pool: providers apply filters natively where they
// can, this pass enforces them uniformly. A paper is only dropped when it
// PROVABLY violates a filter — unknown values always pass through: unknown
// dates / OA status, a null venue (arXiv never reports one), and an author
// list truncated to "First Author et al." (the normalizers collapse >2
// authors, so a hidden co-author can never be proven absent; arXiv/OpenAlex
// enforce the author natively upstream).
export function applyDiscoverPostFilters(papers, filters = {}) {
  const author = filters.author ? normalizeForMatch(filters.author) : "";
  const venue = filters.venue ? normalizeForMatch(filters.venue) : "";
  return papers.filter((p) => {
    if (author) {
      const names = normalizeForMatch(p.authors || "");
      const complete = names && names !== "unknown" && !/ et al\.?$/.test(names);
      if (complete && !names.includes(author)) return false;
    }
    if (venue && p.venue && !normalizeForMatch(p.venue).includes(venue)) return false;
    if ((filters.from || filters.to) && p.date) {
      const bounds = dateBounds(p.date);
      if (bounds) {
        if (filters.from && bounds[1] < filters.from) return false;
        if (filters.to && bounds[0] > filters.to) return false;
      }
    }
    if (filters.openAccess && p.isOpenAccess === false) return false;
    return true;
  });
}

// Sort the merged pool. "relevance" (or anything else) keeps the merge
// order; missing dates / citation counts always sort last.
export function sortDiscoverPapers(papers, sort) {
  const list = [...papers];
  if (sort === "newest") return list.sort((a, b) => String(b.date || "").localeCompare(String(a.date || "")));
  if (sort === "citations") return list.sort((a, b) => (b.citations ?? -1) - (a.citations ?? -1));
  return list;
}

// One literature sweep across all three providers (each failure-isolated).
// `plainQuery` goes to S2/OpenAlex; `arxivQuery` may use arXiv field syntax.
// With `filters` (discover), the arXiv query is built from the filters and
// the merged pool is post-filtered and sorted; without them the behavior is
// unchanged for existing callers.
export async function searchScholarly({ plainQuery, arxivQuery, perSource = 10, filters = null }) {
  const aq = arxivQuery
    || (filters ? buildArxivDiscoverQuery(plainQuery, filters) : `all:"${compactText(plainQuery).replace(/"/g, "")}"`);
  const arxivSort = filters?.sort === "newest" ? "submittedDate" : "relevance";
  const [arx, s2, oa] = await Promise.all([
    fetchArxiv(aq, perSource, arxivSort).catch(() => []),
    fetchSemanticScholar(plainQuery, perSource, filters),
    fetchOpenAlex(plainQuery, perSource, filters),
  ]);
  let papers = mergeScholarly([arx, s2, oa]);
  if (filters) papers = sortDiscoverPapers(applyDiscoverPostFilters(papers, filters), filters.sort);
  return {
    papers,
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
