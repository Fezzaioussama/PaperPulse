import { XMLParser } from "fast-xml-parser";
import { compactText, normId } from "./text.js";

// Node-side arXiv access. Mirrors the client helpers in src/App.jsx but
// without DOMParser and without CORS proxies (the server fetches directly).

const FULLTEXT_LIMIT = 60000;

const parser = new XMLParser({
  ignoreAttributes: false,
  attributeNamePrefix: "@_",
  isArray: (name) => ["entry", "author", "category", "link"].includes(name),
});

// Produces the exact paper shape the client builds in parseArxivXml
// (App.jsx:968–995): {title, authors, date, url, abstract, categories, source}.
// The server additionally ADDS venue/isOpenAccess/pdfUrl (never renames a
// mirrored field): arXiv has no venue, is always open access, and the PDF
// URL derives mechanically from the arXiv id.
export function parseArxivXml(xml) {
  let feed;
  try { feed = parser.parse(xml)?.feed; } catch (_) { return []; }
  const entries = Array.isArray(feed?.entry) ? feed.entry : [];
  return entries.map((e) => {
    const authors = (e.author || [])
      .map((a) => compactText(a?.name || ""))
      .filter(Boolean);
    const authorStr = authors.length > 2 ? `${authors[0]} et al.` : authors.join(", ") || "—";
    const published = compactText(e.published || "");
    const links = e.link || [];
    const absLink = links.find((l) => l?.["@_rel"] === "alternate");
    const categories = (e.category || [])
      .map((c) => c?.["@_term"])
      .filter(Boolean);
    const url = absLink?.["@_href"] || compactText(e.id || "");
    const id = normId(url);
    return {
      title: compactText(e.title || ""),
      authors: authorStr,
      date: published ? published.slice(0, 10) : "",
      url,
      abstract: compactText(e.summary || ""),
      categories,
      source: "arxiv",
      venue: null,
      isOpenAccess: true,
      pdfUrl: /^\d{4}\.\d{4,5}$/.test(id) ? `https://arxiv.org/pdf/${id}` : null,
    };
  }).filter((p) => p.title && p.abstract);
}

const ARXIV_SORTS = ["relevance", "submittedDate", "lastUpdatedDate"];

export async function fetchArxiv(query, count, sortBy = "relevance") {
  const sort = ARXIV_SORTS.includes(sortBy) ? sortBy : "submittedDate";
  const apiUrl = `https://export.arxiv.org/api/query?search_query=${encodeURIComponent(query)}&sortBy=${sort}&sortOrder=descending&max_results=${count}`;
  try {
    const res = await fetch(apiUrl, { signal: AbortSignal.timeout(10000) });
    if (!res.ok) return [];
    const xml = await res.text();
    if (xml && xml.includes("<entry")) return parseArxivXml(xml);
  } catch (_) { /* treat a failed query as empty */ }
  return [];
}

export function dedupe(papers) {
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

// DOMParser-free, regex-based HTML → text. Lossier than the client version,
// which is acceptable: idea extraction defaults to abstracts and full text is
// an opt-in, best-effort input.
export function htmlToTextNode(html) {
  if (!html) return "";
  let s = String(html)
    .replace(/<!--[\s\S]*?-->/g, " ")
    .replace(/<(script|style|nav|header|footer|svg)\b[\s\S]*?<\/\1>/gi, " ");
  s = s
    .replace(/<h[1-4][^>]*>/gi, "\n## ")
    .replace(/<li[^>]*>/gi, "\n- ")
    .replace(/<(p|div|section|article|blockquote|figcaption|tr|br)[^>]*>/gi, "\n")
    .replace(/<[^>]+>/g, " ");
  s = s
    .replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"').replace(/&#0?39;/g, "'").replace(/&nbsp;/g, " ")
    .replace(/&#(\d+);/g, (_, n) => String.fromCharCode(Number(n)));
  return s
    .replace(/[ \t]+/g, " ")
    .replace(/\r?\n[ \t]*/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

async function fetchArxivHtml(id) {
  const urls = [
    `https://arxiv.org/html/${id}`,
    `https://ar5iv.labs.arxiv.org/html/${id}`,
  ];
  for (const u of urls) {
    try {
      const res = await fetch(u, { signal: AbortSignal.timeout(12000) });
      if (!res.ok) continue;
      const html = await res.text();
      if (html && /<\/?(article|main|body)/i.test(html)) return { html, base: u };
    } catch (_) { /* next source */ }
  }
  return null;
}

export async function fetchPaperTextNode(arxivIdOrUrl) {
  const id = normId(String(arxivIdOrUrl || ""));
  if (!/^\d{4}\.\d{4,5}$/.test(id)) return "";
  const got = await fetchArxivHtml(id);
  if (!got) return "";
  const text = htmlToTextNode(got.html);
  return text && text.length > 500 ? text.slice(0, FULLTEXT_LIMIT) : "";
}

// Pull the first figures (image + caption) out of a paper's HTML render.
// Relative srcs resolve against the page URL, e.g. arxiv.org/html/<id>/x1.png.
export function extractFigures(html, baseUrl, max = 4) {
  const figures = [];
  const base = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;
  const figRe = /<figure[^>]*>([\s\S]*?)<\/figure>/gi;
  let m;
  while ((m = figRe.exec(html)) && figures.length < max) {
    const block = m[1];
    const img = block.match(/<img[^>]*src=["']([^"']+)["']/i);
    if (!img || /^data:/i.test(img[1])) continue;
    let src;
    try { src = new URL(img[1], base).href; } catch (_) { continue; }
    if (!/^https:\/\//.test(src) || !/\.(png|jpe?g|gif|svg|webp)(\?|$)/i.test(src)) continue;
    const capM = block.match(/<figcaption[^>]*>([\s\S]*?)<\/figcaption>/i);
    const caption = capM ? compactText(capM[1].replace(/<[^>]+>/g, " ")).slice(0, 200) : "";
    if (!figures.some((f) => f.src === src)) figures.push({ src, caption });
  }
  return figures;
}

export async function fetchPaperFigures(arxivIdOrUrl, max = 4) {
  const id = normId(String(arxivIdOrUrl || ""));
  if (!/^\d{4}\.\d{4,5}$/.test(id)) return [];
  const got = await fetchArxivHtml(id);
  return got ? extractFigures(got.html, got.base, max) : [];
}
