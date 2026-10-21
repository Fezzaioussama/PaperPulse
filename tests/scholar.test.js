import test from "node:test";
import assert from "node:assert/strict";
import {
  invertAbstract, arxivIdFromDoi, fromSemanticScholar, fromOpenAlex,
  scholarKey, mergeScholarly, tagPaper, searchScholarly,
} from "../api/_lib/scholar.js";
import { verifyQuote } from "../api/_lib/text.js";

const S2_REC = {
  title: "Selective State Spaces",
  abstract: "We study selective state spaces.",
  year: 2023,
  publicationDate: "2023-12-01",
  citationCount: 450,
  externalIds: { ArXiv: "2312.00752", DOI: "10.48550/arXiv.2312.00752" },
  url: "https://www.semanticscholar.org/paper/abc",
  authors: [{ name: "Albert Gu" }, { name: "Tri Dao" }, { name: "Someone Else" }],
};

const OA_REC = {
  id: "https://openalex.org/W123",
  doi: "https://doi.org/10.1234/other.5678",
  display_name: "A Survey of KV Caches",
  publication_date: "2024-06-01",
  cited_by_count: 12,
  open_access: { oa_url: "https://example.org/paper.pdf" },
  authorships: [{ author: { display_name: "Jane Roe" } }],
  abstract_inverted_index: { "We": [0], "survey": [1], "KV": [2], "caches.": [3] },
};

test("invertAbstract rebuilds text from OpenAlex inverted index", () => {
  assert.equal(invertAbstract(OA_REC.abstract_inverted_index), "We survey KV caches.");
  assert.equal(invertAbstract(null), "");
  assert.equal(invertAbstract({ junk: "nope" }), "");
});

test("source records normalize to the app paper shape", () => {
  const s2 = fromSemanticScholar(S2_REC);
  assert.equal(s2.url, "https://arxiv.org/abs/2312.00752"); // arXiv id wins over S2 url
  assert.equal(s2.authors, "Albert Gu et al.");
  assert.equal(s2.citations, 450);
  assert.equal(s2.source, "semanticscholar");

  const oa = fromOpenAlex(OA_REC);
  assert.equal(oa.title, "A Survey of KV Caches");
  assert.equal(oa.doi, "10.1234/other.5678");
  assert.equal(oa.abstract, "We survey KV caches.");
  assert.equal(oa.url, "https://example.org/paper.pdf");

  assert.equal(fromSemanticScholar({ title: " " }), null);
  assert.equal(fromOpenAlex(null), null);
  assert.equal(arxivIdFromDoi("10.48550/arXiv.2312.00752"), "2312.00752");
  assert.equal(arxivIdFromDoi("10.1234/x"), "");
});

test("mergeScholarly dedupes across sources and fills missing fields", () => {
  const arx = { title: "Selective State Spaces", url: "http://arxiv.org/abs/2312.00752v1", abstract: "", doi: "", date: "", citations: null, source: "arxiv" };
  const s2 = fromSemanticScholar(S2_REC);
  const oa = fromOpenAlex(OA_REC);
  const merged = mergeScholarly([[arx], [s2, oa]]);
  assert.equal(merged.length, 2); // arXiv + S2 collapse on the arXiv id
  const combined = merged.find((p) => scholarKey(p) === "arxiv:2312.00752");
  // The arXiv record came first and was enriched by the S2 duplicate.
  assert.equal(combined.source, "arxiv");
  assert.equal(combined.citations, 450);
  assert.equal(combined.abstract, "We study selective state spaces.");
  assert.equal(combined.doi, "10.48550/arxiv.2312.00752");
});

test("tagPaper marks foundational and recent work", () => {
  const thisYear = new Date().getFullYear();
  assert.equal(tagPaper({ date: `${thisYear - 4}-01-01`, citations: 900 }), "foundational");
  assert.equal(tagPaper({ date: `${thisYear}-01-01`, citations: 2 }), "recent");
  assert.equal(tagPaper({ date: `${thisYear - 4}-01-01`, citations: 3 }), "");
  assert.equal(tagPaper({ date: "", citations: 900 }), "");
});

test("verifyQuote only accepts verbatim quotes of real length", () => {
  const text = "Our selective layer reduces memory by forty percent, with no quality loss.";
  assert.equal(verifyQuote(text, "reduces MEMORY by forty percent"), true); // case/punct forgiven
  assert.equal(verifyQuote(text, "reduces memory by fifty percent"), false); // paraphrase rejected
  assert.equal(verifyQuote(text, "memory"), false); // too short to count
  assert.equal(verifyQuote("", "reduces memory by forty percent"), false);
});

test("searchScholarly merges three live sources and isolates failures", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url) => {
    const u = String(url);
    if (u.startsWith("https://export.arxiv.org/")) {
      return { ok: true, text: async () => `<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2312.00752v1</id><title>Selective State Spaces</title><summary>abs</summary><published>2023-12-01T00:00:00Z</published><author><name>Albert Gu</name></author><link rel="alternate" href="http://arxiv.org/abs/2312.00752v1"/><category term="cs.LG"/></entry></feed>` };
    }
    if (u.includes("semanticscholar.org")) {
      return { ok: false, status: 429, json: async () => ({}) }; // rate-limited → degrades to []
    }
    if (u.includes("openalex.org")) {
      return { ok: true, json: async () => ({ results: [OA_REC] }) };
    }
    throw new Error(`unexpected fetch: ${u}`);
  };
  try {
    const { papers, counts } = await searchScholarly({ plainQuery: "kv cache", perSource: 5 });
    assert.deepEqual(counts, { arxiv: 1, semanticscholar: 0, openalex: 1 });
    assert.equal(papers.length, 2);
  } finally {
    globalThis.fetch = originalFetch;
  }
});
