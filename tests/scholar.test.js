import test from "node:test";
import assert from "node:assert/strict";
import {
  invertAbstract, arxivIdFromDoi, fromSemanticScholar, fromOpenAlex,
  scholarKey, mergeScholarly, tagPaper, searchScholarly,
  normalizeDiscoverFilters, buildArxivDiscoverQuery,
  applyDiscoverPostFilters, sortDiscoverPapers,
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

test("normalizeDiscoverFilters clamps and validates fields, dropping junk", () => {
  // Junk in → everything dropped, perSource falls back to its default.
  assert.deepEqual(normalizeDiscoverFilters(null), { perSource: 10 });
  assert.deepEqual(normalizeDiscoverFilters("junk"), { perSource: 10 });
  assert.deepEqual(normalizeDiscoverFilters({
    from: "2023-1-1", to: "20241231", sort: "weird", openAccess: "yes",
    author: "   ", venue: 42, category: "cs CL!", perSource: "nope",
  }), { perSource: 10 });
  // Valid fields survive; strings are compacted and clamped; perSource clamps.
  assert.deepEqual(normalizeDiscoverFilters({
    from: "2023-01-01", to: "2024-12-31", sort: "newest", openAccess: true,
    author: "  Jane   Roe ", venue: "NeurIPS", category: "cs.CL", perSource: 99,
  }), {
    from: "2023-01-01", to: "2024-12-31", sort: "newest", openAccess: true,
    author: "Jane Roe", venue: "NeurIPS", category: "cs.CL", perSource: 25,
  });
  assert.equal(normalizeDiscoverFilters({ author: "a".repeat(200) }).author.length, 120);
  assert.equal(normalizeDiscoverFilters({ perSource: 0 }).perSource, 10); // non-positive → default
  assert.equal(normalizeDiscoverFilters({ perSource: 3 }).perSource, 3);
  assert.equal(normalizeDiscoverFilters({ openAccess: false }).openAccess, undefined);
});

test("buildArxivDiscoverQuery passes field syntax through and appends filter clauses", () => {
  // Field-syntax queries are used verbatim; plain text becomes all:"...".
  assert.equal(buildArxivDiscoverQuery('ti:"state spaces" AND cat:cs.LG', {}), 'ti:"state spaces" AND cat:cs.LG');
  assert.equal(buildArxivDiscoverQuery('kv "cache" compression', {}), 'all:"kv cache compression"');
  assert.equal(
    buildArxivDiscoverQuery("state spaces", { category: "cs.CL", author: "Jane Roe" }),
    'all:"state spaces" AND cat:cs.CL AND au:"Jane Roe"'
  );
  // Open-ended date windows fill the missing bound mechanically.
  assert.equal(
    buildArxivDiscoverQuery("ssm", { from: "2023-01-01" }),
    'all:"ssm" AND submittedDate:[202301010000 TO 209912312359]'
  );
  assert.equal(
    buildArxivDiscoverQuery("ssm", { to: "2024-06-30" }),
    'all:"ssm" AND submittedDate:[199101010000 TO 202406302359]'
  );
  assert.equal(
    buildArxivDiscoverQuery("ssm", { from: "2023-01-01", to: "2024-06-30" }),
    'all:"ssm" AND submittedDate:[202301010000 TO 202406302359]'
  );
});

test("searchScholarly maps discover filters to the exact upstream URL params", async () => {
  const originalFetch = globalThis.fetch;
  const seen = {};
  globalThis.fetch = async (url) => {
    const u = String(url);
    if (u.startsWith("https://export.arxiv.org/")) { seen.arxiv = u; return { ok: true, text: async () => "" }; }
    if (u.includes("semanticscholar.org")) { seen.s2 = u; return { ok: true, json: async () => ({ data: [] }) }; }
    if (u.includes("openalex.org")) { seen.oa = u; return { ok: true, json: async () => ({ results: [] }) }; }
    throw new Error(`unexpected fetch: ${u}`);
  };
  try {
    const filters = normalizeDiscoverFilters({
      from: "2023-01-01", to: "2024-12-31", sort: "newest", openAccess: true,
      venue: "NeurIPS", category: "cs.CL", perSource: 5,
    });
    await searchScholarly({ plainQuery: "state space models", perSource: filters.perSource, filters });

    const arx = new URL(seen.arxiv);
    assert.equal(
      arx.searchParams.get("search_query"),
      'all:"state space models" AND cat:cs.CL AND submittedDate:[202301010000 TO 202412312359]'
    );
    assert.equal(arx.searchParams.get("sortBy"), "submittedDate"); // sort=newest
    assert.equal(arx.searchParams.get("max_results"), "5");

    const s2 = new URL(seen.s2);
    assert.equal(s2.searchParams.get("publicationDateOrYear"), "2023-01-01:2024-12-31");
    assert.ok(s2.searchParams.has("openAccessPdf")); // bare flag param
    assert.equal(s2.searchParams.get("venue"), "NeurIPS");
    assert.match(s2.searchParams.get("fields"), /venue,isOpenAccess,openAccessPdf/);

    const oa = new URL(seen.oa);
    assert.equal(
      oa.searchParams.get("filter"),
      "from_publication_date:2023-01-01,to_publication_date:2024-12-31,is_oa:true"
    );
    assert.equal(oa.searchParams.get("sort"), "publication_date:desc");
    assert.match(oa.searchParams.get("select"), /primary_location/);

    // Open date ends and relevance sort: no filter leftovers.
    seen.arxiv = seen.s2 = seen.oa = "";
    await searchScholarly({ plainQuery: "ssm", perSource: 5, filters: normalizeDiscoverFilters({ from: "2023-01-01" }) });
    assert.equal(new URL(seen.s2).searchParams.get("publicationDateOrYear"), "2023-01-01:");
    assert.equal(new URL(seen.oa).searchParams.get("filter"), "from_publication_date:2023-01-01");
    assert.equal(new URL(seen.oa).searchParams.get("sort"), null);
    assert.equal(new URL(seen.arxiv).searchParams.get("sortBy"), "relevance");
    assert.ok(!new URL(seen.s2).searchParams.has("openAccessPdf"));

    // Author maps to arXiv's au: clause AND OpenAlex's native author filter.
    seen.arxiv = seen.s2 = seen.oa = "";
    await searchScholarly({ plainQuery: "ssm", perSource: 5, filters: normalizeDiscoverFilters({ author: "Tri Dao" }) });
    assert.match(new URL(seen.arxiv).searchParams.get("search_query"), /au:"Tri Dao"/);
    assert.equal(new URL(seen.oa).searchParams.get("filter"), "raw_author_name.search:Tri Dao");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("applyDiscoverPostFilters enforces author/venue/date/openAccess on the merged pool", () => {
  const papers = [
    { title: "A", authors: "Jane Roe, Max Mustermann", venue: "NeurIPS 2024", date: "2024-06-01", isOpenAccess: true },
    { title: "B", authors: "Someone Else", venue: "ICML", date: "2022-01-01", isOpenAccess: false },
    { title: "C", authors: "Jane Roe", venue: null, date: "", isOpenAccess: null },
    { title: "D", authors: "jane roe", venue: "Workshop @ NeurIPS", date: "2023", isOpenAccess: null },
  ];
  assert.deepEqual(applyDiscoverPostFilters(papers, {}).map((p) => p.title), ["A", "B", "C", "D"]);
  assert.deepEqual(applyDiscoverPostFilters(papers, { author: "Jane Roe" }).map((p) => p.title), ["A", "C", "D"]);
  // An unknown venue (arXiv reports none) is never a provable violation.
  assert.deepEqual(applyDiscoverPostFilters(papers, { venue: "neurips" }).map((p) => p.title), ["A", "C", "D"]);
  // Date range: missing dates pass; a year-only date is dropped only when
  // provably outside the window.
  assert.deepEqual(applyDiscoverPostFilters(papers, { from: "2023-01-01" }).map((p) => p.title), ["A", "C", "D"]);
  assert.deepEqual(applyDiscoverPostFilters(papers, { to: "2022-12-31" }).map((p) => p.title), ["B", "C"]);
  assert.deepEqual(applyDiscoverPostFilters(papers, { from: "2024-01-01", to: "2024-12-31" }).map((p) => p.title), ["A", "C"]);
  // openAccess drops only papers KNOWN to be closed.
  assert.deepEqual(applyDiscoverPostFilters(papers, { openAccess: true }).map((p) => p.title), ["A", "C", "D"]);
});

test("author post-filter never drops a paper whose author list is truncated to 'et al.'", () => {
  // All three normalizers collapse >2 authors to "First et al.", so a match
  // against the co-author list can never be proven absent — those papers pass
  // (arXiv's au: and OpenAlex's raw_author_name.search enforce it natively).
  const papers = [
    { title: "first-author", authors: "Tri Dao et al." },
    { title: "hidden-coauthor", authors: "Albert Gu et al." },
    { title: "full-list-miss", authors: "Jane Roe, Max Mustermann" },
    { title: "full-list-hit", authors: "Someone Else, Tri Dao" },
    { title: "unknown-authors", authors: "Unknown" },
  ];
  assert.deepEqual(
    applyDiscoverPostFilters(papers, { author: "Tri Dao" }).map((p) => p.title),
    ["first-author", "hidden-coauthor", "full-list-hit", "unknown-authors"]
  );
});

test("sortDiscoverPapers orders by date or citations with unknowns last", () => {
  const papers = [
    { title: "old", date: "2020-01-01", citations: 500 },
    { title: "undated", date: "", citations: 0 },
    { title: "new", date: "2024-06-01", citations: null },
  ];
  assert.deepEqual(sortDiscoverPapers(papers, "newest").map((p) => p.title), ["new", "old", "undated"]);
  assert.deepEqual(sortDiscoverPapers(papers, "citations").map((p) => p.title), ["old", "undated", "new"]);
  // relevance (or anything else) keeps the merge order and copies the array.
  const kept = sortDiscoverPapers(papers, "relevance");
  assert.deepEqual(kept.map((p) => p.title), ["old", "undated", "new"]);
  assert.notEqual(kept, papers);
});

test("mergeScholarly backfills venue, isOpenAccess, and pdfUrl from later duplicates", () => {
  const arx = {
    title: "Selective State Spaces", url: "http://arxiv.org/abs/2312.00752v1",
    abstract: "", doi: "", date: "", citations: null, source: "arxiv",
    venue: null, isOpenAccess: null, pdfUrl: null,
  };
  const s2 = {
    title: "Selective State Spaces", url: "https://arxiv.org/abs/2312.00752",
    abstract: "abs", doi: "10.48550/arxiv.2312.00752", date: "2023-12-01", citations: 450, source: "semanticscholar",
    venue: "NeurIPS", isOpenAccess: false, pdfUrl: "https://example.org/paper.pdf",
  };
  const [merged] = mergeScholarly([[arx], [s2]]);
  assert.equal(merged.source, "arxiv"); // first occurrence wins
  assert.equal(merged.venue, "NeurIPS");
  assert.equal(merged.isOpenAccess, false); // null is backfilled even by a false
  assert.equal(merged.pdfUrl, "https://example.org/paper.pdf");
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
