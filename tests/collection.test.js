import test from "node:test";
import assert from "node:assert/strict";
import { normalizeCollectionPapers, buildCollectionGraph } from "../api/_lib/collection.js";

test("normalizeCollectionPapers rejects bad shapes and clips the good ones", () => {
  // Bad shapes → null (the endpoint answers 400).
  assert.equal(normalizeCollectionPapers(null), null);
  assert.equal(normalizeCollectionPapers("junk"), null);
  assert.equal(normalizeCollectionPapers([{ title: "only one" }]), null);
  assert.equal(normalizeCollectionPapers(Array.from({ length: 17 }, (_, i) => ({ title: `p${i}` }))), null);
  assert.equal(normalizeCollectionPapers([{ title: "a" }, { abstract: "no title" }]), null);
  assert.equal(normalizeCollectionPapers([{ title: "a" }, { title: "   " }]), null);
  assert.equal(normalizeCollectionPapers([{ title: "a" }, "not an object"]), null);

  // Good shapes: strings clipped, unknown fields dropped, defaults filled.
  const papers = normalizeCollectionPapers([
    { title: "t".repeat(400), abstract: "x".repeat(5000), authors: 42, citations: "12", extra: "dropped" },
    { title: "Short", url: "http://arxiv.org/abs/2401.00001v1", date: "2024-01-05T00:00:00Z", venue: "ICLR", citations: 7, doi: "10.1/x" },
  ]);
  assert.equal(papers.length, 2);
  assert.ok(papers[0].title.length <= 304); // clipText(. , 300) + " ..."
  assert.ok(papers[0].abstract.length <= 2004);
  assert.equal(papers[0].authors, "");
  assert.equal(papers[0].citations, null); // non-integer dropped
  assert.equal(papers[0].extra, undefined);
  assert.equal(papers[1].date, "2024-01-05"); // clipped to YYYY-MM-DD
  assert.equal(papers[1].citations, 7);
  assert.equal(papers[1].doi, "10.1/x");
});

// ── End-to-end collection run with a substring-routed stub model. The graph
//    makes no HTTP requests; a tripwire fetch stub proves it. ──

const PAPERS = normalizeCollectionPapers([
  {
    title: "Selective State Spaces", url: "http://arxiv.org/abs/2401.00001v1",
    abstract: "Selective state space models for memory efficiency.",
    authors: "A. Gu, T. Dao", date: "2024-01-05", venue: "ICLR", citations: 450,
  },
  { title: "KV Cache Survey", abstract: "A survey of KV cache compression methods.", authors: "J. Roe" },
]);
// paperId: arXiv id for the first, lowercased title for the url-less second.
const ID1 = "2401.00001";
const ID2 = "kv cache survey";

const ROUTES = [
  ["curating a researcher's paper collection", `{"themes":[{"name":"Memory efficiency","summary":"s","paperIds":["${ID1}","${ID2}","ghost.id"],"agreements":["both cut memory"],"contradictions":[]}],"concepts":[{"name":"KV cache","explanation":"The attention key/value store reused across steps.","paperIds":["${ID2}","ghost.id"]}]}`],
  ["side-by-side comparison of papers in this collection", `{"rows":[{"paperId":"${ID1}","approach":"selective SSM","contribution":"linear memory","evidence":"benchmarks","limitation":"recall"},{"paperId":"ghost.id","approach":"x"}]}`],
  ["an executive brief of this paper collection", "# Brief\n\nBoth papers tackle memory [1][2]. Bogus cite [9]."],
];

const makeModel = (routes, { failMarker = "" } = {}) => ({
  invoke: async (prompt) => {
    const p = String(prompt);
    if (failMarker && p.includes(failMarker)) throw new Error("stubbed model failure");
    const hit = routes.find(([marker]) => p.includes(marker));
    if (!hit) throw new Error(`unexpected prompt: ${p.slice(0, 80)}`);
    return { content: hit[1] };
  },
});

test("collection graph runs end to end: phases, partials, id validation, citation sanitizing", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url) => { throw new Error(`collection graph must not fetch: ${url}`); };
  const events = [];
  try {
    const graph = buildCollectionGraph();
    const final = await graph.invoke(
      { name: "Efficient attention", papers: PAPERS },
      { configurable: { model: makeModel(ROUTES), emit: (e) => events.push(e) }, recursionLimit: 8 }
    );

    // Unknown paper ids are dropped mechanically by the normalizers.
    assert.equal(final.themes.length, 1);
    assert.deepEqual(final.themes[0].paperIds, [ID1, ID2]);
    assert.equal(final.concepts.length, 1);
    assert.deepEqual(final.concepts[0].paperIds, [ID2]);
    assert.equal(final.comparison.length, 1);
    assert.equal(final.comparison[0].paperId, ID1);

    // Out-of-range [n] citations are stripped from the brief.
    assert.ok(final.brief.includes("[1][2]"));
    assert.ok(!final.brief.includes("[9]"));

    const phases = events.filter((e) => e.type === "phase").map((e) => e.phase);
    assert.deepEqual(phases, ["themes", "compare", "brief"]);
    const partialKeys = events.filter((e) => e.type === "partial").map((e) => e.key);
    assert.deepEqual(partialKeys, ["themes", "concepts", "comparison", "brief"]);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("an abort inside a stage is rethrown, never logged-and-continued", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url) => { throw new Error(`collection graph must not fetch: ${url}`); };
  const events = [];
  try {
    const graph = buildCollectionGraph();
    const ac = new AbortController();
    const abortingModel = {
      invoke: async () => {
        ac.abort();
        const err = new Error("This operation was aborted");
        err.name = "AbortError";
        throw err;
      },
    };
    await assert.rejects(
      graph.invoke(
        { name: "Efficient attention", papers: PAPERS },
        { configurable: { model: abortingModel, emit: (e) => events.push(e), signal: ac.signal }, recursionLimit: 8 }
      ),
      (err) => err.name === "AbortError" || /abort/i.test(err.message)
    );
    // The first stage aborted: no degrade log, no later phases ran.
    assert.ok(!events.some((e) => e.type === "log"));
    assert.deepEqual(events.filter((e) => e.type === "phase").map((e) => e.phase), ["themes"]);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("a failing stage logs and degrades without killing the rest of the run", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url) => { throw new Error(`collection graph must not fetch: ${url}`); };
  const events = [];
  try {
    const graph = buildCollectionGraph();
    const final = await graph.invoke(
      { name: "Efficient attention", papers: PAPERS },
      {
        configurable: {
          model: makeModel(ROUTES, { failMarker: "side-by-side comparison of papers in this collection" }),
          emit: (e) => events.push(e),
        },
        recursionLimit: 8,
      }
    );
    assert.equal(final.themes.length, 1); // earlier stage intact
    assert.deepEqual(final.comparison, []); // failed stage degrades to empty
    assert.ok(final.brief.startsWith("# Brief")); // later stage still ran
    assert.ok(events.some((e) => e.type === "log" && /comparison/i.test(e.message)));
    const phases = events.filter((e) => e.type === "phase").map((e) => e.phase);
    assert.deepEqual(phases, ["themes", "compare", "brief"]);
  } finally {
    globalThis.fetch = originalFetch;
  }
});
