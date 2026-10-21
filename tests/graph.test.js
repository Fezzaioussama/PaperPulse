import test from "node:test";
import assert from "node:assert/strict";
import {
  pickRelevant, decideNext, normalizeThemes, resolveLimits,
  buildResearchGraph, lookupKnowledge,
  sanitizeReportCitations, normalizeConcepts, normalizeCritique, normalizePractice,
  normalizeComparison, decideReview,
} from "../api/_lib/graph.js";

test("pickRelevant keeps papers at the threshold but never fewer than minKeep", () => {
  const graded = [
    { title: "a", relevance: 95 },
    { title: "b", relevance: 72 },
    { title: "c", relevance: 40 },
    { title: "d", relevance: 10 },
  ];
  assert.deepEqual(pickRelevant(graded, 70, 2).map((p) => p.title), ["a", "b"]);
  // Only one passes, but minKeep=3 backfills with the top-scored rest.
  assert.deepEqual(pickRelevant(graded, 90, 3).map((p) => p.title), ["a", "b", "c"]);
  // Fewer papers than minKeep: return everything, best first.
  assert.equal(pickRelevant(graded.slice(0, 2), 99, 5).length, 2);
});

test("decideNext refines only while relevant papers are scarce and iterations remain", () => {
  const limits = { minRelevant: 6 };
  assert.equal(decideNext({ relevant: new Array(2), iterations: 1, limits }), "refine");
  assert.equal(decideNext({ relevant: new Array(2), iterations: 2, limits }), "extract");
  assert.equal(decideNext({ relevant: new Array(8), iterations: 1, limits }), "extract");
});

test("normalizeThemes validates model output", () => {
  const ideas = [
    { paperId: "2401.00001", title: "A", ideas: [] },
    { paperId: "2401.00002", title: "B", ideas: [] },
  ];
  const parsed = [
    { name: "Shared theme", summary: "s", paperIds: ["2401.00001", "2401.00001", "unknown.id", "2401.00002"], agreements: ["yes"], contradictions: [] },
    { name: "Ghost theme", summary: "s", paperIds: ["unknown.id"] },
    { summary: "nameless", paperIds: ["2401.00001"] },
    "not an object",
  ];
  const themes = normalizeThemes(parsed, ideas);
  assert.equal(themes.length, 1);
  assert.deepEqual(themes[0].paperIds, ["2401.00001", "2401.00002"]);
  assert.deepEqual(normalizeThemes("garbage", ideas), []);
});

test("resolveLimits clamps options to hard ceilings and fills defaults", () => {
  const limits = resolveLimits({ maxQueries: 50, maxExtract: "4", threshold: 85, junk: 1 });
  assert.equal(limits.maxQueries, 8);     // clamped to hard max
  assert.equal(limits.maxExtract, 4);     // honored below the ceiling
  assert.equal(limits.threshold, 85);
  assert.equal(limits.maxGraded, 60);     // default
  assert.equal(limits.deep, false);
});

const ARXIV_XML = `<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <title>State Space Models for Long Context</title>
    <summary>We study state space models for long context modeling.</summary>
    <published>2024-01-05T00:00:00Z</published>
    <author><name>Alice Smith</name></author>
    <link rel="alternate" href="http://arxiv.org/abs/2401.00001v1"/>
    <category term="cs.LG"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2401.00002v1</id>
    <title>Linear Attention Revisited</title>
    <summary>Linear attention as an alternative to softmax attention.</summary>
    <published>2024-01-06T00:00:00Z</published>
    <author><name>Dan Brown</name></author>
    <link rel="alternate" href="http://arxiv.org/abs/2401.00002v1"/>
    <category term="cs.LG"/>
  </entry>
</feed>`;

test("research graph runs end to end with a stubbed model and arXiv feed", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url) => {
    assert.ok(String(url).startsWith("https://export.arxiv.org/"));
    return { ok: true, text: async () => ARXIV_XML };
  };
  // The analysis passes run in parallel, so canned replies are routed by a
  // distinctive substring of each node's prompt instead of by call order.
  const routes = [
    ["arXiv API search queries", '["all:\\"state space model\\" OR abs:ssm"]'],                   // plan_queries
    ["Rate how closely each paper", '[{"i":0,"relevance":95,"reason":"on topic"},{"i":1,"relevance":80,"reason":"related"}]'], // grade_relevance
    ["Extract the key ideas", '{"ideas":[{"claim":"SSMs scale linearly","kind":"finding","detail":"d1"}]}'], // extract (both papers)
    ["Regroup these ideas", '[{"name":"Sub-quadratic sequence modeling","summary":"s","paperIds":["2401.00001","2401.00002"],"agreements":["both drop softmax"],"contradictions":[]}]'],
    ["patient teacher", '{"concepts":[{"name":"State space model","explanation":"A recurrent view of sequences.","paperIds":["2401.00001"]}]}'],
    ["skeptical senior reviewer", '{"critiques":[{"kind":"conflict","target":"2401.00002","point":"Claims differ on softmax need."}],"openQuestions":["Does it hold at scale?"]}'],
    ["pragmatic engineer", '{"recommendations":[{"scenario":"very long inputs","approach":"use an SSM","why":"linear cost"}],"cautions":["training instability"],"resources":["LRA benchmark"]}'],
    ["side-by-side comparison", '{"rows":[{"paperId":"2401.00001","approach":"selective SSM","contribution":"linear scaling","evidence":"LRA results","limitation":"recall"},{"paperId":"bogus.id","approach":"x"}]}'],
    ["fact-checking a research briefing", '{"verdict":"revise","issues":["Overview claims recall parity the ideas do not support."]}'],
    ["Revise this research briefing", "# Briefing (revised)\n\nBoth papers study sub-quadratic models [1][2]."],
    ["research briefing in markdown", "# Briefing\n\nBoth papers study sub-quadratic models [1][2]. Bogus cite [7]."],
  ];
  const model = {
    invoke: async (prompt) => {
      const hit = routes.find(([marker]) => String(prompt).includes(marker));
      return { content: hit ? hit[1] : "[]" };
    },
  };
  const events = [];
  try {
    const graph = buildResearchGraph();
    const final = await graph.invoke(
      { topic: "state space models", limits: resolveLimits({ minRelevant: 1 }) },
      { configurable: { model, emit: (e) => events.push(e) }, recursionLimit: 25 }
    );
    assert.equal(final.relevant.length, 2);
    assert.equal(final.relevant[0].relevance, 95);
    assert.equal(final.ideas.length, 2);
    assert.equal(final.themes.length, 1);
    assert.deepEqual(final.themes[0].paperIds, ["2401.00001", "2401.00002"]);
    assert.equal(final.concepts.length, 1);
    assert.equal(final.critique.points[0].kind, "conflict");
    assert.equal(final.practice.recommendations[0].approach, "use an SSM");
    // The comparison keeps only rows for known papers.
    assert.equal(final.comparison.length, 1);
    assert.equal(final.comparison[0].approach, "selective SSM");
    // The self-review found an issue, so the revised report must have won.
    assert.equal(final.reportReview.verdict, "revise");
    assert.ok(final.report.startsWith("# Briefing (revised)"));
    assert.ok(final.report.includes("[1][2]"));
    assert.ok(!final.report.includes("[7]")); // out-of-range citation stripped
    const phases = events.filter((e) => e.type === "phase").map((e) => e.phase);
    assert.deepEqual(phases.slice(0, 5), ["plan", "search", "grade", "extract", "regroup"]);
    // The four analysis passes run in parallel: order among them is not fixed.
    assert.deepEqual([...phases.slice(5, 9)].sort(), ["challenge", "compare", "practice", "understand"]);
    assert.deepEqual(phases.slice(9), ["synthesize", "review", "revise"]);
    assert.ok(events.some((e) => e.type === "papers" && e.papers.length === 2));
    assert.ok(events.some((e) => e.type === "idea_progress" && e.done === 2 && e.total === 2));
    // Partial snapshots stream out for client-side salvage.
    const partialKeys = events.filter((e) => e.type === "partial").map((e) => e.key);
    for (const key of ["ideas", "themes", "concepts", "critique", "practice", "comparison", "report"]) {
      assert.ok(partialKeys.includes(key), `missing partial: ${key}`);
    }
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("sanitizeReportCitations strips citations outside the paper list", () => {
  assert.equal(sanitizeReportCitations("Good [1] and [3], bad [4] and [12].", 3), "Good [1] and [3], bad  and .");
  assert.equal(sanitizeReportCitations("No cites here.", 0), "No cites here.");
  assert.equal(sanitizeReportCitations("[0] is never valid.", 5), " is never valid.");
});

test("normalizeConcepts keeps only known paper ids and non-empty explanations", () => {
  const papers = [{ url: "http://arxiv.org/abs/2401.00001v1", title: "A" }];
  const parsed = {
    concepts: [
      { name: "SSM", explanation: "A thing.", paperIds: ["2401.00001", "9999.99999"] },
      { name: "Empty", explanation: "", paperIds: [] },
      { explanation: "nameless" },
    ],
  };
  const out = normalizeConcepts(parsed, papers);
  assert.equal(out.length, 1);
  assert.deepEqual(out[0].paperIds, ["2401.00001"]);
  assert.deepEqual(normalizeConcepts("junk", papers), []);
});

test("normalizeCritique and normalizePractice validate shapes and return null when empty", () => {
  const critique = normalizeCritique({
    critiques: [{ kind: "made-up", target: "T", point: "p" }, { point: "" }],
    openQuestions: ["q1", "", 42],
  });
  assert.equal(critique.points.length, 1);
  assert.equal(critique.points[0].kind, "weakness"); // unknown kind coerced
  assert.deepEqual(critique.openQuestions, ["q1", "42"]); // empties dropped, numbers stringified
  assert.equal(normalizeCritique({ critiques: [], openQuestions: [] }), null);
  assert.equal(normalizeCritique(null), null);

  const practice = normalizePractice({
    recommendations: [{ scenario: "s", approach: "a", why: "w" }, { approach: "" }],
    cautions: ["c"],
    resources: [],
  });
  assert.equal(practice.recommendations.length, 1);
  assert.deepEqual(practice.cautions, ["c"]);
  assert.equal(normalizePractice({}), null);
});

test("normalizeComparison keeps one grounded row per known paper", () => {
  const papers = [
    { url: "http://arxiv.org/abs/2401.00001v1", title: "A" },
    { url: "http://arxiv.org/abs/2401.00002v1", title: "B" },
  ];
  const out = normalizeComparison({
    rows: [
      { paperId: "2401.00001", approach: "a1", contribution: "c", evidence: "e", limitation: "l" },
      { paperId: "2401.00001", approach: "duplicate" },
      { paperId: "unknown", approach: "x" },
      { paperId: "2401.00002", approach: "" }, // approach required
    ],
  }, papers);
  assert.equal(out.length, 1);
  assert.equal(out[0].approach, "a1");
  assert.deepEqual(normalizeComparison(null, papers), []);
});

test("decideReview revises only on a revise verdict with a report present", () => {
  assert.equal(decideReview({ reportReview: { verdict: "revise" }, report: "r" }), "revise");
  assert.equal(decideReview({ reportReview: { verdict: "revise" }, report: "" }), "done");
  assert.equal(decideReview({ reportReview: { verdict: "ok" }, report: "r" }), "done");
  assert.equal(decideReview({ reportReview: null, report: "r" }), "done");
});

test("lookupKnowledge returns lexical matches across papers, ideas, and themes", () => {
  const knowledge = {
    runs: [{
      topic: "ssm",
      papers: [{ title: "State Space Models", abstract: "long context", url: "u", date: "2024-01-01", relevance: 90 }],
      ideas: [{ paperId: "p", title: "State Space Models", ideas: [{ claim: "SSMs scale linearly", kind: "finding", detail: "scaling detail" }] }],
      themes: [{ name: "Scaling behavior", summary: "how models scale", agreements: [], contradictions: [] }],
    }],
  };
  const hit = JSON.parse(lookupKnowledge(knowledge, "state space scaling"));
  assert.ok(hit.matches.length >= 2);
  const none = JSON.parse(lookupKnowledge(knowledge, "quantum chemistry"));
  assert.equal(none.matches.length, 0);
});
