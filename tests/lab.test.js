import test from "node:test";
import assert from "node:assert/strict";
import {
  resolveLabLimits, normalizePlan, normalizeHypotheses, normalizeDebate,
  normalizeProtocols, buildVerificationStats, buildLabGraph, LAB_AGENTS,
} from "../api/_lib/lab.js";

const PAPERS = [
  { title: "Selective State Spaces", url: "http://arxiv.org/abs/2401.00001v1", abstract: "a1" },
  { title: "KV Cache Survey", url: "https://doi.org/10.1/x", abstract: "a2" },
];

test("resolveLabLimits clamps options", () => {
  assert.deepEqual(resolveLabLimits(), { maxPapers: 12, maxAnalyze: 6, deep: false });
  assert.deepEqual(resolveLabLimits({ maxPapers: 99, maxAnalyze: 2, deep: 1 }), { maxPapers: 16, maxAnalyze: 2, deep: true });
});

test("normalizePlan whitelists agents and falls back to the standard plan", () => {
  const plan = normalizePlan({ tasks: [{ agent: "scout", task: "search" }, { agent: "hacker", task: "x" }, { agent: "writer" }] }, "obj");
  assert.deepEqual(plan, [{ agent: "scout", task: "search" }, { agent: "director", task: "x" }]);
  const fallback = normalizePlan("junk", "my objective");
  assert.equal(fallback.length, 6);
  assert.ok(fallback[0].task.includes("my objective"));
});

test("normalizeHypotheses assigns ids and drops unknown paper refs", () => {
  const hs = normalizeHypotheses({ hypotheses: [
    { statement: "S1", rationale: "r", paperIds: ["2401.00001", "bogus"], feasibility: "f" },
    { statement: "  " },
    { statement: "S2" },
  ] }, PAPERS);
  assert.equal(hs.length, 2);
  assert.deepEqual(hs.map((h) => h.id), ["H1", "H2"]);
  assert.deepEqual(hs[0].paperIds, ["2401.00001"]);
});

test("normalizeDebate backfills missing verdicts as contested, never approved", () => {
  const hs = [{ id: "H1" }, { id: "H2" }];
  const debate = normalizeDebate({
    critiques: { critiques: [{ hypothesisId: "H1", point: "weak" }, { hypothesisId: "H9", point: "x" }] },
    rebuttals: { rebuttals: [{ hypothesisId: "H1", point: "but" }] },
    arbiter: { verdicts: [{ hypothesisId: "H1", verdict: "supported", confidence: 80, reason: "ok" }, { hypothesisId: "H1b", verdict: "nope" }], disagreements: ["open point"] },
  }, hs);
  assert.equal(debate.critiques.length, 1);
  assert.equal(debate.verdicts.find((v) => v.hypothesisId === "H1").verdict, "supported");
  const h2 = debate.verdicts.find((v) => v.hypothesisId === "H2");
  assert.equal(h2.verdict, "contested");
  assert.equal(h2.confidence, 0);
  assert.deepEqual(debate.disagreements, ["open point"]);
});

test("normalizeProtocols keeps only protocols for known hypotheses", () => {
  const ps = normalizeProtocols({ protocols: [
    { hypothesisId: "H1", objective: "test it", metrics: ["ppl", ""], baselines: ["full attention"] },
    { hypothesisId: "H9", objective: "x" },
    { hypothesisId: "H1", objective: "" },
  ] }, [{ id: "H1" }]);
  assert.equal(ps.length, 1);
  assert.deepEqual(ps[0].metrics, ["ppl"]);
});

test("buildVerificationStats is computed mechanically", () => {
  const stats = buildVerificationStats({
    papers: PAPERS,
    sourceCounts: { arxiv: 3 },
    findings: [
      { evidence: "fulltext", claims: [{ verified: true }, { verified: false }] },
      { evidence: "abstract", claims: [{ verified: false }] },
    ],
    debate: { verdicts: [{ verdict: "supported" }, { verdict: "contested" }, { verdict: "contested" }] },
  });
  assert.equal(stats.claims, 3);
  assert.equal(stats.verifiedQuotes, 1);
  assert.equal(stats.fulltextAnalyses, 1);
  assert.deepEqual(stats.verdicts, { supported: 1, contested: 2 });
});

// ── End-to-end lab run with a stubbed model and stubbed literature APIs ──

const ARXIV_XML = `<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><id>http://arxiv.org/abs/2401.00001v1</id><title>Selective SSM Memory</title>
<summary>Selective state space models for memory efficiency.</summary>
<published>2024-01-05T00:00:00Z</published><author><name>A. Gu</name></author>
<link rel="alternate" href="http://arxiv.org/abs/2401.00001v1"/><category term="cs.LG"/></entry></feed>`;

const FULLTEXT_SENTENCE = "our selective layer reduces memory by forty percent on long context benchmarks";

test("lab graph runs end to end: verified evidence, debate verdicts, protocols, report", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url) => {
    const u = String(url);
    if (u.startsWith("https://export.arxiv.org/")) return { ok: true, text: async () => ARXIV_XML };
    if (u.includes("semanticscholar.org")) {
      return { ok: true, json: async () => ({ data: [{
        title: "KV Cache Compression Survey", abstract: "A survey of cache compression.",
        year: 2024, publicationDate: "2024-06-01", citationCount: 40,
        externalIds: { DOI: "10.1/kv" }, url: "https://example.org/kv", authors: [{ name: "J. Roe" }],
      }] }) };
    }
    if (u.includes("openalex.org")) return { ok: true, json: async () => ({ results: [] }) };
    if (u.includes("/html/2401.00001")) {
      const filler = "<p>Additional methodological detail about gating, benchmarks, and training procedure.</p>".repeat(10);
      return { ok: true, text: async () => `<body><article><p>${FULLTEXT_SENTENCE}.</p>${filler}</article></body>` };
    }
    if (u.includes("/html/") || u.includes("ar5iv")) return { ok: false, status: 404, text: async () => "" };
    throw new Error(`unexpected fetch: ${u}`);
  };
  const routes = [
    ["research director planning an investigation", '{"tasks":[{"agent":"scout","task":"search kv cache work"},{"agent":"analyst","task":"extract evidence"}]}'],
    ["literature scout", '{"queries":["kv cache compression"],"arxivQuery":"all:\\"kv cache\\""}'],
    ["Rate how closely each paper", '[{"i":0,"relevance":90},{"i":1,"relevance":75}]'],
    ["Extract the key claims", `{"claims":[{"claim":"Memory drops 40%","kind":"finding","quote":"reduces memory by forty percent on long context benchmarks"},{"claim":"Fabricated","kind":"finding","quote":"this exact sentence appears nowhere in the paper"}]}`],
    ["testable hypotheses", '{"hypotheses":[{"statement":"Selective gating can compress KV caches losslessly","rationale":"r","paperIds":["2401.00001"],"feasibility":"single GPU"},{"statement":"Second idea","rationale":"r2","paperIds":[]}]}'],
    ["skeptical reviewer. Attack", '{"critiques":[{"hypothesisId":"H1","point":"Lossless is a strong claim"}]}'],
    ["You are the proponent", '{"rebuttals":[{"hypothesisId":"H1","point":"Verified finding shows 40% headroom"}]}'],
    ["impartial arbiter", '{"verdicts":[{"hypothesisId":"H1","verdict":"supported","confidence":70,"reason":"evidence favors"}],"disagreements":["Scaling beyond 1B is unresolved"]}'],
    ["design rigorous ML experiment protocols", '{"protocols":[{"hypothesisId":"H1","objective":"Measure compression vs quality","independentVars":["gate threshold"],"dependentVars":["perplexity"],"baselines":["full cache"],"datasets":["to be selected"],"metrics":["perplexity"],"ablations":["no gating"],"risks":["overfit"],"acceptance":"ppl within 1%"}]}'],
    ["investigation report in markdown", "# Investigation\n\nEvidence shows a 40% reduction [1]. Bogus cite [9]."],
  ];
  const model = {
    invoke: async (prompt) => {
      const hit = routes.find(([marker]) => String(prompt).includes(marker));
      if (!hit) throw new Error(`unexpected prompt: ${String(prompt).slice(0, 80)}`);
      return { content: hit[1] };
    },
  };
  const events = [];
  try {
    const graph = buildLabGraph();
    const final = await graph.invoke(
      { objective: "kv cache memory compression", limits: resolveLabLimits({ maxAnalyze: 2 }) },
      { configurable: { model, emit: (e) => events.push(e) }, recursionLimit: 12 }
    );

    assert.equal(final.plan.length, 2);
    assert.equal(final.papers.length, 2); // arXiv + S2, deduped pool, both graded in
    assert.equal(final.sourceCounts.arxiv, 1);
    assert.equal(final.sourceCounts.semanticscholar, 1);

    // Quote verification: real quote verified against fetched full text; the
    // fabricated one rejected mechanically.
    const arxivFinding = final.findings.find((f) => f.paperId === "2401.00001");
    assert.equal(arxivFinding.evidence, "fulltext");
    assert.equal(arxivFinding.claims[0].verified, true);
    assert.equal(arxivFinding.claims[1].verified, false);

    assert.equal(final.hypotheses.length, 2);
    assert.ok(final.hypotheses[0].novelty.length > 0); // probe hit the stubbed APIs
    assert.equal(final.debate.verdicts.find((v) => v.hypothesisId === "H1").verdict, "supported");
    assert.equal(final.debate.verdicts.find((v) => v.hypothesisId === "H2").verdict, "contested"); // backfilled
    assert.deepEqual(final.debate.disagreements, ["Scaling beyond 1B is unresolved"]);
    assert.equal(final.protocols.length, 1);
    assert.ok(final.report.includes("[1]"));
    assert.ok(!final.report.includes("[9]"));

    // Agent activity: every named agent reported start and done, in order.
    const agentFlow = events.filter((e) => e.type === "agent").map((e) => `${e.agent}:${e.status}`);
    for (const agent of ["director", "scout", "analyst", "ideator", "skeptic", "proponent", "arbiter", "methodologist", "writer"]) {
      assert.ok(agentFlow.includes(`${agent}:start`) && agentFlow.includes(`${agent}:done`), `missing agent events for ${agent}`);
      assert.ok(LAB_AGENTS[agent]);
    }
    const phases = events.filter((e) => e.type === "phase").map((e) => e.phase);
    assert.deepEqual(phases, ["plan", "literature", "analyze", "hypothesize", "debate", "protocol", "report"]);
    assert.ok(events.some((e) => e.type === "tool" && e.name === "search_literature"));
    assert.ok(events.some((e) => e.type === "tool" && e.name === "novelty_probe"));
    assert.ok(events.some((e) => e.type === "partial" && e.key === "stats" && e.value.verifiedQuotes === 1));
  } finally {
    globalThis.fetch = originalFetch;
  }
});
