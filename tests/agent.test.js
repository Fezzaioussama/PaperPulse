import test from "node:test";
import assert from "node:assert/strict";
import handler, { writeEvent } from "../api/agent.js";

function fakeRes() {
  return {
    chunks: [],
    status(code) { this.code = code; return this; },
    setHeader() { return this; },
    write(chunk) { this.chunks.push(String(chunk)); return true; },
    end(body) { if (body) this.body = JSON.parse(body); },
  };
}

const request = async ({ method = "POST", headers = {}, body } = {}) => {
  const res = fakeRes();
  await handler({ method, headers, body, on() {} }, res);
  return res;
};

test("agent endpoint enforces auth and input validation before any provider request", async () => {
  const originalFetch = globalThis.fetch;
  const originalEnv = { ...process.env };
  let calls = 0;
  globalThis.fetch = async () => {
    calls++;
    return { ok: true, text: async () => "" };
  };
  const validResearch = { action: "research", topic: "state space models" };
  try {
    process.env.OPENROUTER_API_KEY = "test-only-provider-key";

    // Method and auth gates, identical to /api/chat.
    delete process.env.PAPERPULSE_ACCESS_TOKEN;
    assert.equal((await request({ method: "GET" })).code, 405);
    assert.equal((await request({ body: validResearch })).code, 503);
    process.env.PAPERPULSE_ACCESS_TOKEN = "short";
    assert.equal((await request({ headers: { authorization: "Bearer short" }, body: validResearch })).code, 503);
    process.env.PAPERPULSE_ACCESS_TOKEN = "a".repeat(64);
    assert.equal((await request({ body: validResearch })).code, 401);
    assert.equal((await request({ headers: { authorization: "Bearer wrong" }, body: validResearch })).code, 401);
    assert.equal(calls, 0);

    // A user's own OpenRouter key passes auth even with no server token set.
    delete process.env.PAPERPULSE_ACCESS_TOKEN;
    const viaKey = await request({
      headers: { "x-openrouter-key": `sk-or-v1-${"b".repeat(40)}` },
      body: { action: "nonsense" },
    });
    assert.equal(viaKey.code, 400); // reached validation, not the 503 auth wall
    assert.match(viaKey.body.error, /action/);

    // Input validation (authorized requests).
    process.env.PAPERPULSE_ACCESS_TOKEN = "a".repeat(64);
    const authorization = `Bearer ${process.env.PAPERPULSE_ACCESS_TOKEN}`;
    const auth = (body) => request({ headers: { authorization }, body });
    assert.equal((await auth("null")).code, 400);
    assert.equal((await auth({})).code, 400);
    assert.equal((await auth({ action: "research" })).code, 400);
    assert.equal((await auth({ action: "research", topic: "ab" })).code, 400);
    assert.equal((await auth({ action: "chat", messages: [] })).code, 400);
    const bigKnowledge = { runs: [{ topic: "x".repeat(320 * 1024) }] };
    assert.equal(
      (await auth({ action: "chat", messages: [{ role: "user", content: "hi" }], knowledge: bigKnowledge })).code,
      413
    );

    // compose validation
    const okMaterials = { groups: [{ name: "g", items: [{ text: "an idea" }] }] };
    assert.equal((await auth({ action: "compose", goal: "my article", materials: okMaterials })).code, 400); // no mode
    assert.equal((await auth({ action: "compose", mode: "draft", goal: "ab", materials: okMaterials })).code, 400); // short goal
    assert.equal((await auth({ action: "compose", mode: "ideate", goal: "my article" })).code, 400); // no materials
    assert.equal((await auth({ action: "compose", mode: "draft", goal: "my article", materials: { groups: [], sources: [] } })).code, 400); // empty board
    const bigMaterials = { groups: [{ name: "g", items: [{ text: "x".repeat(220 * 1024) }] }] };
    assert.equal((await auth({ action: "compose", mode: "draft", goal: "my article", materials: bigMaterials })).code, 413);

    // lab validation
    assert.equal((await auth({ action: "lab" })).code, 400);
    assert.equal((await auth({ action: "lab", objective: "ab" })).code, 400);
    assert.equal(calls, 0);
  } finally {
    globalThis.fetch = originalFetch;
    for (const key of ["OPENROUTER_API_KEY", "PAPERPULSE_ACCESS_TOKEN"]) {
      if (originalEnv[key] === undefined) delete process.env[key];
      else process.env[key] = originalEnv[key];
    }
  }
});

const DISCOVER_ARXIV_XML = `<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2401.00001v1</id><title>Selective SSM Memory</title><summary>Selective state space models for memory efficiency.</summary><published>2024-01-05T00:00:00Z</published><author><name>A. Gu</name></author><link rel="alternate" href="http://arxiv.org/abs/2401.00001v1"/><category term="cs.LG"/></entry></feed>`;

test("discover validates the query before any provider fetch, then returns merged JSON", async () => {
  const originalFetch = globalThis.fetch;
  const originalEnv = { ...process.env };
  let calls = 0;
  globalThis.fetch = async (url) => {
    calls++;
    const u = String(url);
    if (u.startsWith("https://export.arxiv.org/")) return { ok: true, text: async () => DISCOVER_ARXIV_XML };
    if (u.includes("semanticscholar.org")) return { ok: true, json: async () => ({ data: [] }) };
    if (u.includes("openalex.org")) return { ok: true, json: async () => ({ results: [] }) };
    throw new Error(`unexpected fetch: ${u}`);
  };
  try {
    process.env.OPENROUTER_API_KEY = "test-only-provider-key";
    process.env.PAPERPULSE_ACCESS_TOKEN = "a".repeat(64);
    const authorization = `Bearer ${process.env.PAPERPULSE_ACCESS_TOKEN}`;
    const auth = (body) => request({ headers: { authorization }, body });

    // Invalid queries are rejected before any provider request.
    assert.equal((await auth({ action: "discover" })).code, 400);
    assert.equal((await auth({ action: "discover", query: "x" })).code, 400);
    assert.equal((await auth({ action: "discover", query: "q".repeat(301) })).code, 400);
    assert.equal(calls, 0);

    // Happy path: plain JSON (not SSE) with merged papers, per-source counts,
    // and the normalized filters echoed back (junk fields dropped).
    const res = await auth({
      action: "discover",
      query: "state space models",
      filters: { sort: "bogus", openAccess: "yes", perSource: 2 },
    });
    assert.equal(res.code, 200);
    assert.equal(calls, 3); // one fetch per provider, no LLM call
    assert.deepEqual(res.body.counts, { arxiv: 1, semanticscholar: 0, openalex: 0 });
    assert.deepEqual(res.body.filters, { perSource: 2 });
    assert.equal(res.body.papers.length, 1);
    assert.equal(res.body.papers[0].title, "Selective SSM Memory");
    assert.equal(res.body.papers[0].pdfUrl, "https://arxiv.org/pdf/2401.00001");
  } finally {
    globalThis.fetch = originalFetch;
    for (const key of ["OPENROUTER_API_KEY", "PAPERPULSE_ACCESS_TOKEN"]) {
      if (originalEnv[key] === undefined) delete process.env[key];
      else process.env[key] = originalEnv[key];
    }
  }
});

test("collection validates name, papers shape, and payload size before any model call", async () => {
  const originalFetch = globalThis.fetch;
  const originalEnv = { ...process.env };
  let calls = 0;
  globalThis.fetch = async () => {
    calls++;
    return { ok: true, text: async () => "", json: async () => ({}) };
  };
  try {
    process.env.OPENROUTER_API_KEY = "test-only-provider-key";
    process.env.PAPERPULSE_ACCESS_TOKEN = "a".repeat(64);
    const authorization = `Bearer ${process.env.PAPERPULSE_ACCESS_TOKEN}`;
    const auth = (body) => request({ headers: { authorization }, body });
    const okPapers = [{ title: "Paper A" }, { title: "Paper B" }];

    assert.equal((await auth({ action: "collection", papers: okPapers })).code, 400); // no name
    assert.equal((await auth({ action: "collection", name: "n".repeat(121), papers: okPapers })).code, 400);
    assert.equal((await auth({ action: "collection", name: "Mine" })).code, 400); // no papers
    assert.equal((await auth({ action: "collection", name: "Mine", papers: [{ title: "only one" }] })).code, 400);
    assert.equal(
      (await auth({ action: "collection", name: "Mine", papers: Array.from({ length: 17 }, (_, i) => ({ title: `p${i}` })) })).code,
      400
    );
    assert.equal((await auth({ action: "collection", name: "Mine", papers: [{ title: "a" }, { abstract: "no title" }] })).code, 400);
    const oversized = [
      { title: "a", abstract: "x".repeat(200 * 1024) },
      { title: "b", abstract: "y".repeat(200 * 1024) },
    ];
    assert.equal((await auth({ action: "collection", name: "Mine", papers: oversized })).code, 413);
    assert.equal(calls, 0);
  } finally {
    globalThis.fetch = originalFetch;
    for (const key of ["OPENROUTER_API_KEY", "PAPERPULSE_ACCESS_TOKEN"]) {
      if (originalEnv[key] === undefined) delete process.env[key];
      else process.env[key] = originalEnv[key];
    }
  }
});

test("writeEvent frames events as data-only SSE", () => {
  const res = fakeRes();
  writeEvent(res, { type: "phase", phase: "plan" });
  writeEvent(res, { type: "done" });
  assert.deepEqual(res.chunks, [
    'data: {"type":"phase","phase":"plan"}\n\n',
    'data: {"type":"done"}\n\n',
  ]);
});
