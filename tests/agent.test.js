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
