import test from "node:test";
import assert from "node:assert/strict";
import handler from "../api/chat.js";

test("server chat requires private access before making any provider request", async () => {
  const originalFetch = globalThis.fetch;
  const originalEnv = { ...process.env };
  let calls = 0;
  globalThis.fetch = async () => {
    calls++;
    return { ok: true, text: async () => JSON.stringify({ choices: [] }) };
  };
  const request = async (authorization, body = { messages: [{ role: "user", content: "test" }] }) => {
    const res = {
      status(code) { this.code = code; return this; },
      setHeader() { return this; },
      end(body) { this.body = JSON.parse(body); },
    };
    await handler({ method: "POST", headers: { authorization }, body }, res);
    return res;
  };
  try {
    process.env.OPENROUTER_API_KEY = "test-only-provider-key";
    delete process.env.PAPERPULSE_ACCESS_TOKEN;
    assert.equal((await request()).code, 503);
    process.env.PAPERPULSE_ACCESS_TOKEN = "short";
    assert.equal((await request("Bearer short")).code, 503);
    process.env.PAPERPULSE_ACCESS_TOKEN = "a".repeat(64);
    assert.equal((await request()).code, 401);
    assert.equal((await request("Bearer wrong")).code, 401);
    assert.equal(calls, 0);
    const authorization = `Bearer ${process.env.PAPERPULSE_ACCESS_TOKEN}`;
    assert.equal((await request(authorization, "null")).code, 400);
    assert.equal(calls, 0);
    assert.equal((await request(authorization)).code, 200);
    assert.equal(calls, 1);
    globalThis.fetch = async () => ({ ok: false, status: 429, text: async () => JSON.stringify({ error: { message: "private-provider-detail" } }) });
    const failure = await request(authorization);
    assert.equal(failure.code, 429);
    assert.equal(failure.body.error, "OpenRouter API status 429");
  } finally {
    globalThis.fetch = originalFetch;
    for (const key of ["OPENROUTER_API_KEY", "PAPERPULSE_ACCESS_TOKEN"]) {
      if (originalEnv[key] === undefined) delete process.env[key];
      else process.env[key] = originalEnv[key];
    }
  }
});
