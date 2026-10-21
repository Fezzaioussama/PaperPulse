import test from "node:test";
import assert from "node:assert/strict";
import { isSafeUrl, extractHtmlTitle } from "../api/_lib/ingest.js";
import handler from "../api/agent.js";

test("isSafeUrl allows public http(s) and blocks internal targets", () => {
  assert.equal(isSafeUrl("https://example.com/article"), true);
  assert.equal(isSafeUrl("http://sub.example.org/a?b=c"), true);
  for (const bad of [
    "ftp://example.com/x",
    "file:///etc/passwd",
    "https://example.com:8080/x",       // non-default port
    "http://user:pass@example.com/",    // credentials
    "http://localhost/admin",
    "http://intranet/",                 // no dot
    "http://foo.local/",
    "http://metadata.internal/",
    "http://127.0.0.1/x",
    "http://0.0.0.0/",
    "http://10.1.2.3/",
    "http://192.168.1.5/",
    "http://172.20.0.1/",
    "http://169.254.169.254/latest/meta-data",
    "http://100.100.1.1/",
    "http://224.0.0.1/",
    "http://[::1]/",
    "not a url",
    "",
  ]) {
    assert.equal(isSafeUrl(bad), false, `should block: ${bad}`);
  }
});

test("extractHtmlTitle pulls and compacts the title tag", () => {
  assert.equal(extractHtmlTitle("<html><head><title>  A\n  Title </title></head></html>"), "A Title");
  assert.equal(extractHtmlTitle("<p>no title</p>"), "");
});

test("ingest action validates input and never fetches unsafe URLs", async () => {
  const originalFetch = globalThis.fetch;
  const originalEnv = { ...process.env };
  let calls = 0;
  globalThis.fetch = async () => { calls++; return { ok: false, status: 404, text: async () => "", headers: new Map() }; };
  const request = async (body) => {
    const res = {
      status(code) { this.code = code; return this; },
      setHeader() { return this; },
      end(payload) { if (payload) this.body = JSON.parse(payload); },
      write() { return true; },
    };
    await handler({
      method: "POST",
      headers: { authorization: `Bearer ${process.env.PAPERPULSE_ACCESS_TOKEN}` },
      body,
      on() {},
    }, res);
    return res;
  };
  try {
    process.env.PAPERPULSE_ACCESS_TOKEN = "a".repeat(64);
    process.env.OPENROUTER_API_KEY = "test-only-provider-key";
    assert.equal((await request({ action: "ingest" })).code, 400);
    assert.equal((await request({ action: "ingest", url: "x".repeat(3000) })).code, 400);
    const unsafe = await request({ action: "ingest", url: "http://169.254.169.254/latest/meta-data" });
    assert.equal(unsafe.code, 422);
    assert.equal(calls, 0); // the unsafe URL was never fetched
  } finally {
    globalThis.fetch = originalFetch;
    for (const key of ["OPENROUTER_API_KEY", "PAPERPULSE_ACCESS_TOKEN"]) {
      if (originalEnv[key] === undefined) delete process.env[key];
      else process.env[key] = originalEnv[key];
    }
  }
});
