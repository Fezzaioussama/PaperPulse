import { createHash, timingSafeEqual } from "node:crypto";

export function sendJson(res, status, body) {
  res.status(status).setHeader("Content-Type", "application/json");
  res.end(JSON.stringify(body));
}

export function sanitizeMessages(messages, { maxMessages = 24, maxContent = 12000 } = {}) {
  if (!Array.isArray(messages)) return [];
  return messages
    .filter(
      (m) =>
        m &&
        ["system", "user", "assistant"].includes(m.role) &&
        typeof m.content === "string" &&
        m.content.trim().length > 0
    )
    .slice(-maxMessages)
    .map((m) => ({ role: m.role, content: m.content.slice(0, maxContent) }));
}

// Resolves which OpenRouter key a request may use.
// - allowBrowserKey: the agent endpoint accepts the user's own key via
//   X-OpenRouter-Key (it is only forwarded to OpenRouter, never stored).
// - Otherwise the request must carry the private access token, and the
//   server's own key is used — identical behavior to the original chat.js.
// Returns { ok: true, apiKey, viaBrowserKey } or { ok: false, status, error }.
export function checkAccess(req, { allowBrowserKey = false } = {}) {
  if (allowBrowserKey) {
    const raw = req.headers?.["x-openrouter-key"];
    const browserKey = typeof raw === "string" ? raw.trim() : "";
    if (/^sk-or-[A-Za-z0-9_-]{20,200}$/.test(browserKey)) {
      return { ok: true, apiKey: browserKey, viaBrowserKey: true };
    }
  }

  const accessToken = process.env.PAPERPULSE_ACCESS_TOKEN;
  if (!accessToken || accessToken.length < 32) {
    return { ok: false, status: 503, error: "Server chat is disabled. Use your own API key in Settings." };
  }
  const authorization = req.headers?.authorization;
  const expected = createHash("sha256").update(`Bearer ${accessToken}`).digest();
  const supplied = createHash("sha256").update(typeof authorization === "string" ? authorization : "").digest();
  if (!timingSafeEqual(expected, supplied)) {
    return { ok: false, status: 401, error: "Enter a valid server access token or your own API key in Settings." };
  }

  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) {
    return { ok: false, status: 500, error: "OPENROUTER_API_KEY is not configured on the server." };
  }
  return { ok: true, apiKey, viaBrowserKey: false };
}
