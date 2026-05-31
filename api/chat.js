const DEFAULT_MODEL = "anthropic/claude-3.5-sonnet";
const MAX_TOKENS_LIMIT = 8000;

function sendJson(res, status, body) {
  res.status(status).setHeader("Content-Type", "application/json");
  res.end(JSON.stringify(body));
}

function sanitizeMessages(messages) {
  if (!Array.isArray(messages)) return [];
  return messages
    .filter(message =>
      message &&
      ["system", "user", "assistant"].includes(message.role) &&
      typeof message.content === "string"
    )
    .slice(-24)
    .map(message => ({
      role: message.role,
      content: message.content.slice(0, 12000),
    }));
}

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return sendJson(res, 405, { error: "Method not allowed" });
  }

  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) {
    return sendJson(res, 500, { error: "OPENROUTER_API_KEY is not configured on the server." });
  }

  let body = req.body || {};
  if (typeof body === "string") {
    try {
      body = JSON.parse(body);
    } catch (_) {
      return sendJson(res, 400, { error: "Request body must be valid JSON." });
    }
  }

  const messages = sanitizeMessages(body.messages);
  if (messages.length === 0) {
    return sendJson(res, 400, { error: "Request body must include chat messages." });
  }

  const model = String(body.model || process.env.OPENROUTER_MODEL || DEFAULT_MODEL).trim();
  const requestedMax = Number.parseInt(body.maxTokens ?? body.max_tokens ?? MAX_TOKENS_LIMIT, 8000);
  const maxTokens = Math.max(1, Math.min(Number.isFinite(requestedMax) ? requestedMax : MAX_TOKENS_LIMIT, MAX_TOKENS_LIMIT));

  try {
    const upstream = await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${apiKey}`,
        "HTTP-Referer": req.headers.origin || "https://vercel.app",
        "X-Title": "PaperPulse",
      },
      body: JSON.stringify({
        model,
        max_tokens: maxTokens,
        messages,
      }),
    });

    const text = await upstream.text();
    let data = null;
    try { data = text ? JSON.parse(text) : null; } catch (_) {}

    if (!upstream.ok) {
      return sendJson(res, upstream.status, {
        error: data?.error?.message || data?.error || `OpenRouter API status ${upstream.status}`,
      });
    }

    return sendJson(res, 200, data || {});
  } catch (error) {
    return sendJson(res, 502, { error: error.message || "OpenRouter request failed." });
  }
}
