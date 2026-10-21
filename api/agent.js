import { randomUUID } from "node:crypto";
import { sendJson, sanitizeMessages, checkAccess } from "./_lib/auth.js";
import { fetchSourceText } from "./_lib/ingest.js";
import { resolveModel, makeModel } from "./_lib/model.js";
import {
  buildResearchGraph, buildChatGraph, buildChatSystemPrompt, resolveLimits,
} from "./_lib/graph.js";

// Without this flag Vercel buffers the whole response and SSE silently
// degrades to one blob delivered at the end.
export const config = { supportsResponseStreaming: true };

const MAX_BODY_BYTES = 1.5 * 1024 * 1024;
const MAX_KNOWLEDGE_BYTES = 300 * 1024;
const TOPIC_MIN = 3;
const TOPIC_MAX = 300;

export function writeEvent(res, obj) {
  res.write(`data: ${JSON.stringify(obj)}\n\n`);
}

// Never echo upstream bodies or keys (parity with chat.js error handling).
function safeErrorMessage(error) {
  const status = error?.status ?? error?.response?.status;
  if (Number.isInteger(status)) return `OpenRouter API status ${status}`;
  const raw = String(error?.message || "Agent request failed.");
  return raw.replace(/sk-or-[\w-]+/g, "sk-or-***").slice(0, 200);
}

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return sendJson(res, 405, { error: "Method not allowed" });
  }
  res.setHeader("Cache-Control", "no-store");

  const access = checkAccess(req, { allowBrowserKey: true });
  if (!access.ok) {
    return sendJson(res, access.status, { error: access.error });
  }

  let body = req.body || {};
  if (typeof body === "string") {
    if (body.length > MAX_BODY_BYTES) return sendJson(res, 413, { error: "Request body is too large." });
    try { body = JSON.parse(body); } catch (_) {
      return sendJson(res, 400, { error: "Request body must be valid JSON." });
    }
  }
  if (!body || typeof body !== "object" || Array.isArray(body)) {
    return sendJson(res, 400, { error: "Request body must be a JSON object." });
  }

  const action = body.action;
  if (action !== "research" && action !== "chat" && action !== "ingest") {
    return sendJson(res, 400, { error: 'action must be "research", "chat", or "ingest".' });
  }

  // ingest: fetch one source (arXiv id or safe public URL) and return its
  // readable text for the client-side RAG library. Plain JSON, no SSE.
  if (action === "ingest") {
    const url = typeof body.url === "string" ? body.url.trim() : "";
    if (!url || url.length > 2048) {
      return sendJson(res, 400, { error: "ingest requires a url (arXiv id or http(s) URL)." });
    }
    const source = await fetchSourceText(url);
    if (!source) {
      return sendJson(res, 422, { error: "Could not fetch readable text from that source." });
    }
    return sendJson(res, 200, source);
  }

  let topic = "";
  let messages = [];
  let knowledge = null;
  if (action === "research") {
    topic = typeof body.topic === "string" ? body.topic.trim() : "";
    if (topic.length < TOPIC_MIN || topic.length > TOPIC_MAX) {
      return sendJson(res, 400, { error: `topic must be a string of ${TOPIC_MIN}-${TOPIC_MAX} characters.` });
    }
  } else {
    messages = sanitizeMessages(body.messages);
    if (messages.length === 0) {
      return sendJson(res, 400, { error: "Request body must include at least one chat message." });
    }
    knowledge = body.knowledge && typeof body.knowledge === "object" && !Array.isArray(body.knowledge)
      ? body.knowledge
      : { runs: [] };
    if (JSON.stringify(knowledge).length > MAX_KNOWLEDGE_BYTES) {
      return sendJson(res, 413, { error: "knowledge payload is too large." });
    }
  }

  const limits = resolveLimits(body.options);
  const model = makeModel({
    apiKey: access.apiKey,
    model: resolveModel(access.viaBrowserKey, body.options?.model),
  });

  // ── Open the SSE stream ──
  res.status(200);
  res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("X-Accel-Buffering", "no");

  const ac = new AbortController();
  req.on?.("close", () => ac.abort());
  const heartbeat = setInterval(() => { try { res.write(": ping\n\n"); } catch (_) {} }, 15000);
  const emit = (event) => { try { writeEvent(res, event); } catch (_) {} };

  try {
    if (action === "research") {
      const startedAt = new Date().toISOString();
      const graph = buildResearchGraph();
      const final = await graph.invoke(
        { topic, limits },
        { configurable: { model, emit, signal: ac.signal }, signal: ac.signal, recursionLimit: 25 }
      );
      emit({
        type: "result",
        run: {
          id: randomUUID(),
          topic,
          queries: final.queries,
          papers: final.relevant,
          ideas: final.ideas,
          themes: final.themes,
          concepts: final.concepts,
          critique: final.critique,
          practice: final.practice,
          comparison: final.comparison,
          figures: final.figures,
          report: final.report,
          startedAt,
          finishedAt: new Date().toISOString(),
        },
      });
    } else {
      const graph = buildChatGraph({ model, knowledge });
      const input = { messages: [{ role: "system", content: buildChatSystemPrompt(knowledge) }, ...messages] };
      let finalMessage = "";
      const stream = await graph.stream(input, {
        configurable: { signal: ac.signal }, signal: ac.signal, streamMode: "updates", recursionLimit: 14,
      });
      for await (const update of stream) {
        const agentMessages = update.agent?.messages || [];
        for (const m of agentMessages) {
          if (m.tool_calls?.length) {
            for (const call of m.tool_calls) emit({ type: "tool", name: call.name, args: call.args });
          } else if (typeof m.content === "string" && m.content.trim()) {
            finalMessage = m.content.trim();
          }
        }
      }
      emit({ type: "result", message: finalMessage || "I could not produce an answer. Try rephrasing the question." });
    }
  } catch (error) {
    if (!ac.signal.aborted) {
      emit({ type: "error", error: safeErrorMessage(error) });
    }
  } finally {
    clearInterval(heartbeat);
    try { writeEvent(res, { type: "done" }); } catch (_) {}
    try { res.end(); } catch (_) {}
  }
}
