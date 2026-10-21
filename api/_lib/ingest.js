import { htmlToTextNode, fetchPaperTextNode } from "./arxiv.js";
import { compactText, normId } from "./text.js";

const TEXT_LIMIT = 60000;

// SSRF guard for the ingest action: the server only fetches plain http(s)
// URLs on default ports, and never anything that smells like an internal
// host. (DNS rebinding is out of scope for this app — see README security.)
export function isSafeUrl(raw) {
  let u;
  try { u = new URL(String(raw || "")); } catch (_) { return false; }
  if (u.protocol !== "http:" && u.protocol !== "https:") return false;
  if (u.port && u.port !== "80" && u.port !== "443") return false;
  if (u.username || u.password) return false;
  const host = u.hostname.toLowerCase();
  if (!host.includes(".")) return false;                       // localhost, bare names
  if (host.endsWith(".local") || host.endsWith(".internal")) return false;
  if (/^\[/.test(host) || host.includes(":")) return false;    // IPv6 literals
  const ipv4 = host.match(/^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/);
  if (ipv4) {
    const [a, b] = [Number(ipv4[1]), Number(ipv4[2])];
    if (a === 0 || a === 10 || a === 127 || a >= 224) return false;
    if (a === 169 && b === 254) return false;
    if (a === 172 && b >= 16 && b <= 31) return false;
    if (a === 192 && b === 168) return false;
    if (a === 100 && b >= 64 && b <= 127) return false;
  }
  return true;
}

export function extractHtmlTitle(html) {
  const m = String(html || "").match(/<title[^>]*>([\s\S]{0,300}?)<\/title>/i);
  return m ? compactText(m[1]).slice(0, 200) : "";
}

// Fetch a source for the knowledge library. An arXiv id/URL uses the paper
// pipeline; anything else is fetched once and stripped to readable text.
export async function fetchSourceText(rawUrl) {
  const id = normId(String(rawUrl || ""));
  if (/^\d{4}\.\d{4,5}$/.test(id)) {
    const text = await fetchPaperTextNode(id);
    if (!text) return null;
    return { title: `arXiv:${id}`, text, origin: "arxiv" };
  }
  if (!isSafeUrl(rawUrl)) return null;
  try {
    const res = await fetch(rawUrl, {
      signal: AbortSignal.timeout(12000),
      redirect: "follow",
      headers: { "User-Agent": "PaperPulse/1.0 (+research briefing tool)", "Accept": "text/html, text/plain" },
    });
    if (!res.ok) return null;
    const type = res.headers.get("content-type") || "";
    if (!/text\/html|text\/plain|application\/xhtml/.test(type)) return null;
    const raw = (await res.text()).slice(0, 800000);
    const isHtml = /text\/html|xhtml/.test(type);
    const text = isHtml ? htmlToTextNode(raw) : compactText(raw);
    if (!text || text.length < 200) return null;
    return {
      title: (isHtml && extractHtmlTitle(raw)) || new URL(rawUrl).hostname,
      text: text.slice(0, TEXT_LIMIT),
      origin: "url",
    };
  } catch (_) {
    return null;
  }
}
