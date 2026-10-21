// Pure text helpers shared by the agent graph. Ported from src/App.jsx
// (which deliberately keeps everything module-local in one file), so the
// client stays untouched and the server never imports browser code.

export function extractJson(raw) {
  if (!raw) return null;
  let s = String(raw).replace(/```json/gi, "").replace(/```/g, "").trim();
  try { return JSON.parse(s); } catch (_) {}
  const start = s.indexOf("[") !== -1 ? s.indexOf("[") : s.indexOf("{");
  const lastArr = s.lastIndexOf("]");
  const lastObj = s.lastIndexOf("}");
  const end = Math.max(lastArr, lastObj);
  if (start !== -1 && end > start) {
    try { return JSON.parse(s.slice(start, end + 1)); } catch (_) {}
  }
  return null;
}

export function compactText(value = "") {
  return String(value).replace(/\s+/g, " ").trim();
}

export function clipText(value = "", max = 1800) {
  const text = compactText(value);
  if (text.length <= max) return text;
  const boundary = Math.max(text.lastIndexOf(". ", max - 1), text.lastIndexOf("; ", max - 1));
  const end = boundary > max * 0.65 ? boundary + 1 : max;
  return text.slice(0, end).trim() + " ...";
}

export function normalizeForMatch(value = "") {
  return String(value)
    .toLowerCase()
    .replace(/['"`]/g, "")
    .replace(/[^a-z0-9+#.]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

export function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function tokenVariants(token) {
  const variants = [token];
  if (token.endsWith("ies") && token.length > 4) {
    variants.push(`${token.slice(0, -3)}y`);
  } else if (token.endsWith("s") && token.length > 4 && !token.endsWith("ss") && !token.endsWith("is")) {
    variants.push(token.slice(0, -1));
  } else if (token.length > 2) {
    variants.push(`${token}s`);
  }
  return variants;
}

export function hasMatch(haystack, value) {
  const needle = normalizeForMatch(value);
  if (!needle) return false;
  if (needle.includes(" ")) return haystack.includes(needle);
  return tokenVariants(needle).some(token =>
    new RegExp(`(^|\\s)${escapeRegExp(token)}(?=\\s|$)`).test(haystack)
  );
}

export function normId(url = "") {
  const m = url.match(/(\d{4}\.\d{4,5})(v\d+)?/);
  return m ? m[1] : url.toLowerCase().trim();
}

const STOPWORDS = new Set([
  "the", "and", "for", "with", "that", "this", "from", "into", "are", "was",
  "has", "have", "can", "will", "about", "using", "use", "via", "how", "what",
  "when", "where", "which", "their", "them", "then", "than", "its", "over",
]);

export function topicTokens(value = "") {
  return normalizeForMatch(value)
    .split(" ")
    .filter(t => t.length >= 3 && !STOPWORDS.has(t));
}

// Slim lexical scorer (the topical half of App.jsx scorePaperForTopic):
// used only to prefilter candidates so the LLM grading batch stays small.
export function lexicalTopicScore(paper, profile) {
  const title = normalizeForMatch(paper.title);
  const abstract = normalizeForMatch(paper.abstract);
  let score = 0;
  for (const phrase of profile.phrases || []) {
    if (hasMatch(title, phrase)) score += 18;
    else if (hasMatch(abstract, phrase)) score += 8;
  }
  for (const token of profile.tokens || []) {
    if (hasMatch(title, token)) score += 5;
    else if (hasMatch(abstract, token)) score += 2;
  }
  return score;
}

export function buildTopicLexicalProfile(topic = "") {
  const raw = compactText(topic);
  return {
    phrases: raw && normalizeForMatch(raw).includes(" ") ? [raw] : [],
    tokens: topicTokens(raw),
  };
}
