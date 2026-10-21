import { Annotation, StateGraph, START, END } from "@langchain/langgraph";
import { callJson, sanitizeReportCitations } from "./graph.js";
import { clipText, compactText } from "./text.js";

// ── Writing studio: turn the user's regrouped idea board into article angles
//    (ideate) or a full drafted article (outline → draft sections → polish).
//    LLM budget: ideate = 1 call; draft ≤ 1 + maxSections + 1 = 10 calls. ──

const noop = () => {};

export const ARTICLE_LENGTHS = {
  short: { maxSections: 4, words: 200 },
  medium: { maxSections: 6, words: 300 },
  long: { maxSections: 8, words: 400 },
};

export function resolveStyle(raw = {}) {
  return {
    audience: clipText(raw?.audience || "a technically curious general reader", 80),
    tone: clipText(raw?.tone || "clear and engaging", 80),
    length: ARTICLE_LENGTHS[raw?.length] ? raw.length : "medium",
  };
}

// Clamp the client's idea-board payload: numbered sources the article can
// cite, plus the user's own grouping of idea cards (their regrouping IS the
// signal — prompts respect it rather than re-clustering).
export function normalizeMaterials(raw) {
  const sources = (Array.isArray(raw?.sources) ? raw.sources : [])
    .filter((s) => s && typeof s === "object" && typeof s.title === "string" && s.title.trim())
    .slice(0, 40)
    .map((s) => ({
      title: clipText(s.title, 200),
      kind: clipText(s.kind || "source", 40),
      url: typeof s.url === "string" ? s.url.slice(0, 400) : "",
      date: typeof s.date === "string" ? s.date.slice(0, 10) : "",
    }));
  let budget = 80; // global card cap across all groups
  const groups = [];
  for (const g of (Array.isArray(raw?.groups) ? raw.groups : []).slice(0, 10)) {
    if (!g || typeof g !== "object" || budget <= 0) continue;
    const items = (Array.isArray(g.items) ? g.items : [])
      .filter((it) => it && typeof it.text === "string" && it.text.trim())
      .slice(0, budget)
      .map((it) => ({
        text: clipText(it.text, 400),
        kind: clipText(it.kind || "idea", 30),
        source: Number.isInteger(it.source) && it.source >= 1 && it.source <= sources.length ? it.source : 0,
      }));
    if (!items.length) continue;
    budget -= items.length;
    groups.push({ name: clipText(g.name || "Ideas", 80), items });
  }
  return { sources, groups };
}

// The prompt block every compose call is grounded in.
export function materialsDigest(materials) {
  const src = materials.sources
    .map((s, i) => `[${i + 1}] ${s.kind}: ${compactText(s.title)}${s.date ? ` (${s.date})` : ""}`)
    .join("\n");
  const groups = materials.groups
    .map((g) => `Group "${g.name}":\n${g.items
      .map((it) => `- (${it.kind})${it.source ? ` [${it.source}]` : ""} ${it.text}`)
      .join("\n")}`)
    .join("\n\n");
  return `${src ? `Sources (cite with [n]):\n${src}\n\n` : ""}${groups || "(no idea cards)"}`;
}

// Validate an outline — from the planner OR straight from the client (the
// user may write their own; sections may be plain heading strings).
export function normalizeOutline(parsed, maxSections = 8) {
  if (!parsed || typeof parsed !== "object") return null;
  const sections = (Array.isArray(parsed.sections) ? parsed.sections : [])
    .map((s) => (typeof s === "string" ? { heading: s, points: [] } : s))
    .filter((s) => s && typeof s.heading === "string" && s.heading.trim())
    .slice(0, maxSections)
    .map((s) => ({
      heading: clipText(s.heading, 120),
      points: (Array.isArray(s.points) ? s.points : []).map((p) => clipText(p, 250)).filter(Boolean).slice(0, 5),
    }));
  if (!sections.length) return null;
  return { title: clipText(parsed.title || "", 160), sections };
}

// Validate the ideate output; drop angles with no title/thesis and source
// refs outside the numbered list.
export function normalizeAngles(parsed, sourceCount = 0) {
  const list = Array.isArray(parsed?.angles) ? parsed.angles : Array.isArray(parsed) ? parsed : [];
  return list
    .filter((a) => a && typeof a === "object"
      && typeof a.title === "string" && a.title.trim()
      && typeof a.thesis === "string" && a.thesis.trim())
    .slice(0, 6)
    .map((a) => ({
      title: clipText(a.title, 140),
      thesis: clipText(a.thesis, 400),
      angle: clipText(a.angle || "", 300),
      outline: (Array.isArray(a.outline) ? a.outline : []).map((h) => clipText(h, 120)).filter(Boolean).slice(0, 8),
      sources: [...new Set(
        (Array.isArray(a.sources) ? a.sources : [])
          .map((n) => Number(n))
          .filter((n) => Number.isInteger(n) && n >= 1 && n <= sourceCount)
      )].slice(0, 12),
    }));
}

// ── Ideate: one call proposing article angles the board could support. ──
export async function ideateAngles({ goal, materials }, config) {
  const { model, emit = noop } = config.configurable;
  emit({ type: "phase", phase: "ideate", label: "Generating article angles" });
  const prompt = `You are an editor brainstorming article ideas with an author.

The author's interest: "${goal}"

Material the author collected and regrouped into groups (sources have [n] ids):

${materialsDigest(materials)}

Propose 3-5 DISTINCT article angles this material could support. An angle is a specific, arguable take — not a topic. Prefer angles that CONNECT ideas across different groups or sources: a tension between them, a trend they jointly show, a practical playbook, a contrarian case. For each angle give a working title, a one-sentence thesis, why it is worth writing now, a 3-6 item section outline, and the [n] source ids it draws on.

Respond with ONLY a JSON object (no prose, no markdown fences):
{"angles": [{"title": "...", "thesis": "...", "angle": "why now", "outline": ["section heading", ...], "sources": [1, 2]}]}`;
  try {
    const parsed = await callJson(model, prompt, config);
    const angles = normalizeAngles(parsed, materials.sources.length);
    if (angles.length) {
      emit({ type: "partial", key: "angles", value: angles });
      return angles;
    }
  } catch (err) {
    if (config.configurable.signal?.aborted) throw err;
  }
  emit({ type: "log", message: "Angle generation produced nothing usable. Add a few more idea cards and retry." });
  return [];
}

// ── Draft graph state ──

const ComposeState = Annotation.Root({
  goal: Annotation(),
  style: Annotation(),
  materials: Annotation(),
  outline: Annotation({ reducer: (a = null, b = null) => b ?? a, default: () => null }),
  sections: Annotation({ reducer: (a = [], b = []) => (b.length ? b : a), default: () => [] }),
  article: Annotation({ reducer: (a = "", b = "") => b || a, default: () => "" }),
});

async function planOutline(state, config) {
  const { model, emit = noop } = config.configurable;
  const { maxSections } = ARTICLE_LENGTHS[state.style.length];
  // The user's own outline wins; the planner only fills the gap.
  if (state.outline?.sections?.length) {
    emit({ type: "phase", phase: "outline", label: "Using your outline" });
    emit({ type: "partial", key: "outline", value: state.outline });
    return {};
  }
  emit({ type: "phase", phase: "outline", label: "Planning the article outline" });
  const prompt = `You are planning an article.

Working title / goal: "${state.goal}"
Audience: ${state.style.audience}. Tone: ${state.style.tone}.

Material the author regrouped into groups (sources have [n] ids):

${materialsDigest(state.materials)}

Plan an article of at most ${maxSections} sections that builds one ARGUMENT from this material — not a list of summaries. The groups are the author's own organization of the ideas; respect it. For each section give a heading and 2-4 bullet points of what it will argue, naming [n] source ids where relevant.

Respond with ONLY a JSON object (no prose, no markdown fences):
{"title": "...", "sections": [{"heading": "...", "points": ["..."]}]}`;
  try {
    const parsed = await callJson(model, prompt, config);
    const outline = normalizeOutline(parsed, maxSections);
    if (outline) {
      emit({ type: "partial", key: "outline", value: outline });
      return { outline };
    }
  } catch (err) {
    if (config.configurable.signal?.aborted) throw err;
  }
  // Mechanical fallback: one section per idea group.
  const sections = state.materials.groups.slice(0, maxSections).map((g) => ({ heading: g.name, points: [] }));
  const outline = { title: state.goal, sections: sections.length ? sections : [{ heading: "Overview", points: [] }] };
  emit({ type: "log", message: "Outline planning failed; using one section per idea group." });
  emit({ type: "partial", key: "outline", value: outline });
  return { outline };
}

async function draftSections(state, config) {
  const { model, emit = noop } = config.configurable;
  const { maxSections, words } = ARTICLE_LENGTHS[state.style.length];
  const outline = state.outline;
  const planned = outline.sections.slice(0, maxSections);
  const title = outline.title || state.goal;
  const digest = materialsDigest(state.materials);
  emit({ type: "phase", phase: "draft", label: `Drafting ${planned.length} sections` });
  const sections = [];
  for (let i = 0; i < planned.length; i++) {
    const sec = planned[i];
    const prompt = `You are writing section ${i + 1} of ${planned.length} of an article titled "${title}".
Audience: ${state.style.audience}. Tone: ${state.style.tone}. Target ≈ ${words} words for this section.

Source material (cite facts with [n] ids from this list ONLY):

${digest}

Full outline:
${planned.map((s, j) => `${j + 1}. ${s.heading}`).join("\n")}
${sections.length ? `\nThe article so far ends with:\n...${clipText(sections[sections.length - 1].markdown.slice(-900), 900)}\n` : ""}
Write ONLY section ${i + 1}, "${sec.heading}"${sec.points.length ? `, covering:\n${sec.points.map((p) => `- ${p}`).join("\n")}` : "."}

Rules: start with the markdown heading "## ${sec.heading}"; flow naturally from the previous text; do not repeat earlier sections or add intro/outro boilerplate; ground claims in the sources with [n] citations; respond with markdown only.`;
    try {
      const res = await model.invoke(prompt, { signal: config.configurable.signal });
      const raw = typeof res.content === "string" ? res.content.trim() : "";
      if (raw) {
        const markdown = raw.startsWith("#") ? raw : `## ${sec.heading}\n\n${raw}`;
        sections.push({ heading: sec.heading, markdown: clipText(markdown, 6000) });
        emit({ type: "partial", key: "sections", value: sections });
      }
    } catch (err) {
      if (config.configurable.signal?.aborted) throw err;
      emit({ type: "log", message: `Drafting failed for "${clipText(sec.heading, 60)}"; skipped.` });
    }
    emit({ type: "section_progress", done: i + 1, total: planned.length });
  }
  return { sections };
}

async function polishArticle(state, config) {
  const { model, emit = noop } = config.configurable;
  if (!state.sections.length) {
    emit({ type: "log", message: "No sections were drafted; nothing to polish." });
    return { article: "" };
  }
  emit({ type: "phase", phase: "polish", label: "Polishing the full draft" });
  const title = state.outline?.title || state.goal;
  const stitched = [`# ${title}`, "", ...state.sections.map((s) => s.markdown)].join("\n\n");
  const sourceCount = state.materials.sources.length;
  const prompt = `Polish this article draft into its final form. Keep the structure, headings, facts, and [n] citations exactly as they are. Fix repetition between sections, smooth the transitions, make the opening paragraph hook the reader, and end with a short conclusion that lands the main argument. Audience: ${state.style.audience}. Tone: ${state.style.tone}.

---
${clipText(stitched, 24000)}
---

Respond with ONLY the full polished article in markdown; no preamble, no commentary.`;
  try {
    const res = await model.invoke(prompt, { signal: config.configurable.signal });
    const raw = typeof res.content === "string" ? res.content.trim() : "";
    if (raw) {
      const article = sanitizeReportCitations(raw, sourceCount);
      emit({ type: "partial", key: "article", value: article });
      return { article };
    }
  } catch (err) {
    if (config.configurable.signal?.aborted) throw err;
  }
  // Polish is a bonus pass: the stitched draft is already a usable article.
  const article = sanitizeReportCitations(stitched, sourceCount);
  emit({ type: "log", message: "Polish failed; keeping the stitched draft." });
  emit({ type: "partial", key: "article", value: article });
  return { article };
}

export function buildComposeGraph() {
  return new StateGraph(ComposeState)
    .addNode("plan_outline", planOutline)
    .addNode("draft_sections", draftSections)
    .addNode("polish_article", polishArticle)
    .addEdge(START, "plan_outline")
    .addEdge("plan_outline", "draft_sections")
    .addEdge("draft_sections", "polish_article")
    .addEdge("polish_article", END)
    .compile();
}
