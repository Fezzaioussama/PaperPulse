import test from "node:test";
import assert from "node:assert/strict";
import {
  normalizeMaterials, materialsDigest, normalizeOutline, normalizeAngles,
  resolveStyle, ideateAngles, buildComposeGraph, ARTICLE_LENGTHS,
} from "../api/_lib/compose.js";

const MATERIALS = normalizeMaterials({
  sources: [
    { title: "State Space Models for Long Context", kind: "paper", url: "http://arxiv.org/abs/2401.00001", date: "2024-01-05" },
    { title: "My reading notes", kind: "note" },
  ],
  groups: [
    { name: "Efficiency", items: [
      { text: "SSMs scale linearly with sequence length", kind: "finding", source: 1 },
      { text: "Attention is quadratic", kind: "claim", source: 7 }, // out-of-range → 0
    ] },
    { name: "Open problems", items: [{ text: "Recall remains weaker than attention", kind: "limitation", source: 2 }] },
  ],
});

test("normalizeMaterials clamps shapes, drops junk, validates source refs", () => {
  assert.equal(MATERIALS.sources.length, 2);
  assert.equal(MATERIALS.groups.length, 2);
  assert.equal(MATERIALS.groups[0].items[0].source, 1);
  assert.equal(MATERIALS.groups[0].items[1].source, 0); // ref outside the source list
  // junk input
  assert.deepEqual(normalizeMaterials(null), { sources: [], groups: [] });
  assert.deepEqual(normalizeMaterials({ groups: [{ name: "x", items: [{ text: "  " }] }] }).groups, []);
  // global card budget across groups
  const many = normalizeMaterials({
    groups: [
      { name: "a", items: Array.from({ length: 70 }, (_, i) => ({ text: `idea ${i}` })) },
      { name: "b", items: Array.from({ length: 70 }, (_, i) => ({ text: `more ${i}` })) },
    ],
  });
  assert.equal(many.groups.reduce((n, g) => n + g.items.length, 0), 80);
});

test("materialsDigest numbers sources and keeps the user's grouping", () => {
  const digest = materialsDigest(MATERIALS);
  assert.ok(digest.includes("[1] paper: State Space Models for Long Context (2024-01-05)"));
  assert.ok(digest.includes('Group "Efficiency":'));
  assert.ok(digest.includes("- (finding) [1] SSMs scale linearly with sequence length"));
  assert.ok(digest.includes("- (claim) Attention is quadratic")); // no bogus [7]
  assert.equal(materialsDigest(normalizeMaterials(null)), "(no idea cards)");
});

test("resolveStyle defaults and rejects unknown lengths", () => {
  assert.deepEqual(resolveStyle(), {
    audience: "a technically curious general reader", tone: "clear and engaging", length: "medium",
  });
  assert.equal(resolveStyle({ length: "epic" }).length, "medium");
  assert.equal(resolveStyle({ length: "long", tone: "opinionated" }).tone, "opinionated");
});

test("normalizeOutline accepts heading strings, clamps sections, rejects junk", () => {
  const o = normalizeOutline({ title: "T", sections: ["Intro", { heading: "Body", points: ["p1", "", "p2"] }, { nope: 1 }] }, 8);
  assert.deepEqual(o, {
    title: "T",
    sections: [{ heading: "Intro", points: [] }, { heading: "Body", points: ["p1", "p2"] }],
  });
  assert.equal(normalizeOutline({ sections: Array.from({ length: 12 }, (_, i) => `S${i}`) }, 4).sections.length, 4);
  assert.equal(normalizeOutline(null), null);
  assert.equal(normalizeOutline({ sections: [] }), null);
});

test("normalizeAngles keeps valid angles and drops out-of-range source refs", () => {
  const angles = normalizeAngles({ angles: [
    { title: "The linear-attention bet", thesis: "SSMs will eat long-context workloads.", angle: "fresh results", outline: ["Setup", "Evidence"], sources: [1, 2, 9, 1] },
    { title: "No thesis here" },
  ] }, 2);
  assert.equal(angles.length, 1);
  assert.deepEqual(angles[0].sources, [1, 2]);
  assert.deepEqual(normalizeAngles("junk", 2), []);
});

test("ideateAngles emits a phase and partial angles from the model JSON", async () => {
  const events = [];
  const model = {
    invoke: async (prompt) => {
      assert.ok(String(prompt).includes("brainstorming article ideas"));
      return { content: '{"angles":[{"title":"A","thesis":"B","outline":["S1"],"sources":[1]}]}' };
    },
  };
  const angles = await ideateAngles(
    { goal: "long context models", materials: MATERIALS },
    { configurable: { model, emit: (e) => events.push(e) } }
  );
  assert.equal(angles.length, 1);
  assert.equal(angles[0].title, "A");
  assert.deepEqual(events.map((e) => e.type), ["phase", "partial"]);
  assert.equal(events[1].key, "angles");
});

function stubComposeModel(routes) {
  return {
    invoke: async (prompt) => {
      const hit = routes.find(([marker]) => String(prompt).includes(marker));
      if (!hit) throw new Error(`unexpected prompt: ${String(prompt).slice(0, 80)}`);
      return { content: hit[1] };
    },
  };
}

test("compose graph drafts an article: outline → sections → polish, with partials", async () => {
  const events = [];
  const model = stubComposeModel([
    ["Plan an article", '{"title":"The Linear Era","sections":[{"heading":"Why attention hit a wall","points":["cost"]},{"heading":"What SSMs change","points":[]}]}'],
    ["Write ONLY section 1", "## Why attention hit a wall\n\nQuadratic cost bites [1]."],
    // No leading heading → the node must prepend it.
    ["Write ONLY section 2", "SSMs scale linearly [1]. Bogus cite [9]."],
    ["Polish this article draft", "# The Linear Era\n\nQuadratic cost bites [1]. SSMs scale linearly [1]. Bogus cite [9]."],
  ]);
  const graph = buildComposeGraph();
  const final = await graph.invoke(
    { goal: "long context models", style: resolveStyle({ length: "short" }), materials: MATERIALS, outline: null },
    { configurable: { model, emit: (e) => events.push(e) }, recursionLimit: 10 }
  );

  assert.equal(final.outline.title, "The Linear Era");
  assert.equal(final.sections.length, 2);
  assert.ok(final.sections[1].markdown.startsWith("## What SSMs change"));
  // Polish keeps real citations and strips ones outside the 2-source list.
  assert.ok(final.article.includes("[1]"));
  assert.ok(!final.article.includes("[9]"));

  const phases = events.filter((e) => e.type === "phase").map((e) => e.phase);
  assert.deepEqual(phases, ["outline", "draft", "polish"]);
  const partials = events.filter((e) => e.type === "partial").map((e) => e.key);
  assert.deepEqual(partials, ["outline", "sections", "sections", "article"]);
  assert.deepEqual(
    events.filter((e) => e.type === "section_progress").map((e) => `${e.done}/${e.total}`),
    ["1/2", "2/2"]
  );
});

test("compose graph respects a user-provided outline and survives a failed polish", async () => {
  const events = [];
  // No "Plan an article" route: reaching the planner would throw.
  const model = stubComposeModel([
    ["Write ONLY section 1", "## My section\n\nHand-outlined content [2]."],
    ["Polish this article draft", ""], // empty → fallback to the stitched draft
  ]);
  const graph = buildComposeGraph();
  const outline = normalizeOutline({ title: "Hand-made", sections: ["My section"] }, ARTICLE_LENGTHS.short.maxSections);
  const final = await graph.invoke(
    { goal: "long context models", style: resolveStyle({ length: "short" }), materials: MATERIALS, outline },
    { configurable: { model, emit: (e) => events.push(e) }, recursionLimit: 10 }
  );
  assert.equal(events.find((e) => e.type === "phase" && e.phase === "outline").label, "Using your outline");
  assert.ok(final.article.startsWith("# Hand-made"));
  assert.ok(final.article.includes("Hand-outlined content [2]."));
  assert.ok(events.some((e) => e.type === "log" && /Polish failed/.test(e.message)));
});
