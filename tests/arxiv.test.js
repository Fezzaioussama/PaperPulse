import test from "node:test";
import assert from "node:assert/strict";
import { parseArxivXml, htmlToTextNode, dedupe } from "../api/_lib/arxiv.js";

const TWO_ENTRY_XML = `<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <title>State Space  Models
      for Long Context</title>
    <summary>We study state space models   for long context modeling.</summary>
    <published>2024-01-05T00:00:00Z</published>
    <author><name>Alice Smith</name></author>
    <author><name>Bob Jones</name></author>
    <author><name>Carol White</name></author>
    <link rel="alternate" href="http://arxiv.org/abs/2401.00001v1" type="text/html"/>
    <link rel="related" href="http://arxiv.org/pdf/2401.00001v1"/>
    <category term="cs.LG"/>
    <category term="cs.CL"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2401.00002v1</id>
    <title>Attention Alternatives</title>
    <summary>A survey of attention alternatives.</summary>
    <published>2024-01-06T00:00:00Z</published>
    <author><name>Dan Brown</name></author>
    <category term="cs.AI"/>
  </entry>
</feed>`;

const ONE_ENTRY_XML = `<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2402.09999v2</id>
    <title>Single Entry</title>
    <summary>Only one entry in the feed.</summary>
    <published>2024-02-01T12:30:00Z</published>
    <author><name>Eve Adams</name></author>
    <link rel="alternate" href="http://arxiv.org/abs/2402.09999v2"/>
    <category term="stat.ML"/>
  </entry>
</feed>`;

test("parseArxivXml maps entries to the client paper shape", () => {
  const papers = parseArxivXml(TWO_ENTRY_XML);
  assert.equal(papers.length, 2);
  assert.deepEqual(papers[0], {
    title: "State Space Models for Long Context",
    authors: "Alice Smith et al.",
    date: "2024-01-05",
    url: "http://arxiv.org/abs/2401.00001v1",
    abstract: "We study state space models for long context modeling.",
    categories: ["cs.LG", "cs.CL"],
    source: "arxiv",
  });
  // No alternate link → falls back to <id>; ≤2 authors are joined, not "et al.".
  assert.equal(papers[1].url, "http://arxiv.org/abs/2401.00002v1");
  assert.equal(papers[1].authors, "Dan Brown");
});

test("parseArxivXml handles a single-entry feed and junk input", () => {
  const papers = parseArxivXml(ONE_ENTRY_XML);
  assert.equal(papers.length, 1);
  assert.equal(papers[0].title, "Single Entry");
  assert.equal(papers[0].date, "2024-02-01");
  assert.deepEqual(papers[0].categories, ["stat.ML"]);
  assert.deepEqual(parseArxivXml("not xml at all"), []);
  assert.deepEqual(parseArxivXml("<feed></feed>"), []);
});

test("htmlToTextNode strips chrome, keeps structure, decodes entities", () => {
  const html = `<html><head><style>body{color:red}</style>
    <script>alert("x")</script></head>
    <body><nav>Menu</nav>
    <h2>Introduction</h2>
    <p>Models &amp; methods are &quot;compared&quot; &#39;here&#39;.</p>
    <ul><li>First point</li><li>Second&nbsp;point</li></ul>
    <footer>ignore me</footer></body></html>`;
  const text = htmlToTextNode(html);
  assert.ok(!text.includes("alert"));
  assert.ok(!text.includes("color:red"));
  assert.ok(!text.includes("Menu"));
  assert.ok(!text.includes("ignore me"));
  assert.ok(text.indexOf("## Introduction") < text.indexOf("Models"));
  assert.ok(text.includes('Models & methods are "compared" \'here\'.'));
  assert.ok(text.includes("- First point"));
  assert.ok(text.includes("- Second point"));
});

test("dedupe keys on arXiv id, else lowercased title", () => {
  const papers = dedupe([
    { url: "http://arxiv.org/abs/2401.00001v1", title: "A" },
    { url: "http://arxiv.org/abs/2401.00001v2", title: "A (v2)" },
    { url: "", title: "Same Title" },
    { url: "", title: "same title" },
  ]);
  assert.equal(papers.length, 2);
});
