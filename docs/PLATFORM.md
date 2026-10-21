# PaperPulse Research Platform — Audit, Architecture, Roadmap

_Last updated: 2026-10-21. This is the engineering document behind the
"autonomous AI research studio" direction: what exists, what was added, what
is deliberately **not** claimed, and what each next phase requires._

## 1. Audit of the existing system

| Layer | Implementation | Assessment |
|---|---|---|
| Orchestration | LangGraph JS (`@langchain/langgraph`), four server-side graphs streamed over SSE | Real agentic orchestration; keep. LangGraph JS in this repo was an explicit product decision — a Python/FastAPI rewrite is out of scope unless re-decided |
| Transport | Vercel Node functions, data-only SSE, heartbeats, abort-on-disconnect | Works; `maxDuration 300` bounds any workflow |
| Literature tools | arXiv Atom API; **added:** Semantic Scholar Graph API, OpenAlex API (`api/_lib/scholar.js`) | Three real sources, merged and deduped by arXiv id → DOI → title |
| Evidence | **Added:** verbatim-quote extraction with *mechanical* verification (`verifyQuote`: normalized containment against fetched source text) | A claim's quote is marked verified only when it literally occurs in the source |
| Hypotheses | **Added:** falsifiable hypotheses + API-based novelty probe (real title search, not generation) | |
| Debate | **Added:** Skeptic → Proponent → Arbiter with per-hypothesis verdicts (supported/contested/rejected), confidence, and **recorded disagreements**; missing verdicts backfill as *contested*, never approved | Anti-sycophancy by construction |
| Methodology | **Added:** experiment protocols (variables, baselines, datasets, metrics, ablations, risks, acceptance criteria) | Plans only — see §3 |
| Memory | Browser: `localStorage` knowledge base + IndexedDB vector RAG (explicit product decision) | Keeps the server stateless; limits: per-browser, ~5MB |
| UI | Digest / Research / Lab / Studio modes; Lab adds a live pipeline view and an agent activity monitor (agent + tool events) | |
| Security | Bearer/own-key auth, SSRF guard on ingestion, no upstream error echo, prompt notes treating fetched text as data | Sound for current scope |

## 2. What the Lab run actually does (`action: "lab"`)

Nine named agents, ≤17 LLM calls, every stage failure-isolated:

```
Director(plan) → Scout(arXiv+S2+OpenAlex, grade) → Analyst(full-text reads,
quote-verified claims) → Ideator(hypotheses + novelty probe) →
Skeptic → Proponent → Arbiter(verdicts + disagreements) →
Methodologist(protocols) → Writer(cited report + mechanical provenance stats)
```

Reliability mechanics (all mechanical, none LLM-asserted):

- `verifyQuote` — evidence quotes string-matched against fetched text.
- `sanitizeReportCitations` — `[n]` outside the paper list stripped.
- `probeNovelty` — closest prior work found by real search APIs.
- `buildVerificationStats` — provenance counts computed from state.
- Verdict backfill — an undebated hypothesis is *contested*, never silently supported.
- All external text (papers, abstracts, web pages) is prompt-labeled as data, not instructions.

## 3. Deliberately NOT claimed (and what each needs)

| Capability | Why not now | Unblocking decision |
|---|---|---|
| Code execution / experiments | Vercel functions are no sandbox; running generated code in-process is the exact anti-pattern the security model forbids | A separate runner (Docker host or Modal/Fly worker) with resource limits, approval gates, artifact storage. Protocols in the Lab are designed to hand off to it |
| Durable server-side projects DB | Browser storage was an explicit decision; revisit only if multi-device/team features are wanted | Postgres + pgvector behind the same API |
| PDF parsing (non-arXiv) | No server-side PDF pipeline; arXiv HTML covers most of the current corpus | GROBID or Docling as a service the ingest action calls |
| Workflow canvas editing | The Lab pipeline view is read-only; drag-and-drop workflow *authoring* is a large product in itself | React Flow + a workflow definition schema |
| Resumable long runs | A run lives within one SSE request (≤300s); partials are salvaged client-side | Durable queue (e.g. Inngest/Temporal) if runs must outlive requests |

## 4. Roadmap

- **Phase A (done):** multi-source literature, verified evidence, hypothesis
  board, debate with recorded disagreement, protocols, report, Lab UI, tests.
- **Phase B:** ingestion upgrades — PDF drag-and-drop via a parsing service,
  DOI/GitHub import, per-paper claim linking in the reader.
- **Phase C:** execution runner — sandboxed Python (Docker), protocol → code
  scaffold (PyTorch project + tests) generated as reviewable artifacts, human
  approval gate, metrics captured from real process output only.
- **Phase D:** experiment dashboard fed by runner artifacts; comparisons vs
  baselines; reproduction workflow (paper → plan → code → run → report).
- **Phase E:** optional durable backend (Postgres/pgvector) for team
  workspaces; keep browser mode as the zero-infra default.

## 5. Testing & verification discipline

`npm test` (node:test, 43 tests) covers: SSE endpoint auth/validation, all
pure normalizers, scholar source parsing + cross-source merge, quote
verification, and full stubbed end-to-end runs of the research, compose, and
lab graphs (prompt-marker-routed stub model + stubbed HTTP). Live smoke tests
run against the dev server with real literature APIs; LLM-dependent stages
degrade gracefully and say so in the stream.
