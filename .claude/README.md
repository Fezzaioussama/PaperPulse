# Claude Code project setup

## Agents (`.claude/agents/`)

| Agent | Role | Typical use |
|---|---|---|
| `implementer` | Builds features/fixes following repo conventions, writes tests, self-verifies (tests + build + smoke test) | "Use the implementer agent to add X" |
| `code-reviewer` | Adversarial, read-only review of a diff; re-runs tests itself; checks the security invariants in `CLAUDE.md` | After any implementation, before committing |
| `ui-verifier` | Drives the running app in a real browser via the Playwright MCP; all modes, both themes, console errors, screenshots | After frontend changes |

The intended loop: **implementer → code-reviewer → (fixes) → ui-verifier**.

## MCP servers (`.mcp.json`, auto-approved via `settings.json`)

- **playwright** (`@playwright/mcp`) — real browser automation for the
  ui-verifier and ad-hoc UI checks. First launch downloads the server
  package; Chromium is usually already cached by the test tooling.
- **context7** (`@upstash/context7-mcp`) — up-to-date library docs
  (LangGraph JS, motion, Vite...) to keep API usage current. Works keyless
  with rate limits; set `CONTEXT7_API_KEY` for higher ones.

## Optional: Graphiti (knowledge-graph memory MCP)

[Graphiti](https://github.com/getzep/graphiti) gives agents a temporal
knowledge-graph memory — a natural fit for connecting papers ↔ concepts ↔
experiments across sessions. It is **not enabled by default** because it
needs real infrastructure: a Neo4j (or FalkorDB) instance plus an LLM API
key for its extraction step.

To enable it, follow the `mcp_server` README in the Graphiti repo (their
Docker Compose is the simplest route — it exposes an SSE endpoint), then add
to `.mcp.json`, e.g.:

```json
"graphiti": {
  "type": "sse",
  "url": "http://localhost:8000/sse"
}
```

Check their README for the current transport/port and required env
(`NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `OPENAI_API_KEY`) — the
project moves quickly.

## Settings

`settings.json` auto-approves the project MCP servers and allowlists the
safe repo commands (tests, build, make targets, read-only git). Personal
overrides belong in `settings.local.json` (gitignored by Claude Code
convention — do not commit it).
