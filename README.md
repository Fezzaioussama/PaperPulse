<div align="center">
  <img src="public/favicon.svg" width="96" height="96" alt="PaperPulse icon" />
  <h1>PaperPulse</h1>
  <p><strong>AI research briefings for fast paper triage.</strong></p>
  <p>
    Fetch, rank, and summarize the latest arXiv and Hugging Face Daily papers by topic — with per-paper AI briefings and an interactive chat interface.
  </p>
</div>

---

## What it does

PaperPulse is a single-page React app that helps researchers and engineers quickly cut through the daily flood of AI/ML papers. You choose one or more topic presets (LLMs, AI Agents, Diffusion, etc.) or type any custom subject, then hit **Analyze**. The app:

1. Fetches fresh papers from the [arXiv API](https://export.arxiv.org/api/) and the [Hugging Face Daily Papers feed](https://huggingface.co/api/daily_papers).
2. Deduplicates, topic-ranks, and trims the pool to your requested count.
3. Calls an LLM (via [OpenRouter](https://openrouter.ai)) to produce a structured briefing for every paper: TL;DR, key points, method summary, results, impact score, and tags.
4. Lets you filter by tag, impact level, or a free-text keyword/subject query — and export everything as Markdown.
5. Opens a **per-paper chat** where the LLM answers questions grounded in the full paper text fetched from `arxiv.org/html/` (with an `ar5iv` fallback).

---

## Features

| Feature | Details |
|---|---|
| **Topic presets** | 14 built-in presets (LLMs, AI Agents, Reasoning, RAG, RLHF/Alignment, Multimodal, Computer Vision, Diffusion, Mech Interp, Efficiency/MoE, Robotics/VLA, AI Safety, Benchmarks, Machine Learning) backed by arXiv category codes and keyword lists |
| **Custom topics** | Free-text topic input; related preset terms are automatically expanded before the arXiv query is built |
| **HF Daily source** | Toggle Hugging Face Daily Papers (community-upvoted, great for trending work) alongside arXiv |
| **Paper count** | Choose 10, 20, or 30 papers per run |
| **Structured briefings** | Per-paper TL;DR, 4 key points, method, results, impact (1–5 stars), relevance tags, and a "why it matters" blurb |
| **Impact filter** | Show only papers with impact ≥ 3, ≥ 4, or exactly 5 |
| **Tag filter** | Click any AI-generated tag to filter the digest to papers that share that tag |
| **Subject finder** | Real-time subject search that scores and re-ranks the loaded papers without a new API call |
| **Free-text filter** | Keyword filter across title, authors, abstract, TL;DR, method, results, and tags |
| **Paper chat** | RAG-style chat grounded in the full paper text; evidence chunks are ranked by query relevance and cited with `[E1]` IDs |
| **Bookmarks** | Star any paper to save it to the local Saved tab, persisted in `localStorage` |
| **Markdown export** | Copy or download the current digest as a tidy Markdown file |
| **Dark mode** | Toggle between light and dark themes, persisted across sessions |
| **Result caching** | The last digest is cached in `localStorage` and restored on reload so you never start with a blank screen |
| **Vercel serverless** | `/api/chat` proxies OpenRouter calls on the server so the API key is never exposed to the browser |

---

## Tech stack

- **Frontend**: React 18, Vite 5 — no CSS framework, styles are inlined as a JS template literal
- **AI**: [OpenRouter](https://openrouter.ai) — any model supported (Claude, Gemini, DeepSeek, etc.)
- **Paper sources**: arXiv Atom API, Hugging Face Daily Papers JSON API
- **Full-text**: `arxiv.org/html/{id}` → `ar5iv.labs.arxiv.org/html/{id}` fallback, fetched via CORS proxies
- **Deployment**: Vercel (static site + Node.js serverless function)

---

## Project layout

```
paperpulse/
├── api/
│   └── chat.js          # Vercel serverless function — proxies OpenRouter requests
├── public/
│   ├── favicon.svg      # App icon (paper + pulse line)
│   ├── favicon-32x32.png
│   ├── icon-192.png
│   ├── icon-512.png
│   ├── apple-touch-icon.png
│   └── site.webmanifest
├── src/
│   ├── App.jsx          # Entire app: data fetching, ranking, UI, chat
│   └── main.jsx         # React entry point
├── index.html
├── vite.config.js
├── vercel.json          # Vercel build + serverless function config
├── .env.example         # Environment variable template
└── package.json
```

---

## Quick start (local)

### Prerequisites

- Node.js ≥ 18
- An [OpenRouter API key](https://openrouter.ai/keys)

### 1. Install dependencies

```bash
npm install
```

### 2. Configure environment variables

Copy the example file and fill in your key:

```bash
cp .env.example .env.local
```

`.env.local`:
```env
# Used by the Vercel serverless function (api/chat.js) — never exposed to the browser
OPENROUTER_API_KEY=sk-or-v1-your-key-here

# Optional: default model for the server route
OPENROUTER_MODEL=deepseek/deepseek-v4-flash

# Optional: default model shown in the browser Settings panel
VITE_OPENROUTER_MODEL=deepseek/deepseek-v4-flash
```

> **Security note**: Never prefix your secret key with `VITE_`. Vite bundles any `VITE_*` variable into the browser bundle. Only `VITE_OPENROUTER_MODEL` (a non-secret model ID) is safe to expose this way.

### 3. Run the dev server

```bash
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

> **For the serverless route locally**: use `vercel dev` instead of `npm run dev`. The `/api/chat` endpoint is a Vercel function and is not started by the plain Vite dev server. As an alternative, add your OpenRouter key directly in the app's **Settings** panel — the browser will then call OpenRouter directly (bypassing the server route).

---

## Deploy on Vercel

1. Push this repository to GitHub (or import it directly).
2. [Import the project in Vercel](https://vercel.com/new).
3. In **Project Settings → Environment Variables**, add:

| Variable | Value |
|---|---|
| `OPENROUTER_API_KEY` | Your OpenRouter secret key |
| `OPENROUTER_MODEL` | *(optional)* e.g. `anthropic/claude-3.5-sonnet` |
| `VITE_OPENROUTER_MODEL` | *(optional)* model ID shown in the browser Settings panel |

4. Deploy. The included `vercel.json` handles everything:
   - `buildCommand`: `npm run build`
   - `outputDirectory`: `dist`
   - Serverless function: `api/chat.js` (30 s max duration)
   - SPA rewrite: all non-API routes → `index.html`

---

## Environment variables reference

| Variable | Required | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | Yes (server) | OpenRouter secret key. Used only in `api/chat.js`. |
| `OPENROUTER_MODEL` | No | Server-side default model. Overridden per-request by the client. |
| `VITE_OPENROUTER_MODEL` | No | Browser default model shown in the Settings panel. Falls back to `anthropic/claude-3.5-sonnet`. |

---

## How the chat works

When you open a paper's chat:

1. The app fetches the full HTML render of the paper from `arxiv.org/html/{id}` (or `ar5iv` as fallback) via CORS proxies.
2. The text is split into structured sections (Abstract, Introduction, Method, Results, etc.) and then into overlapping chunks of ~1 700 characters.
3. For each user message, the top-7 chunks most relevant to the query are selected using a BM25-style token scoring function with intent boosting (e.g. questions about "limitations" boost limitation/discussion sections).
4. The selected evidence chunks and up to 12 recent turns of chat history are sent to the LLM with a system prompt that instructs it to cite evidence with `[E1]`, `[E2]` IDs and to say so when the evidence doesn't support an answer.
5. Chat history is persisted per paper in `localStorage`.

---

## Keyboard shortcuts

| Key | Action |
|---|---|
| `Escape` | Close the paper chat modal |
| `Enter` (in topic input) | Start a new Analyze run |

---

## Local development tips

- The last digest is cached in `localStorage` under the key `pb_cache`. Clear it in DevTools → Application → Local Storage if you want a clean state.
- Bookmarks are stored under `pb_bookmarks`; chat histories under `pb_paper_chats`.
- The topic ranking algorithm (`scorePaperForTopic`) weights: arXiv category match (+8 per category), phrase matches in title (+18) and abstract (+8), token matches in title (+5) and abstract (+2), recency, HF upvotes, and quality signals ("state of the art", "benchmark", etc.).
- CORS proxies used for arXiv full-text: direct → `corsproxy.io` → `api.allorigins.win`. If all fail, chat falls back to the abstract + generated briefing.

---

## License

Private project — see `package.json`.
