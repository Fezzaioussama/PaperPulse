# PaperPulse AI

Vite + React research briefing app for arXiv and Hugging Face daily papers.

## Deploy on Vercel

1. Import this repository in Vercel.
2. Set these Environment Variables in the Vercel project:
   - `OPENROUTER_API_KEY`: your OpenRouter API key
   - `OPENROUTER_MODEL`: optional model id, for example `anthropic/claude-3.5-sonnet`
3. Deploy with the default Vercel settings. The project includes `vercel.json`:
   - build command: `npm run build`
   - output directory: `dist`
   - serverless API route: `/api/chat`

Do not set the OpenRouter key with a `VITE_` prefix. `VITE_` variables are bundled into browser JavaScript.

## Local development

```bash
npm install
npm run dev
```

For local chat/summarization with plain Vite dev, add an OpenRouter key in the app Settings panel. To test the Vercel serverless route locally, use `vercel dev` with `OPENROUTER_API_KEY` configured.
