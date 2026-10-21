import { ChatOpenAI } from "@langchain/openai";

// Parity with api/chat.js: the server-funded path always uses the server's
// model; only a user paying with their own key may choose one.
export const DEFAULT_MODEL = "deepseek/deepseek-v4-pro";

export function resolveModel(viaBrowserKey, requested) {
  return viaBrowserKey && typeof requested === "string" && requested.trim()
    ? requested.trim().slice(0, 100)
    : (process.env.OPENROUTER_MODEL || DEFAULT_MODEL).trim();
}

export function makeModel({ apiKey, model, maxTokens = 4000, temperature = 0.2 }) {
  return new ChatOpenAI({
    apiKey,
    model,
    maxTokens,
    temperature,
    configuration: {
      baseURL: "https://openrouter.ai/api/v1",
      defaultHeaders: {
        "HTTP-Referer": "https://paperpulse.vercel.app",
        "X-Title": "PaperPulse",
      },
    },
  });
}
