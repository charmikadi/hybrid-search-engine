/**
 * Search UI: posts to FastAPI `/search` via the Vite dev proxy (`/api` → backend).
 */

import { FormEvent, useState } from "react";

/** Mirrors backend `SearchRequest.mode` / `SearchResponse.mode`. */
type SearchMode = "keyword" | "vector" | "hybrid";

/** One row from `SearchResponse.hits`; `rrf_score` is set only in hybrid mode. */
type SearchHit = {
  doc_id: string;
  title: string;
  snippet: string;
  score: number;
  rrf_score: number | null;
};

/** Shape of `POST /search` JSON body (echoed) plus hits and server timing. */
type SearchResponse = {
  query: string;
  mode: SearchMode;
  hits: SearchHit[];
  took_ms: number;
};

/** Prefix `/api` so `vite.config.ts` proxy strips it and forwards to port 8000. */
const api = (path: string, init?: RequestInit) =>
  fetch(`/api${path}`, init);

export default function App() {
  const [query, setQuery] = useState("semantic search FAISS");
  const [mode, setMode] = useState<SearchMode>("hybrid");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SearchResponse | null>(null);

  /** POST JSON to backend; surface HTTP errors as `error`, clear `result` on failure. */
  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const res = await api("/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, mode, top_k: 10 }),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || res.statusText);
      }
      setResult((await res.json()) as SearchResponse);
    } catch (err) {
      setResult(null);
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <header className="header">
        <h1>Hybrid search</h1>
        <p className="lede">
          Keyword TF‑IDF over an inverted index, dense vectors (MiniLM +
          FAISS), and RRF fusion for hybrid mode.
        </p>
      </header>

      <form className="search-bar" onSubmit={onSubmit}>
        <input
          className="input"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Try: GPU inference, inverted index, operating system…"
          aria-label="Search query"
        />
        <div className="modes">
          {(
            [
              ["keyword", "Keyword (TF‑IDF)"],
              ["vector", "Semantic (vectors)"],
              ["hybrid", "Hybrid (RRF)"],
            ] as const
          ).map(([m, label]) => (
            <label key={m} className="mode">
              <input
                type="radio"
                name="mode"
                checked={mode === m}
                onChange={() => setMode(m)}
              />
              {label}
            </label>
          ))}
        </div>
        <button className="btn" type="submit" disabled={loading}>
          {loading ? "Searching…" : "Search"}
        </button>
      </form>

      {error ? <p className="error">{error}</p> : null}

      {result ? (
        <section className="results">
          <p className="meta">
            <strong>{result.hits.length}</strong> hits · {result.took_ms} ms ·
            mode <code>{result.mode}</code>
          </p>
          <ol className="hit-list">
            {result.hits.map((h, i) => (
              <li key={h.doc_id} className="hit">
                <div className="hit-title">
                  <span className="rank">{i + 1}.</span> {h.title}
                </div>
                <div className="hit-id">{h.doc_id}</div>
                <p className="snippet">{h.snippet}</p>
                <div className="scores">
                  <span>score: {h.score.toFixed(4)}</span>
                  {h.rrf_score != null ? (
                    <span>rrf: {h.rrf_score.toFixed(4)}</span>
                  ) : null}
                </div>
              </li>
            ))}
          </ol>
        </section>
      ) : null}
    </div>
  );
}
