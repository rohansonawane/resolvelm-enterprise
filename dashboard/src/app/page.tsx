"use client";

import { useEffect, useState } from "react";
import {
  apiGet,
  apiPost,
  type AnalyzeResponse,
  type DemoIncident,
} from "@/lib/api";

export default function AnalyzerPage() {
  const [demos, setDemos] = useState<DemoIncident[]>([]);
  const [selected, setSelected] = useState<string>("connection_exhaustion");
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    apiGet<{ demos: DemoIncident[] }>("/api/v1/demos")
      .then((d) => setDemos(d.demos))
      .catch((e: Error) => setError(e.message));
  }, []);

  async function onAnalyze() {
    const demo = demos.find((d) => d.demo_id === selected);
    if (!demo) return;
    setLoading(true);
    setError(null);
    try {
      const body = await apiPost<AnalyzeResponse>("/api/v1/incidents/analyze", demo.request);
      setResult(body);
    } catch (e) {
      setError(e instanceof Error ? e.message : "request failed");
    } finally {
      setLoading(false);
    }
  }

  const demo = demos.find((d) => d.demo_id === selected);

  return (
    <section>
      <h1>Incident Analyzer</h1>
      <p className="muted">
        Demo incidents only. Research-style logs, not a live production ticket stream.
      </p>
      {error && <p className="error">{error}</p>}
      {demos.length === 0 && !error && <p className="muted">Loading demos…</p>}
      <div className="grid">
        <div className="card">
          <label htmlFor="demo">Demo incident</label>
          <select id="demo" value={selected} onChange={(e) => setSelected(e.target.value)}>
            {demos.map((d) => (
              <option key={d.demo_id} value={d.demo_id}>
                {d.title} (demo)
              </option>
            ))}
          </select>
          {demo && (
            <pre>{JSON.stringify(demo.request, null, 2)}</pre>
          )}
          <button type="button" onClick={onAnalyze} disabled={loading || !demo}>
            {loading ? "Analyzing…" : "Analyze"}
          </button>
        </div>
        <div className="card">
          <h2>Diagnosis</h2>
          {!result && <p className="muted">No analysis yet.</p>}
          {result?.error && (
            <div>
              <p className="error">{result.error.error}</p>
              <pre>{result.error.message}</pre>
            </div>
          )}
          {result?.analysis && (
            <div>
              <p>
                <span className="badge">{result.analysis.severity}</span>{" "}
                <strong>{result.analysis.root_cause_code}</strong>
              </p>
              <p>{result.analysis.root_cause}</p>
              <p className="muted">
                confidence {result.analysis.confidence.toFixed(2)} · {result.latency_ms.toFixed(1)} ms ·{" "}
                {result.backend} · {result.model_version}
              </p>
              <h3>Evidence</h3>
              <ul>
                {result.analysis.evidence.map((e) => (
                  <li key={e.source_id}>
                    {e.source_id}: {e.reason}
                  </li>
                ))}
              </ul>
              <h3>Actions</h3>
              <ol>
                {result.analysis.recommended_actions.map((a) => (
                  <li key={a.action_id}>
                    {a.action_id}: {a.description}
                  </li>
                ))}
              </ol>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
