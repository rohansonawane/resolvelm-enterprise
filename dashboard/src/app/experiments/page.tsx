"use client";

import { useEffect, useState } from "react";
import { apiGet } from "@/lib/api";

export default function ExperimentsPage() {
  const [data, setData] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    apiGet<{ artifacts: Record<string, unknown>; mlflow_note: string }>("/api/v1/experiments")
      .then(setData)
      .catch((e: Error) => setError(e.message));
  }, []);

  return (
    <section>
      <h1>Experiment Explorer</h1>
      <p className="muted">
        MLflow export / local training artifacts. CUDA 4B runs are not claimed here.
      </p>
      {error && <p className="error">{error}</p>}
      {!data && !error && <p className="muted">Loading artifacts…</p>}
      {data && (
        <div className="card">
          <pre>{JSON.stringify(data, null, 2)}</pre>
        </div>
      )}
    </section>
  );
}
