"use client";

import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { apiGet, type ComparisonRow } from "@/lib/api";

export default function ComparisonPage() {
  const [rows, setRows] = useState<ComparisonRow[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    apiGet<{ rows: ComparisonRow[]; available: boolean; note?: string }>("/api/v1/benchmarks")
      .then((d) => {
        setRows(d.rows ?? []);
        if (!d.available) setError(d.note ?? "No measured artifacts yet.");
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  const chart = rows.map((r) => ({
    model: r.model,
    accuracy: Number(r.root_cause_accuracy ?? 0),
    schema: Number(r.pydantic_schema_success ?? 0),
  }));

  return (
    <section>
      <h1>Model Comparison</h1>
      <p className="muted">
        Charts read live API data from `artifacts/`. No placeholder metrics.
      </p>
      {error && <p className="muted">{error}</p>}
      {rows.length === 0 && <p className="muted">Empty — run `make smoke` first.</p>}
      {chart.length > 0 && (
        <div className="card" style={{ height: 320 }}>
          <ResponsiveContainer>
            <BarChart data={chart}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="model" />
              <YAxis domain={[0, 1]} />
              <Tooltip />
              <Legend />
              <Bar dataKey="accuracy" fill="#6ea8fe" name="root cause accuracy" />
              <Bar dataKey="schema" fill="#63e6be" name="schema success" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
      {rows.length > 0 && (
        <div className="card">
          <table>
            <thead>
              <tr>
                {Object.keys(rows[0]).map((k) => (
                  <th key={k}>{k}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.model}>
                  {Object.keys(rows[0]).map((k) => (
                    <td key={k}>{r[k]}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
