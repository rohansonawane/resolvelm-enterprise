export type DemoIncident = {
  demo_id: string;
  title: string;
  is_demo: boolean;
  request: {
    service: string;
    environment: string;
    symptoms: string[];
    log_events: Array<{
      event_id: string;
      timestamp: string;
      level: string;
      component: string;
      message: string;
    }>;
    demo_id?: string;
  };
};

export type AnalyzeResponse = {
  request_id: string;
  model_version: string;
  backend: string;
  latency_ms: number;
  needs_human_review: boolean;
  analysis: {
    severity: string;
    root_cause_code: string;
    root_cause: string;
    confidence: number;
    evidence: Array<{ source_id: string; reason: string }>;
    recommended_actions: Array<{
      action_id: string;
      priority: number;
      description: string;
    }>;
    needs_human_review: boolean;
  } | null;
  error: { error: string; message: string; raw_output_preview?: string } | null;
};

export type ComparisonRow = Record<string, string>;

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8101";

export async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`API ${path} failed: ${res.status}`);
  }
  return (await res.json()) as T;
}

export async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(`API ${path} failed: ${res.status}`);
  }
  return (await res.json()) as T;
}
