export default function ArchitecturePage() {
  return (
    <section>
      <h1>Architecture</h1>
      <div className="card">
        <pre>{`raw incidents → validate → grouped split
    → base eval → LoRA/QLoRA SFT → DPO
    → eval → MLflow registry
    → FastAPI (Transformers | vLLM | rule/classifier)
    → Next.js dashboard`}</pre>
      </div>
      <p className="muted">
        Local default is the rule/classifier path on Apple Silicon. Production target is
        Qwen3-4B-Instruct-2507 with QLoRA + vLLM on CUDA — not executed on this laptop.
      </p>
    </section>
  );
}
