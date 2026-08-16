from pathlib import Path

import pytest

from resolvelm.training.hf_load import load_causal_lm


def test_check_gpu_refuses_apple_silicon() -> None:
    import platform
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[2] / "scripts" / "check_gpu.py"
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        assert result.returncode == 1
        assert "Apple Silicon" in result.stderr or "smoke machine" in result.stderr


def test_cuda_requested_raises_without_gpu() -> None:
    cfg = {
        "model": {"id": "Qwen/does-not-download"},
        "train": {"device": "cuda", "precision": "bf16"},
        "lora": {"qlora": True},
    }
    with pytest.raises(RuntimeError, match="CUDA"):
        load_causal_lm(cfg)


def test_qlora_requires_cuda_even_on_auto() -> None:
    cfg = {
        "model": {"id": "Qwen/does-not-download"},
        "train": {"device": "auto"},
        "lora": {"qlora": True},
    }
    with pytest.raises(RuntimeError, match="QLoRA"):
        load_causal_lm(cfg)


def test_quantization_kwargs_skips_bnb_off_cuda() -> None:
    pytest.importorskip("torch")
    from resolvelm.training.hf_load import quantization_kwargs

    kwargs = quantization_kwargs(
        {
            "model": {},
            "lora": {"qlora": True, "quant_type": "nf4"},
            "train": {"device": "cpu", "precision": "bf16"},
        }
    )
    assert "quantization_config" not in kwargs
    assert kwargs.get("trust_remote_code") is False
