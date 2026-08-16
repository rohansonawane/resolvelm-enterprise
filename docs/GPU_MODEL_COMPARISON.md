# GPU model comparison

**Not benchmarked on this hardware.**

Clone this repo on a Linux NVIDIA machine, then:

```bash
make gpu-setup && make gpu-check
make gpu-data
make gpu-train
make gpu-eval
```

`make gpu-eval` overwrites this file from `artifacts/gpu/model_comparison.csv`. Until that command has been run, there are no Qwen3-4B numbers to report.

Do not copy Mac 0.5B / rule-baseline scores into this page.
