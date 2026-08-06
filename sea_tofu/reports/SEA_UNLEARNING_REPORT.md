# SEA-on-TOFU — standard unlearning report (forget10, rank 16)

## Methodology (summary; full writeup in ../REPORT.md)

SEA reframes TOFU so each author is one SEA *user* with a deletable per-author personal-LoRA
proxy over a frozen 4-bit meta-llama/Llama-2-7b-chat-hf base; unlearning = `rm` of the proxy dir.
Each proxy = one LoRA on q/k/v/o across all 32 layers (256 tensors, params = 1,048,576·r; r16 =
16,777,216 params, 65 MB), trained by SFT (12 epochs, lr 2e-4) on only that author's 20 QA pairs.
We evaluate the SAME base model in three states and score each with the canonical
TOFU metrics (reused verbatim from tofu_sisa_lora/eval_tofu.py — ROUGE-L recall, length-normalized
probability, perturbed-answer truth ratio, KS Forget Quality vs the retrain gold, harmonic-mean
Model Utility over Retain×{prob,rouge,truth}+Real+World):

- **Original** — forget authors with their proxies loaded (the model that *knows* the forget data).
- **Unlearned** — forget proxies deleted → forget set = frozen base (omission mode); retain authors
  keep their proxies; Real/World = base. This *is* the retrain gold by construction.
- **Retrain gold** — base on the forget set (identical to Unlearned's forget side by construction).

Model Utility is identical across states because deletion never touches retain/real/world.

## Results

| State | Forget ROUGE-L | Forget Prob | Forget TR | Retain ROUGE | Retain Prob | Real | World | Forget Quality | Model Utility |
|---|---|---|---|---|---|---|---|---|---|
| Original (proxies loaded) | 1.0 | 0.9987 | 0.4758 | 1.0 | 0.9986 | 0.689 | 0.8561 | 0.0 | 0.7106 |
| Unlearned (proxies deleted) | 0.403 | 0.1607 | 0.7012 | 1.0 | 0.9986 | 0.689 | 0.8561 | 1.0 | 0.7106 |
| Retrain gold (= base) | 0.403 | 0.1607 | 0.7012 | 1.0 | 0.9986 | 0.689 | 0.8561 | 1.0 | 0.7106 |

Forget Quality = KS p-value of forget truth-ratios vs the retrain gold (base on forget).
Model Utility = harmonic mean of retain/real/world × prob/rouge/truth (unchanged by deletion).
Deletion cost = `rm` of the proxy dir (ms); retrain gold reached by construction.

_Source: job 436005, `eval_unlearning_report.py` (max_new=100, retain sample 40, seed 42)._
