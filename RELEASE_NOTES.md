# Release v1.0 — Pretrained adapters

LoRA adapters for `unsloth/qwen2.5-1.5b-instruct-bnb-4bit`, accompanying the
SFT→GRPO Pokémon Showdown pipeline. Attach the zipped adapter directories below
as release assets.

## Assets

| Asset | What it is | Used for |
|---|---|---|
| `sft_qwen_vanilla.zip` | SFT on raw gen9ou replays | The under-switching baseline (switch-EM 14.8%) |
| `sft_qwen_balanced.zip` | SFT with switch decisions upsampled to ~50% | The over-switching variant (switch-EM 88.9%); GRPO start point |
| `grpo_ou1_step5.zip` | GRPO checkpoint, gen9ou matched, step 5 | The generalization test: 98–2 on team1, 41–59 on team2 |
| `grpo_real3_step20.zip` | GRPO checkpoint from the earlier gen9randombattle run | The format-confound discussion (131/300 vs SFT) |

## Usage

```python
from unsloth import FastLanguageModel
model, tok = FastLanguageModel.from_pretrained(
    "sft_qwen_balanced", max_seq_length=512, load_in_4bit=True)
FastLanguageModel.for_inference(model)
```

For live play, pair with a running Showdown server and the constrained-scoring
player in `src/rl_stage_a.py` (see repo README).

## Notes

- These are small QLoRA adapters, not full models; the 4-bit Qwen2.5-1.5B base
  is pulled from Hugging Face automatically.
- `grpo_ou1_step5` is the *peak pre-divergence* checkpoint. Later steps (10, 15)
  are progressively degraded as training diverges and are not recommended.
