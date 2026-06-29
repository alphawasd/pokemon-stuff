# Training a Small Language Model to Play Competitive Pokémon

An end-to-end pipeline that fine-tunes **Qwen2.5-1.5B-Instruct** to play **Gen 9 OU**
Pokémon on [Pokémon Showdown](https://pokemonshowdown.com/), via supervised
fine-tuning (SFT) on human replays followed by online **GRPO**, all on a single
free-tier **T4** GPU.

This is a **negative-results + systems** project. The headline finding is not
"RL makes a great Pokémon player" — it is that small-scale on-policy GRPO,
given a narrow training distribution, **overfits to its training matchup
spectacularly rather than acquiring transferable skill**, and we characterise
exactly how and why. The pipeline, the constrained-decoding action mechanism,
and a documented gradient-throttling pitfall are the reusable contributions.

## TL;DR results

| Finding | Result |
|---|---|
| SFT under/over-switch bias | Vanilla SFT under-switches (switch-EM **14.8%**, move-EM 23.4%); balancing **inverts** it (switch-EM **88.9%**, move-EM 11.0%) — it does not fix the bias |
| Constrained decoding | Illegal-action rate **97% → 0%** by construction; yields exact action log-probs for the policy gradient |
| GRPO (matched OU) overfits | Best checkpoint beats SFT **98–2** on its **trained** team, but only **41–59** on an **unseen** team — the gain does not transfer |
| GRPO training is unstable | Win rate vs scripted bot rises to a peak (~step 8) then collapses to 0; gradient norm diverges past ~10³ |
| Gradient-throttle pitfall | A clip+averaging combination silently held the effective step ≈ lr regardless of lr, flattening five runs; caught only by measuring weight deltas directly |
| Format confound | poke-env defaults to `gen9randombattle`; the RL phase accidentally trained off-distribution from the `gen9ou` SFT until this was caught and corrected |

See [`results/`](results/) for the raw numbers and [`figures/`](figures/) for the plots.

## The generalization test (the key experiment)

The same GRPO checkpoint (`grpo_ou1_step5`), same opponent (its SFT start),
same greedy decoding — only the team changes:

| Matchup | GRPO win rate vs SFT | Median turns |
|---|---|---|
| **team1** (trained on) | **98 / 100** | 41 |
| **team2** (never seen)  | **41 / 100** | 25 |

Both conditions are full-length battles (0 forfeits/timeouts), so the 98% is
real in-battle play, not a degenerate artifact. The collapse from 98% to 41%
on an unseen team is the clean, controlled demonstration of matchup-specific
overfitting.

## Repository layout

```
src/
  sft_data_prep.py   # download high-Elo gen9ou replays -> (state, action) JSONL
  split_eval.py      # leak-free split BY REPLAY ID; AA/EM/per-class eval
  balance_data.py    # upsample switch decisions to ~50% (train only)
  sft_train.py       # Unsloth QLoRA SFT (r=16, completion-only loss)
  ou_team.py         # gen9ou format + fixed OU teams (team1 trained, team2 unseen)
  rl_stage_a.py      # constrained-scoring poke-env Player + trajectory logging
  rl_stage_b.py      # GRPO group advantages (terminal + dense shaping)
  rl_stage_c.py      # GRPO update loop (KL to frozen SFT ref); --dense option
  rl_eval.py         # static / head-to-head / drift eval
  make_figures.py    # paper figures
results/             # FINAL_results.json, ou_final_result.json, grpo_ou1_result.json
figures/             # fig1 (training dynamics), fig2 (SFT bias)
data/sample_data.jsonl   # 20 example (state, action) rows
run.sh               # end-to-end invocation reference
requirements.txt     # pinned environment
```

## Setup

```bash
pip install -r requirements.txt
```

Pinned to the exact stack used for the results: Python 3.12, torch 2.10,
transformers 5.5.0, unsloth 2026.6.9, trl 0.24.0, peft 0.18.1, poke-env 0.15.0.

### Showdown server (required for any live play)

RL training and head-to-head eval need a local Showdown server (Node.js):

```bash
git clone --depth 1 https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown && npm install && cd ..
# detached so it survives notebook cell reruns:
setsid nohup node pokemon-showdown/pokemon-showdown start --no-security > ps.log 2>&1 &
ss -ltn | grep 8000   # confirm it's listening
```

## Reproducing

`run.sh` documents every stage in order. Stages are normally run individually
while watching logs. Key gotchas baked into the code and worth knowing:

- **Format must be pinned.** poke-env defaults to `gen9randombattle`.
  `ou_team.py` forces `gen9ou` + a team for every player; without this the RL
  phase silently trains off-distribution from the OU SFT data.
- **Watch grad norm, not loss.** GRPO here rises then diverges (norm → 10³+
  past ~step 8–12). The useful checkpoints are early (step 5).
- **Measure weight deltas.** `rl_stage_c.py` prints the max weight change after
  step 1. If it is ~1e-5 the policy is not actually moving (see the
  gradient-throttle pitfall in the paper); a healthy step is ~1e-3.

## Pretrained adapters

LoRA adapters are released as assets on the
[GitHub Releases](../../releases) page (not committed in-tree):

- `sft_qwen_vanilla` — SFT on raw gen9ou replays
- `sft_qwen_balanced` — SFT with switch decisions upsampled to ~50%
- `grpo_ou1_step5` — the GRPO checkpoint used in the generalization test
- `grpo_real3_step20` — the earlier (gen9randombattle) GRPO checkpoint referenced
  in the format-confound discussion

Each is a small QLoRA adapter for `unsloth/qwen2.5-1.5b-instruct-bnb-4bit`.

## Citing / related work

This work sits alongside, and is distinct from:
- **PokéChamp** (Karten et al., 2025, arXiv:2503.04094) — LLM-in-minimax, *no training*.
- **PokeAgent Challenge** (Karten et al., 2026, arXiv:2603.15563).
- **Meta Discovery** (Saravanan & Guzdial, 2024, arXiv:2409.07340) — small PPO net, not an LLM.
- **PokerBench** (arXiv:2501.08328) — trains LLMs for poker; SFT brittleness motivates RL.
- **poke-env** (Sahovic, 2019) — the Showdown interface used here.

## License

MIT — see [LICENSE](LICENSE).
