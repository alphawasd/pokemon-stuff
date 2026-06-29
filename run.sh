#!/usr/bin/env bash
# =============================================================================
# End-to-end pipeline: SFT -> balance -> GRPO -> eval, for a small LLM on
# Gen 9 OU Pokemon Showdown. Built and run on a single free-tier T4.
#
# This is a REFERENCE of the exact invocations used to produce the paper's
# results, not a one-shot script — the live-play stages need a running
# Showdown server and a GPU, and each stage is normally run interactively
# while watching the logs. Run stages individually.
# =============================================================================
set -e

ADAPTERS_DIR=${ADAPTERS_DIR:-./adapters}      # where SFT/GRPO adapters live
SFT_DATA=sft_train.jsonl

# -----------------------------------------------------------------------------
# 0. Showdown server (required for ALL live-play: RL training + h2h eval)
#    Detached with setsid+nohup so it survives notebook cell reruns — this
#    detachment was necessary; a plain background process kept dying mid-run.
# -----------------------------------------------------------------------------
start_server() {
  [ -d pokemon-showdown ] || git clone --depth 1 https://github.com/smogon/pokemon-showdown.git
  [ -d pokemon-showdown/node_modules ] || (cd pokemon-showdown && npm install)
  chmod -R +x pokemon-showdown/node_modules/@esbuild pokemon-showdown/node_modules/.bin 2>/dev/null || true
  setsid nohup node pokemon-showdown/pokemon-showdown start --no-security > ps.log 2>&1 &
  sleep 25
  ss -ltn | grep -q 8000 && echo "server UP" || echo "server DOWN — check ps.log"
}

# -----------------------------------------------------------------------------
# 1. SFT data prep: download high-Elo gen9ou replays -> (state, action) JSONL
# -----------------------------------------------------------------------------
python src/sft_data_prep.py            # -> sft_train.jsonl

# -----------------------------------------------------------------------------
# 2. Leak-free split BY REPLAY ID (no battle's turns straddle train/test)
# -----------------------------------------------------------------------------
python src/split_eval.py split --data $SFT_DATA --test-frac 0.1

# -----------------------------------------------------------------------------
# 3. Train SFT (vanilla). QLoRA r=16, completion-only loss, 2 epochs.
# -----------------------------------------------------------------------------
python src/sft_train.py --data sft_train_split_train.jsonl --out $ADAPTERS_DIR/sft_qwen_vanilla

# 3b. Balanced variant: upsample switch decisions to ~50% in TRAIN only
python src/balance_data.py --data sft_train_split_train.jsonl --out sft_train_balanced.jsonl
python src/sft_train.py --data sft_train_balanced.jsonl --out $ADAPTERS_DIR/sft_qwen_balanced

# -----------------------------------------------------------------------------
# 4. Static eval (no battles): AA / EM / per-class switch & move EM
# -----------------------------------------------------------------------------
python src/split_eval.py eval --adapter $ADAPTERS_DIR/sft_qwen_vanilla  --test sft_train_split_test.jsonl
python src/split_eval.py eval --adapter $ADAPTERS_DIR/sft_qwen_balanced --test sft_train_split_test.jsonl

# -----------------------------------------------------------------------------
# 5. GRPO (online) — gen9ou, fixed mirror team (src/ou_team.py).
#    NOTE: poke-env defaults to gen9randombattle if battle_format is unset;
#    ou_team.py pins gen9ou + a team so RL matches the SFT distribution.
#    group=8, lr=5e-5, kl=0.1. Watch grad_norm; it diverges past ~step 8-12.
# -----------------------------------------------------------------------------
start_server
python src/rl_stage_c.py --adapter $ADAPTERS_DIR/sft_qwen_balanced \
  --opponent maxpower --group-size 8 --temperature 0.6 \
  --steps 25 --lr 5e-5 --kl-beta 0.1 --save-every 5 --out $ADAPTERS_DIR/grpo_ou1

# -----------------------------------------------------------------------------
# 6. Head-to-head: GRPO checkpoint vs its SFT start, on the SAME format.
#    team1 = trained matchup; team2 = unseen (generalization test).
# -----------------------------------------------------------------------------
start_server
python src/rl_eval.py h2h --adapter-a $ADAPTERS_DIR/grpo_ou1_step5 \
  --adapter-b $ADAPTERS_DIR/sft_qwen_balanced --n-battles 100 --team team1
python src/rl_eval.py h2h --adapter-a $ADAPTERS_DIR/grpo_ou1_step5 \
  --adapter-b $ADAPTERS_DIR/sft_qwen_balanced --n-battles 100 --team team2

# -----------------------------------------------------------------------------
# 7. Figures
# -----------------------------------------------------------------------------
python src/make_figures.py
