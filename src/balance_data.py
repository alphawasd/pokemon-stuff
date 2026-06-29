"""
Balanced-SFT experiment: does upsampling switches fix the 0% switch collapse?
=============================================================================

DIAGNOSIS (from the vanilla SFT eval)
-------------------------------------
The vanilla model NEVER switches: switch-EM = 0/108 = 0.0%. The training split
is ~77% move / 23% switch, and SFT collapsed to the majority action. This is
the PokerBench imbalance-collapse failure, reproduced in Pokemon.

THE FIX (PokerBench's approach)
-------------------------------
Upsample switch decisions in the TRAINING data until move:switch is ~50:50,
then retrain. If switch-EM rises off zero, the collapse was caused by class
imbalance — a clean, directly-demonstrated result.

CRITICAL: only the TRAIN split is rebalanced. The TEST split stays at its
natural 23% switch rate, or the SFT-vs-balanced comparison is meaningless
(you'd be evaluating on a distribution you engineered).

USAGE
-----
  python balance_data.py --train sft_train_clean_split_train_v2.jsonl --ratio 0.5
  # writes sft_train_clean_split_train_v2_balanced.jsonl

  python sft_train.py --data sft_train_clean_split_train_v2_balanced.jsonl --out sft_qwen_balanced
  python split_eval.py eval --adapter sft_qwen_balanced --test sft_train_clean_split_test_v2.jsonl

Then compare switch-EM: vanilla (0%) vs balanced (?). That delta is a result.
"""

import argparse
import json
import random
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="sft_train_clean_split_train_v2.jsonl")
    ap.add_argument("--ratio", type=float, default=0.5,
                    help="target fraction of switch decisions (0.5 = balanced)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.train)]
    moves = [r for r in rows if json.loads(r["completion"])["action"] == "move"]
    switches = [r for r in rows if json.loads(r["completion"])["action"] == "switch"]

    print(f"[balance] original: {len(moves)} move, {len(switches)} switch "
          f"({len(switches)/len(rows):.1%} switch)")

    # We keep all moves and upsample switches to hit the target ratio.
    # If target switch fraction = r, and we keep all M moves, we need
    # S_new switches such that S_new / (M + S_new) = r  ->  S_new = r*M/(1-r).
    rng = random.Random(args.seed)
    target_switches = int(args.ratio * len(moves) / (1 - args.ratio))

    if target_switches <= len(switches):
        # downsample (unlikely here) — just sample without replacement
        balanced_switches = rng.sample(switches, target_switches)
    else:
        # upsample with replacement (duplicates allowed)
        balanced_switches = [rng.choice(switches) for _ in range(target_switches)]

    balanced = moves + balanced_switches
    rng.shuffle(balanced)

    out = args.train.replace(".jsonl", "_balanced.jsonl")
    with open(out, "w") as f:
        for r in balanced:
            f.write(json.dumps(r) + "\n")

    final = Counter(json.loads(r["completion"])["action"] for r in balanced)
    print(f"[balance] upsampled switches {len(switches)} -> {target_switches} "
          f"(with replacement)")
    print(f"[balance] balanced set: {dict(final)} "
          f"({final['switch']/len(balanced):.1%} switch), {len(balanced)} total")
    print(f"[balance] wrote -> {out}")
    print("\nNOTE: switches are duplicated, so the model sees the same switch")
    print("decisions multiple times. This tests the imbalance hypothesis but")
    print("risks overfitting to those specific spots — watch switch-EM on the")
    print("UNCHANGED test set, not train loss, to judge real improvement.")


if __name__ == "__main__":
    main()
