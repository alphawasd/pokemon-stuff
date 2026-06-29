"""
Split + Eval harness for the Showdown SFT agent
===============================================

Two tools the rest of the project reuses:

  split : leak-free train/test split of the SFT pairs, grouped by REPLAY ID.
  eval  : move-accuracy of a trained adapter on the held-out test set,
          reporting PokerBench-style AA (action type) and EM (exact action).

WHY SPLIT BY REPLAY ID (not by individual pair)
-----------------------------------------------
Pairs from the same battle are highly correlated (same teams, same players,
adjacent turns). If turns from one battle land in both train and test, the
model has effectively seen the test distribution -> inflated, dishonest
accuracy. A reviewer will catch this instantly. So we split on replay_id:
every decision from a given battle goes entirely to train OR entirely to test.

TWO METRICS (mirrors PokerBench's AA / EM)
------------------------------------------
  Action Accuracy (AA): did the model pick the right ACTION TYPE
      (move vs switch)? Coarse but meaningful.
  Exact Match (EM): did it pick the exact same move/switch target as the
      strong human? Strict.
  We also report move-only and switch-only breakdowns, because a model can
  game AA by always picking the majority class (move). EM and per-class
  numbers expose that.

USAGE
-----
  # 1. make the split (writes train/test JSONL)
  python split_eval.py split --data sft_train_clean.jsonl --test-frac 0.1

  # 2. retrain on the TRAIN split only (use your sft_train.py)
  python sft_train.py --data sft_train_split_train.jsonl --out sft_qwen_train

  # 3. evaluate the trained adapter on the held-out TEST split
  python split_eval.py eval --adapter sft_qwen_train --test sft_train_split_test.jsonl

The eval prints AA / EM / per-class accuracy + a few example predictions.
Reuse `eval` later on each RL checkpoint to build the SFT-vs-RL comparison.
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict


# --------------------------------------------------------------------------- #
# split
# --------------------------------------------------------------------------- #
def split(args):
    rows = [json.loads(l) for l in open(args.data)]

    # group pairs by replay id
    by_replay = defaultdict(list)
    for r in rows:
        rid = r.get("meta", {}).get("replay_id", "unknown")
        by_replay[rid].append(r)

    replay_ids = list(by_replay.keys())
    random.Random(args.seed).shuffle(replay_ids)

    n_test_replays = max(1, int(len(replay_ids) * args.test_frac))
    test_ids = set(replay_ids[:n_test_replays])
    train_ids = set(replay_ids[n_test_replays:])

    train_rows = [r for rid in train_ids for r in by_replay[rid]]
    test_rows = [r for rid in test_ids for r in by_replay[rid]]

    train_path = args.data.replace(".jsonl", "_split_train.jsonl")
    test_path = args.data.replace(".jsonl", "_split_test.jsonl")
    # if the input wasn't named *.jsonl, fall back to fixed names
    if train_path == args.data:
        train_path, test_path = "split_train.jsonl", "split_test.jsonl"

    with open(train_path, "w") as f:
        for r in train_rows:
            f.write(json.dumps(r) + "\n")
    with open(test_path, "w") as f:
        for r in test_rows:
            f.write(json.dumps(r) + "\n")

    print(f"[split] {len(replay_ids)} replays -> "
          f"{len(train_ids)} train / {len(test_ids)} test")
    print(f"[split] {len(train_rows)} train pairs -> {train_path}")
    print(f"[split] {len(test_rows)} test pairs  -> {test_path}")

    # sanity: confirm zero replay overlap between splits
    overlap = train_ids & test_ids
    print(f"[split] replay overlap (must be 0): {len(overlap)}")


# --------------------------------------------------------------------------- #
# eval
# --------------------------------------------------------------------------- #
def _parse_action(text: str):
    """Extract {'action':..., 'value':...} from model output. Returns None on
    failure (a failure to produce valid JSON is itself a model error we count)."""
    try:
        m = re.search(r"\{.*?\}", text, re.DOTALL)
        if not m:
            # model sometimes emits bare  "action": "move", "value": "X"
            am = re.search(r'"action"\s*:\s*"(\w+)"', text)
            vm = re.search(r'"value"\s*:\s*"([^"]+)"', text)
            if am and vm:
                return {"action": am.group(1), "value": vm.group(1)}
            return None
        obj = json.loads(m.group(0))
        if "action" in obj and "value" in obj:
            return {"action": str(obj["action"]), "value": str(obj["value"])}
    except Exception:
        # try the bare-fields fallback before giving up
        am = re.search(r'"action"\s*:\s*"(\w+)"', text)
        vm = re.search(r'"value"\s*:\s*"([^"]+)"', text)
        if am and vm:
            return {"action": am.group(1), "value": vm.group(1)}
    return None


def _norm(s: str) -> str:
    return re.sub(r"[\s\-_.]", "", s.lower())


def eval_adapter(args):
    from unsloth import FastLanguageModel
    import torch

    model, tokenizer = FastLanguageModel.from_pretrained(
        args.adapter, max_seq_length=512, load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    rows = [json.loads(l) for l in open(args.test)]
    if args.limit > 0:
        rows = rows[:args.limit]

    n = 0
    aa_correct = 0      # action type matches
    em_correct = 0      # exact action+value matches
    parse_fail = 0      # model produced unparseable output
    per_class_total = Counter()
    per_class_em = Counter()
    examples = []

    for r in rows:
        prompt = r["prompt"]
        gold = json.loads(r["completion"])
        gold_action = str(gold["action"])
        gold_value = str(gold["value"])

        # FROZEN FORMAT: chat template + generation prompt, identical to training
        messages = [{"role": "user", "content": prompt}]
        templated = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(templated, return_tensors="pt").to("cuda")
        out = model.generate(**inputs, max_new_tokens=48, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True)

        pred = _parse_action(text)
        n += 1
        per_class_total[gold_action] += 1

        if pred is None:
            parse_fail += 1
        else:
            if pred["action"] == gold_action:
                aa_correct += 1
            if (pred["action"] == gold_action
                    and _norm(pred["value"]) == _norm(gold_value)):
                em_correct += 1
                per_class_em[gold_action] += 1

        if len(examples) < 8:
            examples.append((gold, pred, text.strip()[:60]))

    print("\n" + "=" * 60)
    print(f"EVAL on {n} held-out decisions  (adapter: {args.adapter})")
    print("=" * 60)
    print(f"Action Accuracy (AA, move-vs-switch): {aa_correct/n:.1%}")
    print(f"Exact Match    (EM, exact action)   : {em_correct/n:.1%}")
    print(f"Parse failures (invalid output)     : {parse_fail/n:.1%}")
    print("\nPer-class Exact Match:")
    for cls in per_class_total:
        tot = per_class_total[cls]
        em = per_class_em[cls]
        print(f"  {cls:7s}: {em}/{tot} = {em/tot:.1%}  "
              f"({tot/n:.0%} of test set)")
    print("\nBaseline to beat: always-'move' would score "
          f"AA={per_class_total['move']/n:.1%}")
    print("\nExample predictions (gold | pred | raw):")
    for gold, pred, raw in examples:
        print(f"  gold={json.dumps(gold)}")
        print(f"  pred={json.dumps(pred)}  raw={raw!r}\n")
    print("=" * 60)
    print("READ: if EM >> always-move baseline, the model learned real")
    print("strategy, not just the majority class. Low switch-EM with high")
    print("move-EM means it under-switches (the PokerBench 'over-passive' leak).")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("split")
    s.add_argument("--data", default="sft_train_clean.jsonl")
    s.add_argument("--test-frac", type=float, default=0.1)
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(func=split)

    e = sub.add_parser("eval")
    e.add_argument("--adapter", required=True, help="path to trained LoRA adapter dir")
    e.add_argument("--test", default="sft_train_clean_split_test.jsonl")
    e.add_argument("--limit", type=int, default=0, help="0 = full test set")
    e.set_defaults(func=eval_adapter)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
