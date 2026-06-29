"""
RL evaluation harness — the paper's results section
====================================================

Two comparisons, both needed because win-rate-vs-opponent came out flat:

1. STATIC (action distribution): run split_eval-style metrics for several
   adapters on the SAME fresh test split. Shows whether GRPO shifted the
   action distribution (switch-EM / move-EM) away from the SFT baseline —
   "RL changed behavior" even if win rate didn't move.

2. HEAD-TO-HEAD (live): play adapter A directly against adapter B via the
   Stage A constrained-scoring player. If the GRPO model beats its own SFT
   starting point head-to-head, that's a POSITIVE result regardless of the
   flat win rate vs the scripted opponent.

NOTE ON COMPARABILITY: this regenerates a fresh split from current ladder
replays, so absolute numbers differ from the original SFT eval (that batch is
gone). Mitigation: evaluate ALL adapters on this SAME new split, so the
three-rung comparison (vanilla -> balanced -> grpo) is internally consistent.

USAGE:
  # static metrics for the three rungs on one fresh split:
  python rl_eval.py static --test sft_train_split_test.jsonl \\
      --adapters sft_qwen_vanilla sft_qwen_balanced grpo_run3_step35

  # head-to-head: GRPO vs its SFT starting point
  python rl_eval.py h2h --adapter-a grpo_run3_step35 \\
      --adapter-b sft_qwen_balanced --n-battles 20
"""

import argparse
import asyncio
import json
import re


# --------------------------------------------------------------------------- #
# STATIC: action-distribution metrics on a held-out split (no battles)
# Mirrors split_eval's metric definitions so numbers are comparable to the
# SFT-phase methodology.
# --------------------------------------------------------------------------- #
def _parse(text):
    try:
        m = re.search(r"\{.*?\}", text, re.DOTALL)
        if m:
            o = json.loads(m.group(0))
            return str(o.get("action", "")), str(o.get("value", ""))
    except Exception:
        pass
    return None, None


def eval_static(adapter, test_path):
    from unsloth import FastLanguageModel
    import torch

    model, tok = FastLanguageModel.from_pretrained(
        adapter, max_seq_length=512, load_in_4bit=True)
    FastLanguageModel.for_inference(model)

    rows = [json.loads(l) for l in open(test_path)]
    n = len(rows)
    aa_correct = em_correct = parse_fail = 0
    sw_tot = sw_em = mv_tot = mv_em = 0

    for r in rows:
        gold_a, gold_v = _parse(r["completion"])
        messages = [{"role": "user", "content": r["prompt"]}]
        templated = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(templated, return_tensors="pt").to("cuda")
        out = model.generate(**inputs, max_new_tokens=48, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        pred = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                          skip_special_tokens=True)
        pa, pv = _parse(pred)

        if pa is None:
            parse_fail += 1
        if pa == gold_a:
            aa_correct += 1
        if pa == gold_a and (pv or "").lower() == (gold_v or "").lower():
            em_correct += 1
        if gold_a == "switch":
            sw_tot += 1
            if pa == gold_a and (pv or "").lower() == (gold_v or "").lower():
                sw_em += 1
        elif gold_a == "move":
            mv_tot += 1
            if pa == gold_a and (pv or "").lower() == (gold_v or "").lower():
                mv_em += 1

    return {
        "adapter": adapter, "n": n,
        "AA": aa_correct / n, "EM": em_correct / n,
        "parse_fail": parse_fail / n,
        "switch_EM": (sw_em / sw_tot) if sw_tot else 0.0,
        "move_EM": (mv_em / mv_tot) if mv_tot else 0.0,
        "switch_frac_gold": sw_tot / n,
    }


def run_static(args):
    results = [eval_static(a, args.test) for a in args.adapters]
    print("\n" + "=" * 78)
    print("STATIC ACTION-DISTRIBUTION EVAL (same test split for all adapters)")
    print("=" * 78)
    print(f"{'adapter':<28} {'AA':>6} {'EM':>6} {'sw-EM':>7} {'mv-EM':>7} "
          f"{'pfail':>6}")
    for r in results:
        print(f"{r['adapter']:<28} {r['AA']:>6.1%} {r['EM']:>6.1%} "
              f"{r['switch_EM']:>7.1%} {r['move_EM']:>7.1%} "
              f"{r['parse_fail']:>6.1%}")
    print("=" * 78)
    print("READ: compare switch-EM/move-EM across rungs. If GRPO's row differs")
    print("from its SFT start, RL shifted the action distribution (a behavior")
    print("result) even when win rate vs the scripted opponent stayed flat.")


# --------------------------------------------------------------------------- #
# DRIFT: measure sub-argmax policy change between two adapters.
# EM only moves when the top action flips. A small weight change shifts the
# PROBABILITY mass without flipping argmax, so EM looks identical. This metric
# scores the gold action's logprob under each adapter and reports how the
# distribution moved — the change EM can't see.
# --------------------------------------------------------------------------- #
def _gold_action_logprob(model, tok, prompt, completion):
    import torch
    messages = [{"role": "user", "content": prompt}]
    base = tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=True)
    base_ids = tok(base, return_tensors="pt")["input_ids"][0]
    comp_ids = tok(completion, add_special_tokens=False,
                   return_tensors="pt")["input_ids"][0]
    seq = torch.cat([base_ids, comp_ids]).unsqueeze(0).to("cuda")
    blen, clen = base_ids.shape[0], comp_ids.shape[0]
    with torch.no_grad():
        logits = model(input_ids=seq).logits[0]
        logprobs = torch.log_softmax(logits, dim=-1)
        total = 0.0
        for j in range(clen):
            pos = blen + j
            total += logprobs[pos - 1, seq[0, pos]].item()
    return total


def run_drift(args):
    from unsloth import FastLanguageModel
    import json, statistics

    rows = [json.loads(l) for l in open(args.test)]

    def score_all(adapter):
        m, t = FastLanguageModel.from_pretrained(
            adapter, max_seq_length=512, load_in_4bit=True)
        FastLanguageModel.for_inference(m)
        out = []
        for r in rows:
            ga, gv = _parse(r["completion"])
            lp = _gold_action_logprob(m, t, r["prompt"], r["completion"])
            out.append((ga, lp))
        del m
        import torch, gc; gc.collect(); torch.cuda.empty_cache()
        return out

    print(f"[drift] scoring A = {args.adapter_a}")
    a = score_all(args.adapter_a)
    print(f"[drift] scoring B = {args.adapter_b}")
    b = score_all(args.adapter_b)

    # per-decision logprob deltas (B - A), split by gold action type
    d_all, d_switch, d_move = [], [], []
    for (ga, la), (_, lb) in zip(a, b):
        d = lb - la
        d_all.append(d)
        (d_switch if ga == "switch" else d_move).append(d)

    def summ(xs):
        if not xs:
            return "n/a"
        return (f"mean {statistics.mean(xs):+.4f}  median "
                f"{statistics.median(xs):+.4f}  n={len(xs)}")

    print("\n" + "=" * 70)
    print(f"DRIFT  B - A   (A={args.adapter_a}, B={args.adapter_b})")
    print("Δlogprob of the GOLD action under B vs A. + => B assigns the human")
    print("action MORE probability than A did.")
    print("=" * 70)
    print(f"all gold actions : {summ(d_all)}")
    print(f"gold = switch    : {summ(d_switch)}")
    print(f"gold = move      : {summ(d_move)}")
    # how many decisions shifted by a meaningful margin
    big = sum(1 for d in d_all if abs(d) > 0.05)
    pos = sum(1 for d in d_all if d > 0)
    print(f"\ndecisions shifted >0.05 logprob: {big}/{len(d_all)} "
          f"({big/len(d_all):.1%})")
    print(f"decisions where B raised gold-action prob: {pos}/{len(d_all)} "
          f"({pos/len(d_all):.1%})")
    print("=" * 70)
    print("READ: a systematic + shift on gold=switch (or gold=move) means RL")
    print("moved the policy toward the human distribution for that action type,")
    print("even though greedy EM was unchanged. Direction + magnitude = the")
    print("quantified RL effect for the paper.")
async def run_h2h(args):
    from unsloth import FastLanguageModel
    from rl_stage_a import make_llm_player
    from ou_team import BATTLE_FORMAT, TEAMS
    team = TEAMS[getattr(args, "team", "team1")]

    print(f"[h2h] loading A = {args.adapter_a}")
    model_a, tok_a = FastLanguageModel.from_pretrained(
        args.adapter_a, max_seq_length=512, load_in_4bit=True)
    FastLanguageModel.for_inference(model_a)

    print(f"[h2h] loading B = {args.adapter_b}")
    model_b, tok_b = FastLanguageModel.from_pretrained(
        args.adapter_b, max_seq_length=512, load_in_4bit=True)
    FastLanguageModel.for_inference(model_b)

    PlayerA = make_llm_player(model_a, tok_a, temperature=0.0)
    PlayerB = make_llm_player(model_b, tok_b, temperature=0.0)
    from poke_env.ps_client.account_configuration import AccountConfiguration
    a = PlayerA(account_configuration=AccountConfiguration("grpo_A", None),
                max_concurrent_battles=1,
                battle_format=BATTLE_FORMAT, team=team)
    b = PlayerB(account_configuration=AccountConfiguration("sft_B", None),
                max_concurrent_battles=1,
                battle_format=BATTLE_FORMAT, team=team)

    print(f"[h2h] {args.adapter_a} vs {args.adapter_b}, "
          f"{args.n_battles} battles (both greedy)")
    await a.battle_against(b, n_battles=args.n_battles)

    a_trajs = [t for t in a.trajectories.values() if t.won is not None]
    a_wins = sum(1 for t in a_trajs if t.won)
    print("\n" + "=" * 60)
    print("HEAD-TO-HEAD RESULT")
    print("=" * 60)
    print(f"{args.adapter_a}: {a_wins}/{len(a_trajs)} = {a_wins/max(len(a_trajs),1):.1%}")
    print(f"{args.adapter_b}: {len(a_trajs)-a_wins}/{len(a_trajs)} = "
          f"{(len(a_trajs)-a_wins)/max(len(a_trajs),1):.1%}")

    # --- battle-quality diagnostics: are wins real or degenerate? ---
    turns = [t.n_turns for t in a_trajs]
    decs = [len(t.decisions) for t in a_trajs]
    if turns:
        import statistics as _st
        short = sum(1 for x in turns if x <= 3)   # forfeit/timeout-like
        print("\n[battle quality]")
        print(f"  turns: min={min(turns)} median={int(_st.median(turns))} "
              f"max={max(turns)} mean={_st.mean(turns):.1f}")
        print(f"  decisions/battle: median={int(_st.median(decs))}")
        print(f"  suspiciously short battles (<=3 turns): {short}/{len(turns)}")
        print(f"  -> if most battles are normal length (~15-30 turns), the wins")
        print(f"     are real play, not forfeits/timeouts/degenerate loops.")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("static")
    s.add_argument("--test", required=True)
    s.add_argument("--adapters", nargs="+", required=True)

    h = sub.add_parser("h2h")
    h.add_argument("--adapter-a", required=True)
    h.add_argument("--adapter-b", required=True)
    h.add_argument("--n-battles", type=int, default=20)
    h.add_argument("--team", default="team1", choices=["team1", "team2"],
                   help="team1 = trained matchup; team2 = generalization test")

    dr = sub.add_parser("drift")
    dr.add_argument("--test", required=True)
    dr.add_argument("--adapter-a", required=True)
    dr.add_argument("--adapter-b", required=True)

    args = ap.parse_args()
    if args.cmd == "static":
        run_static(args)
    elif args.cmd == "drift":
        run_drift(args)
    else:
        asyncio.run(run_h2h(args))


if __name__ == "__main__":
    main()
