"""
Stage C — GRPO update loop (the actual learning)
================================================

Builds on Stage A (constrained-scoring player) + Stage B (group advantages).
This is the finish line: roll a group, recompute each chosen action's logprob
under the CURRENT policy WITH GRADIENTS, compute the GRPO loss against the
group-relative advantages, KL-penalize against the frozen SFT reference, and
update the LoRA. Loop over groups, logging win rate so you can watch it climb.

GRPO LOSS (per decision):
    L = -( advantage * logπ_θ(action) )  +  β * KL( π_θ || π_ref )
  - advantage: group-normalized terminal reward (Stage B), shared by all
    decisions in a trajectory.
  - logπ_θ(action): summed token-logprob of the chosen legal action under the
    trainable policy (constrained-scoring, now with grad).
  - KL term keeps the policy from drifting far from the SFT reference (which
    preserves what SFT learned). Approximated per-decision with the same
    chosen-action logprobs under policy vs reference.

DESIGN FOR DEBUGGABILITY:
  - runs ONE update step, prints loss / grad-norm / win rate, and checkpoints,
    BEFORE looping. A bug (grad flow, indexing, OOM) surfaces on step 1.
  - degenerate groups (all-win/all-loss -> zero advantage) are skipped.

THROUGHPUT REALITY: one step = one group = group_size battles. On a T4 expect
a few minutes per step, so a few dozen steps per session. Enough to DEMONSTRATE
learning (the paper's goal), not to fully train a champion. Checkpoints let you
resume across sessions.

USAGE:
  python rl_stage_c.py --adapter /kaggle/working/sft_qwen_balanced \\
      --opponent maxpower --group-size 8 --temperature 0.6 \\
      --steps 5 --lr 1e-5 --kl-beta 0.05 --out grpo_ckpt
"""

import argparse
import asyncio
import statistics

from rl_stage_a import _build_prompt, Trajectory, DecisionRecord
from rl_stage_b import compute_group_advantages, compute_decision_advantages


# --------------------------------------------------------------------------- #
# Re-score a recorded decision under a given model, WITH gradients.
# Mirrors Stage A's constrained scoring but returns a differentiable logprob
# for the single chosen action (we already know which action was taken).
# --------------------------------------------------------------------------- #
def chosen_action_logprob(model, tokenizer, prompt, completion, requires_grad):
    import torch
    messages = [{"role": "user", "content": prompt}]
    base = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    base_ids = tokenizer(base, return_tensors="pt")["input_ids"][0]
    comp_ids = tokenizer(completion, add_special_tokens=False,
                         return_tensors="pt")["input_ids"][0]
    seq = torch.cat([base_ids, comp_ids]).unsqueeze(0).to("cuda")
    base_len = base_ids.shape[0]
    clen = comp_ids.shape[0]

    ctx = torch.enable_grad() if requires_grad else torch.no_grad()
    with ctx:
        logits = model(input_ids=seq).logits[0]      # [seq, vocab]
        logprobs = torch.log_softmax(logits, dim=-1)
        total = 0.0
        for j in range(clen):
            pos = base_len + j
            tok = seq[0, pos]
            total = total + logprobs[pos - 1, tok]
    return total   # scalar tensor (grad if requires_grad)


async def roll_group(model, tokenizer, group_size, temperature, opponent_cls):
    """Play a group with the CURRENT policy (no grad), return trajectories.
    Reuses Stage A's player via make_llm_player."""
    from rl_stage_a import make_llm_player
    from ou_team import BATTLE_FORMAT, OU_TEAM
    LLMPlayer = make_llm_player(model, tokenizer, temperature=temperature)
    llm = LLMPlayer(max_concurrent_battles=1,
                    battle_format=BATTLE_FORMAT, team=OU_TEAM)
    opp = opponent_cls(max_concurrent_battles=1,
                       battle_format=BATTLE_FORMAT, team=OU_TEAM)
    await llm.battle_against(opp, n_battles=group_size)
    return [t for t in llm.trajectories.values() if t.won is not None]


def grpo_step(policy, ref, tokenizer, scored, optimizer, kl_beta):
    """One GRPO gradient update over the group's (decision, advantage) pairs.

    scored: list of (trajectory, reward, advantage) from compute_group_advantages.
    Returns (loss_value, n_decisions, grad_norm).

    NOTE on update magnitude (the fix): earlier we divided the accumulated grad
    by the decision count n (~550) AND clipped to norm 1.0. That combination
    made the effective step size ~= lr regardless of lr (the clip sets the norm,
    lr only scales a unit vector), so raising lr did nothing — weight deltas
    stayed ~1e-5 across runs. Fix: average by NUMBER OF TRAJECTORIES (the GRPO
    unit), not per-decision, so within-trajectory gradients sum (reinforcing the
    shared advantage) instead of being washed out; and clip to a loose 10.0 so
    the real gradient magnitude flows through and lr genuinely controls the step.
    """
    import torch
    optimizer.zero_grad()
    total_loss = 0.0
    n_dec = 0
    n_traj = 0

    for traj, _r, adv in scored:
        if adv == 0.0:           # degenerate group contributes nothing
            continue
        n_traj += 1
        for d in traj.decisions:
            lp_policy = chosen_action_logprob(
                policy, tokenizer, d.prompt, d.action_text, requires_grad=True)
            with torch.no_grad():
                lp_ref = chosen_action_logprob(
                    ref, tokenizer, d.prompt, d.action_text, requires_grad=False)

            pg = -adv * lp_policy
            kl = kl_beta * (lp_policy - lp_ref)
            loss = pg + kl

            loss.backward()
            total_loss += loss.item()
            n_dec += 1

    if n_dec == 0:
        return 0.0, 0, 0.0

    # average by trajectory count (GRPO's unit), NOT per-decision: this keeps
    # the advantage-weighted signal intact instead of dividing it ~550x away.
    if n_traj > 0:
        for p in policy.parameters():
            if p.grad is not None:
                p.grad /= n_traj
    # clip=1.0 for STABILITY: the raw grad norm here swings wildly (300-1300),
    # so clip=10 let step magnitude vary chaotically -> unstable. Clip=1.0
    # normalizes every step to unit direction; lr alone then sets magnitude.
    # Use a LARGER lr (e.g. 3e-4) since step size is now ~lr, not lr*norm.
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()
    return total_loss / n_dec, n_dec, float(grad_norm)


def grpo_step_dense(policy, ref, tokenizer, per_traj, optimizer, kl_beta):
    """Dense GRPO update: per-DECISION advantages (from shaped rewards).

    per_traj: list of (trajectory, [adv_per_decision]) from
    compute_decision_advantages. Each decision is weighted by its OWN shaped
    advantage, so a good move in a lost game gets positive signal. This is the
    lower-variance update that dense shaping enables.
    """
    import torch
    optimizer.zero_grad()
    total_loss = 0.0
    n_dec = 0
    n_traj = 0

    for traj, advs in per_traj:
        if not advs or all(a == 0.0 for a in advs):
            continue
        n_traj += 1
        for d, adv in zip(traj.decisions, advs):
            lp_policy = chosen_action_logprob(
                policy, tokenizer, d.prompt, d.action_text, requires_grad=True)
            with torch.no_grad():
                lp_ref = chosen_action_logprob(
                    ref, tokenizer, d.prompt, d.action_text, requires_grad=False)
            pg = -adv * lp_policy
            kl = kl_beta * (lp_policy - lp_ref)
            loss = pg + kl
            loss.backward()
            total_loss += loss.item()
            n_dec += 1

    if n_dec == 0:
        return 0.0, 0, 0.0
    if n_traj > 0:
        for p in policy.parameters():
            if p.grad is not None:
                p.grad /= n_traj
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()
    return total_loss / n_dec, n_dec, float(grad_norm)


async def run(args):
    import torch
    from unsloth import FastLanguageModel
    from peft import PeftModel
    from poke_env.player import (
        SimpleHeuristicsPlayer, MaxBasePowerPlayer, RandomPlayer)

    opp_map = {"random": RandomPlayer, "maxpower": MaxBasePowerPlayer,
               "heuristic": SimpleHeuristicsPlayer}
    opponent_cls = opp_map[args.opponent]

    # ---- policy: SFT adapter, LoRA trainable ----
    print(f"[stageC] loading policy from {args.adapter} ...")
    policy, tokenizer = FastLanguageModel.from_pretrained(
        args.adapter, max_seq_length=512, load_in_4bit=True)
    FastLanguageModel.for_training(policy)   # enable grad on LoRA

    # ---- reference: same SFT adapter, frozen (for KL) ----
    print(f"[stageC] loading frozen reference ...")
    ref, _ = FastLanguageModel.from_pretrained(
        args.adapter, max_seq_length=512, load_in_4bit=True)
    FastLanguageModel.for_inference(ref)
    for p in ref.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad], lr=args.lr)

    print(f"[stageC] opponent={args.opponent}  group={args.group_size}  "
          f"temp={args.temperature}  lr={args.lr}  kl_beta={args.kl_beta}")
    print(f"[stageC] running {args.steps} GRPO steps "
          f"(1 step = {args.group_size} battles)\n")

    win_history = []
    for step in range(args.steps):
        # 1. roll a group with current policy (inference mode for rollout)
        FastLanguageModel.for_inference(policy)
        trajs = await roll_group(policy, tokenizer, args.group_size,
                                 args.temperature, opponent_cls)
        FastLanguageModel.for_training(policy)

        if not trajs:
            print(f"step {step+1}: no finished battles, skipping")
            continue

        # win rate is still reported from terminal outcomes either way
        wins = sum(1 for t in trajs if t.won)
        win_rate = wins / len(trajs)
        win_history.append(win_rate)

        if args.dense:
            # dense path: per-decision shaped rewards + decision-level advantages
            per_traj, (rmean, rstd) = compute_decision_advantages(
                trajs, hp_w=args.hp_weight, faint_w=args.faint_weight)
            if not per_traj or all(all(a == 0.0 for a in advs)
                                   for _, advs in per_traj):
                print(f"step {step+1}: win {wins}/{len(trajs)} -> "
                      f"no advantage signal, skipping")
                continue
            loss, n_dec, grad_norm = grpo_step_dense(
                policy, ref, tokenizer, per_traj, optimizer, args.kl_beta)
            print(f"step {step+1}: win {wins}/{len(trajs)} ({win_rate:.0%})  "
                  f"loss {loss:.4f}  grad_norm {grad_norm:.1f}  "
                  f"dec {n_dec}  r(mu={rmean:.2f},sd={rstd:.2f})")
        else:
            # sparse path: terminal-only, one advantage per trajectory
            scored = compute_group_advantages(trajs)
            if all(adv == 0.0 for _, _, adv in scored):
                print(f"step {step+1}: win {wins}/{len(trajs)} -> "
                      f"degenerate group (no advantage signal), skipping update")
                continue
            loss, n_dec, grad_norm = grpo_step(
                policy, ref, tokenizer, scored, optimizer, args.kl_beta)
            print(f"step {step+1}: win {wins}/{len(trajs)} ({win_rate:.0%})  "
                  f"loss {loss:.4f}  grad_norm {grad_norm:.3f}  decisions {n_dec}")

        # 3. checkpoint after the FIRST successful step, then periodically
        if step == 0 or (step + 1) % args.save_every == 0:
            ckpt = f"{args.out}_step{step+1}"
            policy.save_pretrained(ckpt)
            print(f"         checkpoint -> {ckpt}/")

            # after the FIRST checkpoint, measure how much the LoRA actually
            # moved vs the SFT start. A change ~1e-4 means the update is too
            # small to flip decisions (greedy EM won't budge) -> raise lr.
            # A change ~1e-2+ means the policy is moving meaningfully.
            if step == 0:
                try:
                    from safetensors.torch import load_file
                    base = load_file(f"{args.adapter}/adapter_model.safetensors")
                    new = load_file(f"{ckpt}/adapter_model.safetensors")
                    mx = max((base[k] - new[k]).abs().max().item()
                             for k in base if k in new)
                    print(f"         [weight-delta after step 1] max |Δ| = {mx:.2e}")
                    if mx < 1e-3:
                        print(f"         WARNING: tiny update ({mx:.2e}). Greedy "
                              f"behavior likely unchanged — consider raising --lr.")
                    else:
                        print(f"         OK: policy moving meaningfully.")
                except Exception as e:
                    print(f"         [weight-delta check skipped: {e}]")

    # ---- summary ----
    print("\n" + "=" * 60)
    print("STAGE C SUMMARY")
    print("=" * 60)
    if win_history:
        print(f"steps with battles: {len(win_history)}")
        print(f"win rate per step: "
              f"{', '.join(f'{w:.0%}' for w in win_history)}")
        if len(win_history) >= 2:
            early = statistics.mean(win_history[:max(1, len(win_history)//2)])
            late = statistics.mean(win_history[len(win_history)//2:])
            print(f"early-half avg: {early:.1%}  late-half avg: {late:.1%}  "
                  f"-> {'UP' if late > early else 'flat/down'}")
    policy.save_pretrained(f"{args.out}_final")
    print(f"final policy -> {args.out}_final/")
    print("\nREAD: watch win rate trend across steps. Going UP = GRPO is")
    print("learning to win. Flat over few steps is normal (high variance,")
    print("few updates). Compare {args.out}_final vs the SFT baseline with")
    print("split_eval.py and head-to-head battles for the paper's RL result.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="SFT adapter to start from")
    ap.add_argument("--opponent", default="maxpower",
                    choices=["random", "maxpower", "heuristic"])
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--kl-beta", type=float, default=0.05,
                    help="KL penalty coefficient vs SFT reference")
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--out", default="grpo_ckpt")
    ap.add_argument("--dense", action="store_true",
                    help="use dense per-decision shaped rewards (HP + faints) "
                         "instead of terminal-only")
    ap.add_argument("--hp-weight", type=float, default=1.0,
                    help="weight on per-turn HP differential (dense mode)")
    ap.add_argument("--faint-weight", type=float, default=0.3,
                    help="weight on per-turn faint differential (dense mode)")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
