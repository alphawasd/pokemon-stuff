"""
Stage B ,  Group rollouts + GRPO advantages (no gradient update yet)
===================================================================

Builds on Stage A's validated constrained-scoring player. Adds the GRPO group
machinery and verifies the reward/advantage signal is sane BEFORE we wire in
the actual policy update (Stage C).

GRPO IN ONE PARAGRAPH
---------------------
No value network. Instead, for the same setup, sample a GROUP of G full
battles with a stochastic policy (temperature > 0) so the group explores
different action sequences. Each battle gets a scalar reward (terminal:
+1 win / -1 loss). The advantage of battle i is its reward normalized within
the group:  A_i = (r_i - mean(r)) / (std(r) + eps). Battles that beat the
group average get positive advantage, worse-than-average negative. Every
decision in a trajectory inherits that trajectory's advantage (credit
assignment: the whole game shares the outcome).

REWARD: terminal-only (+1/-1). Dense shaping is deliberately omitted ,  the
Pokemon-Red RL work shows shaping gets hacked; we want a clean signal first.

WHAT THIS STAGE PROVES
----------------------
  - we can run G battles and collect G trajectories with rewards,
  - group-relative advantages compute correctly (winners +, losers -),
  - the per-decision (logprob, advantage) pairs that Stage C's loss needs
    are all present and sane.
NO LEARNING. If the advantages look right, Stage C adds the GRPO loss.

USAGE:
  python rl_stage_b.py --adapter /kaggle/working/sft_qwen_balanced \\
      --group-size 8 --temperature 0.8 --n-groups 1
"""

import argparse
import asyncio
import statistics

# Reuse Stage A's player + trajectory machinery.
from rl_stage_a import make_llm_player, Trajectory, DecisionRecord


def compute_group_advantages(trajectories, eps=1e-4):
    """Terminal reward (+1 win / -1 loss) -> group-normalized advantage.

    Returns a list of (trajectory, reward, advantage). Every decision in a
    trajectory shares that trajectory's advantage. (Original sparse version.)
    """
    rewards = [1.0 if t.won else -1.0 for t in trajectories]
    mean = statistics.mean(rewards)
    std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
    out = []
    for t, r in zip(trajectories, rewards):
        adv = (r - mean) / (std + eps)
        out.append((t, r, adv))
    return out


def _decision_shaped_rewards(traj, terminal_w=1.0, hp_w=1.0, faint_w=0.3):
    """Per-DECISION shaped reward for one trajectory.

    Each decision gets:
        terminal_w * (+1 win / -1 loss)
      + hp_w   * (opp_hp_lost - my_hp_lost) THIS turn   [HP fractions, 0..6]
      + faint_w* (opp_fainted - my_fainted) THIS turn

    HP/faint deltas are computed between consecutive decisions. The terminal
    term stays dominant; shaping is small and tightly coupled to winning
    (deal damage, take less), which is hack-resistant. Returns a list of
    per-decision rewards aligned with traj.decisions.
    """
    term = terminal_w * (1.0 if traj.won else -1.0)
    decs = traj.decisions
    rewards = []
    for i, d in enumerate(decs):
        if i + 1 < len(decs):
            nxt = decs[i + 1]
            opp_hp_lost = d.opp_hp - nxt.opp_hp      # + good (we damaged them)
            my_hp_lost = d.my_hp - nxt.my_hp         # + bad  (we took damage)
            opp_fainted = d.opp_alive - nxt.opp_alive
            my_fainted = d.my_alive - nxt.my_alive
        else:
            opp_hp_lost = my_hp_lost = opp_fainted = my_fainted = 0.0
        shaped = (term
                  + hp_w * (opp_hp_lost - my_hp_lost)
                  + faint_w * (opp_fainted - my_fainted))
        rewards.append(shaped)
    return rewards


def compute_decision_advantages(trajectories, eps=1e-4, **shape_kw):
    """Dense version: per-decision shaped rewards, group-normalized across ALL
    decisions in the group.

    Returns (per_traj, flat_stats) where per_traj is a list of
    (trajectory, [adv_per_decision]) and flat_stats is (mean, std) for logging.
    Group-normalizing at the decision level (not trajectory level) is what
    lowers gradient variance: a good move in a lost game can get +advantage.
    """
    all_r = []
    per_traj_r = []
    for t in trajectories:
        r = _decision_shaped_rewards(t, **shape_kw)
        per_traj_r.append(r)
        all_r.extend(r)
    if not all_r:
        return [], (0.0, 0.0)
    mean = statistics.mean(all_r)
    std = statistics.pstdev(all_r) if len(all_r) > 1 else 0.0
    per_traj = []
    for t, r in zip(trajectories, per_traj_r):
        advs = [(ri - mean) / (std + eps) for ri in r]
        per_traj.append((t, advs))
    return per_traj, (mean, std)


async def run_one_group(adapter_model, adapter_tokenizer, group_size,
                        temperature, opponent_cls):
    """Play `group_size` battles with a stochastic policy; return trajectories."""
    LLMPlayer = make_llm_player(adapter_model, adapter_tokenizer,
                                temperature=temperature)
    from ou_team import BATTLE_FORMAT, OU_TEAM
    llm = LLMPlayer(max_concurrent_battles=1,
                    battle_format=BATTLE_FORMAT, team=OU_TEAM)
    opponent = opponent_cls(max_concurrent_battles=1,
                            battle_format=BATTLE_FORMAT, team=OU_TEAM)
    await llm.battle_against(opponent, n_battles=group_size)
    return [t for t in llm.trajectories.values() if t.won is not None]


async def run(args):
    from unsloth import FastLanguageModel
    from poke_env.player import (
        SimpleHeuristicsPlayer, MaxBasePowerPlayer, RandomPlayer)

    # opponent difficulty: random (easiest) -> maxpower -> heuristic (hardest).
    # Pick the one where your win rate is near 50%: that maximizes within-group
    # reward variance, which is exactly the GRPO learning signal. Too-hard ->
    # all-loss groups (zero advantage); too-easy -> all-win groups (same).
    opp_map = {
        "random": RandomPlayer,
        "maxpower": MaxBasePowerPlayer,
        "heuristic": SimpleHeuristicsPlayer,
    }
    opponent_cls = opp_map[args.opponent]
    print(f"[stageB] loading adapter {args.adapter} ...")
    print(f"[stageB] opponent: {args.opponent} ({opponent_cls.__name__})")
    model, tokenizer = FastLanguageModel.from_pretrained(
        args.adapter, max_seq_length=512, load_in_4bit=True)
    FastLanguageModel.for_inference(model)

    all_group_stats = []
    for g in range(args.n_groups):
        print(f"\n[stageB] === group {g+1}/{args.n_groups}: "
              f"{args.group_size} battles @ temp={args.temperature} "
              f"vs {args.opponent} ===")
        trajs = await run_one_group(
            model, tokenizer, args.group_size, args.temperature,
            opponent_cls)

        if not trajs:
            print("  no finished battles in this group ,  skipping.")
            continue

        scored = compute_group_advantages(trajs)
        wins = sum(1 for t, _, _ in scored if t.won)

        print(f"  wins: {wins}/{len(scored)}")
        print(f"  {'won':>5} {'reward':>7} {'advantage':>10} {'decisions':>10} "
              f"{'sum_logprob':>12}")
        for t, r, adv in scored:
            print(f"  {str(t.won):>5} {r:>7.1f} {adv:>10.3f} "
                  f"{len(t.decisions):>10} {t.total_logprob():>12.1f}")

        # sanity: winners should have positive advantage, losers negative
        win_advs = [adv for t, _, adv in scored if t.won]
        loss_advs = [adv for t, _, adv in scored if not t.won]
        ok = ((not win_advs or min(win_advs) > 0) and
              (not loss_advs or max(loss_advs) < 0))
        print(f"  advantage signal sane (winners>0, losers<0): {ok}")

        # build the (logprob, advantage) pairs Stage C will optimize
        n_pairs = sum(len(t.decisions) for t, _, _ in scored)
        all_group_stats.append((wins, len(scored), n_pairs, ok))

    # ---- summary ----
    print("\n" + "=" * 60)
    print("STAGE B SUMMARY")
    print("=" * 60)
    if all_group_stats:
        total_wins = sum(w for w, _, _, _ in all_group_stats)
        total_battles = sum(n for _, n, _, _ in all_group_stats)
        total_pairs = sum(p for _, _, p, _ in all_group_stats)
        all_sane = all(ok for _, _, _, ok in all_group_stats)
        print(f"groups: {len(all_group_stats)}  "
              f"battles: {total_battles}  wins: {total_wins} "
              f"({total_wins/total_battles:.1%})")
        print(f"total (decision, advantage) pairs for GRPO: {total_pairs}")
        print(f"all groups' advantage signal sane: {all_sane}")
        print("\nREAD: if advantages are sane (winners +, losers -) and we have")
        print("decision-level logprobs + advantages, Stage C can compute the")
        print("GRPO loss and update the LoRA. A group of all-wins or all-losses")
        print("gives ZERO advantage (no signal) ,  that's expected; more groups")
        print("/ a tougher or matched opponent fixes it.")
    else:
        print("no groups produced finished battles ,  check the server / adapter.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--group-size", type=int, default=8,
                    help="G: battles per GRPO group (relative advantage needs >1)")
    ap.add_argument("--temperature", type=float, default=0.8,
                    help=">0 so the group explores different action sequences")
    ap.add_argument("--n-groups", type=int, default=1)
    ap.add_argument("--opponent", default="maxpower",
                    choices=["random", "maxpower", "heuristic"],
                    help="training opponent; pick the one giving ~50% win rate "
                         "for maximal within-group variance (GRPO signal)")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
