"""
Stage A: trajectory-logging LLM player (foundation for online GRPO).

A poke-env Player that plays full Gen 9 OU battles with the SFT model and logs
everything GRPO needs for each decision: the prompt, the chosen action text,
token log-probabilities, HP state, and the win/loss outcome. No learning here;
this isolates the systems plumbing (LLM generation in poke-env's async loop,
mapping text actions to legal orders, capturing token log-probs, clean battle
outcomes) so Stage B and Stage C can build on it.

Format contract (must match SFT): the prompt is built by _build_prompt below;
the chat template is applied with add_generation_prompt=True at inference.

Prereqs: a local Showdown server (node pokemon-showdown start --no-security),
poke-env, unsloth, transformers, and an SFT adapter directory.

Usage:
  python rl_stage_a.py --adapter sft_qwen_balanced --n-battles 5
"""

import argparse
import asyncio
import json
import re
from dataclasses import dataclass, field


# Per-decision record ,  everything GRPO will consume later
@dataclass
class DecisionRecord:
    turn: int
    prompt: str                 # the user-content prompt (pre chat-template)
    action_text: str            # raw model output
    action_tokens: list         # generated token ids
    token_logprobs: list        # per-token log-prob under the policy at gen time
    chosen_kind: str            # "move" | "switch" | "fallback"
    legal: bool                 # was the parsed action legal / used as-is?
    # HP state AT this decision (fractions 0..1), for dense reward shaping:
    my_hp: float = 1.0          # sum of own team HP fractions
    opp_hp: float = 1.0         # sum of opponent team HP fractions (seen)
    my_alive: int = 6           # own mons not fainted
    opp_alive: int = 6          # opponent mons not fainted (seen)


@dataclass
class Trajectory:
    decisions: list = field(default_factory=list)
    won: bool = None
    n_turns: int = 0

    def total_logprob(self) -> float:
        return sum(sum(d.token_logprobs) for d in self.decisions)


# Prompt builder ,  LIVE version with LEGAL ACTIONS listed.
#
# This deliberately DIFFERS from the SFT prompt: it appends the actual legal
# moves and switch targets from the poke-env Battle. SFT taught a prior over
# Showdown play from replays (no legal lists); RL adapts that prior to legal,
# live decision-making. Listing the menu is what kills the ~97% fallback rate
# seen when the model guessed action names blind.
def _build_prompt(battle) -> str:
    active = battle.active_pokemon
    opp = battle.opponent_active_pokemon
    my_team = [p.species for p in battle.team.values()]
    opp_team = [p.species for p in battle.opponent_team.values()]
    hp_frac = (f"{int(active.current_hp_fraction*100)}/100"
               if active and active.current_hp_fraction is not None else "?")
    opp_hp = (f"{int(opp.current_hp_fraction*100)}/100"
              if opp and opp.current_hp_fraction is not None else "?")

    # the legal menu ,  the key addition vs the SFT prompt
    legal_moves = [m.id for m in battle.available_moves]
    legal_switches = [s.species for s in battle.available_switches]

    return (
        "You are an expert Gen 9 OU Pokemon battler. Choose ONE legal action.\n"
        'Respond with ONE line of JSON: {"action":"move"|"switch","value":"<name>"}.\n'
        "The value MUST be chosen from the legal lists below.\n"
        f"Turn: {battle.turn}\n"
        f"Your active: {active.species if active else '?'} (HP {hp_frac})\n"
        f"Your revealed team: {', '.join(my_team)}\n"
        f"Opponent active: {opp.species if opp else '?'} (HP {opp_hp})\n"
        f"Opponent revealed: {', '.join(opp_team)}\n"
        f"LEGAL moves: {', '.join(legal_moves) if legal_moves else 'none'}\n"
        f"LEGAL switches: {', '.join(legal_switches) if legal_switches else 'none'}"
    )


def _parse_action(text: str):
    try:
        m = re.search(r"\{.*?\}", text, re.DOTALL)
        if m:
            obj = json.loads(m.group(0))
            return str(obj.get("action", "")), str(obj.get("value", ""))
        am = re.search(r'"action"\s*:\s*"(\w+)"', text)
        vm = re.search(r'"value"\s*:\s*"([^"]+)"', text)
        if am and vm:
            return am.group(1), vm.group(1)
    except Exception:
        pass
    return None, None


def _norm(s: str) -> str:
    return re.sub(r"[\s\-_.]", "", s.lower())


# Build the LLM player class (factory so we can inject model/tokenizer)
def make_llm_player(model, tokenizer, temperature: float):
    import torch
    from poke_env.player import Player

    class LLMPlayer(Player):
        """Plays via CONSTRAINED SCORING over legal actions.

        Rather than free-generating an action string (which the model emits
        from its prior, ignoring the legal menu -> ~72% illegal), we enumerate
        the legal actions, score each candidate JSON completion under the
        model, and pick:
          - temperature == 0 : argmax (greedy / eval)
          - temperature  > 0 : softmax-sample over scores (RL exploration)
        The chosen candidate's total token-logprob is exactly the policy
        logprob GRPO needs. Fallback is impossible by construction.

        Trajectories are stored in self.trajectories keyed by battle tag.
        """
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.trajectories = {}

        def _candidate_actions(self, battle):
            """List legal actions as (kind, obj, completion_str) tuples."""
            cands = []
            for mv in battle.available_moves:
                comp = f'{{"action": "move", "value": "{mv.id}"}}'
                cands.append(("move", mv, comp))
            for sw in battle.available_switches:
                comp = f'{{"action": "switch", "value": "{sw.species}"}}'
                cands.append(("switch", sw, comp))
            return cands

        @torch.no_grad()
        def _score_candidates(self, prompt, candidates):
            """Return total logprob of each candidate completion under policy.

            We build [chat-templated prompt] + [candidate completion], run one
            forward pass per candidate, and sum the log-probs of the completion
            tokens (teacher-forced). Batched across candidates for speed.
            """
            messages = [{"role": "user", "content": prompt}]
            base = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            base_ids = tokenizer(base, return_tensors="pt")["input_ids"][0]
            base_len = base_ids.shape[0]

            seqs, comp_lens = [], []
            for _, _, comp in candidates:
                comp_ids = tokenizer(comp, add_special_tokens=False,
                                     return_tensors="pt")["input_ids"][0]
                seqs.append(torch.cat([base_ids, comp_ids]))
                comp_lens.append(comp_ids.shape[0])

            # left-pad to a batch
            maxlen = max(s.shape[0] for s in seqs)
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            batch = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
            attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
            for i, s in enumerate(seqs):
                batch[i, maxlen - s.shape[0]:] = s
                attn[i, maxlen - s.shape[0]:] = 1
            batch, attn = batch.to("cuda"), attn.to("cuda")

            logits = model(input_ids=batch, attention_mask=attn).logits
            logprobs = torch.log_softmax(logits, dim=-1)

            scores, per_tok = [], []
            for i, clen in enumerate(comp_lens):
                # completion occupies the last clen positions; the logit that
                # predicts token t sits at position t-1.
                total, toks = 0.0, []
                for j in range(clen):
                    pos = maxlen - clen + j          # position of the comp token
                    tok = batch[i, pos]
                    lp = logprobs[i, pos - 1, tok].item()
                    total += lp
                    toks.append(lp)
                scores.append(total)
                per_tok.append(toks)
            return scores, per_tok

        def choose_move(self, battle):
            prompt = _build_prompt(battle)
            candidates = self._candidate_actions(battle)

            # no legal actions (rare: forced/empty) -> let poke-env decide
            if not candidates:
                return self.choose_random_move(battle)

            scores, per_tok = self._score_candidates(prompt, candidates)
            scores_t = torch.tensor(scores)

            if temperature <= 0:
                idx = int(torch.argmax(scores_t))
            else:
                probs = torch.softmax(scores_t / temperature, dim=-1)
                idx = int(torch.multinomial(probs, 1))

            kind, obj, comp = candidates[idx]
            order = self.create_order(obj)

            # capture HP state for dense reward shaping. Sum of HP fractions
            # across the team (unseen opponent mons count as full = 1.0).
            def _team_hp(team):
                tot, alive = 0.0, 0
                for p in team.values():
                    frac = p.current_hp_fraction
                    if frac is None:
                        frac = 0.0 if p.fainted else 1.0
                    tot += frac
                    if not p.fainted:
                        alive += 1
                return tot, alive
            my_hp, my_alive = _team_hp(battle.team)
            opp_hp, opp_alive = _team_hp(battle.opponent_team)

            traj = self.trajectories.setdefault(battle.battle_tag, Trajectory())
            traj.decisions.append(DecisionRecord(
                turn=battle.turn, prompt=prompt, action_text=comp,
                action_tokens=[], token_logprobs=per_tok[idx],
                chosen_kind=kind, legal=True,   # always legal by construction
                my_hp=my_hp, opp_hp=opp_hp,
                my_alive=my_alive, opp_alive=opp_alive,
            ))
            return order

        def _battle_finished_callback(self, battle):
            traj = self.trajectories.setdefault(battle.battle_tag, Trajectory())
            traj.won = battle.won
            traj.n_turns = battle.turn
            super()._battle_finished_callback(battle)

    return LLMPlayer


async def run(args):
    from unsloth import FastLanguageModel
    from poke_env.player import SimpleHeuristicsPlayer

    print(f"[stageA] loading adapter {args.adapter} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        args.adapter, max_seq_length=512, load_in_4bit=True)
    FastLanguageModel.for_inference(model)

    from ou_team import BATTLE_FORMAT, OU_TEAM
    LLMPlayer = make_llm_player(model, tokenizer, temperature=args.temperature)
    llm = LLMPlayer(max_concurrent_battles=1,
                    battle_format=BATTLE_FORMAT, team=OU_TEAM)
    opponent = SimpleHeuristicsPlayer(max_concurrent_battles=1,
                                      battle_format=BATTLE_FORMAT, team=OU_TEAM)

    print(f"[stageA] playing {args.n_battles} battles vs SimpleHeuristicsPlayer ...")
    await llm.battle_against(opponent, n_battles=args.n_battles)

    # ---- report ----
    trajs = [t for t in llm.trajectories.values() if t.won is not None]
    wins = sum(1 for t in trajs if t.won)
    print("\n" + "=" * 60)
    print(f"STAGE A RESULT  (adapter: {args.adapter}, temp={args.temperature})")
    print("=" * 60)
    print(f"battles: {len(trajs)}  wins: {wins}  win rate: {wins/max(len(trajs),1):.1%}")

    # trajectory sanity: do we actually have logprobs + decisions?
    if trajs:
        t0 = trajs[0]
        n_dec = len(t0.decisions)
        n_switch = sum(1 for d in t0.decisions if d.chosen_kind == "switch")
        print(f"\nfirst battle: {n_dec} decisions, {t0.n_turns} turns, "
              f"won={t0.won}")
        print(f"  switches chosen: {n_switch}/{n_dec}")
        if t0.decisions:
            d = t0.decisions[0]
            print(f"  sample decision: kind={d.chosen_kind}")
            print(f"    chosen action: {d.action_text}")
            print(f"    n_tokens: {len(d.token_logprobs)}  "
                  f"sum_logprob: {sum(d.token_logprobs):.2f}")
        print(f"  total trajectory logprob: {t0.total_logprob():.2f}")

    # aggregate behaviour signal across all battles
    all_dec = [d for t in trajs for d in t.decisions]
    if all_dec:
        sw = sum(1 for d in all_dec if d.chosen_kind == "switch")
        print(f"\nacross all battles: {len(all_dec)} decisions, "
              f"{sw/len(all_dec):.1%} switch (all legal by construction)")
    print("=" * 60)
    print("READ: all actions are legal now (constrained scoring). If win rate")
    print("is sane, switch rate is nonzero/plausible, and logprobs are present")
    print("-> plumbing works. Ready for Stage B (group rollouts + advantages).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="path to SFT LoRA adapter")
    ap.add_argument("--n-battles", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="0 = greedy; >0 needed for GRPO group diversity later")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
