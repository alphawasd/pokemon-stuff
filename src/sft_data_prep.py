"""
SFT Data Prep for a Gen 9 OU Showdown agent
===========================================

Turns public high-Elo Pokemon Showdown replays into supervised fine-tuning
pairs of the form  (state the player saw)  ->  (action they took).

WHY THIS IS THE HARD PART (read before running)
-----------------------------------------------
A replay is a SPECTATOR log. It records what happened, not what each player
knew at decision time, and not their reasoning. So we must *replay the log
forward* and, at each point a player acted, reconstruct an APPROXIMATION of
that player's information state. This approximation is imperfect by nature:

  - A spectator log hides a player's un-revealed bench Pokemon until they
    switch in. So early-game "available switches" are only partially known.
  - HP in public replays is shown as a percentage (e.g. 64/100), not exact.
  - Items/abilities/EV spreads are hidden until revealed by an effect.

That's fine for a first SFT pass ,  strong-human action labels are still the
signal we want ,  but it means you MUST eyeball the produced pairs before
trusting them at scale. The PokerBench lesson applies: garbage labels train a
confident-but-wrong model. So this script is built to be INSPECTED first.

TWO STAGES, run separately
--------------------------
  Stage 1 (download): hit the replay search API, save raw replay JSON to disk.
      python sft_data_prep.py download --format gen9ou --pages 5 --min-rating 1500
  Stage 2 (parse): turn saved replays into state->action JSONL.
      python sft_data_prep.py parse --limit 1     # parse ONE, pretty-print it
      python sft_data_prep.py parse               # parse all, write JSONL

WORKFLOW
--------
  1. download a small batch (--pages 1) first to confirm the API works.
  2. parse --limit 1 and READ the output. Does the state make sense? Does the
     action match what the log says happened? If not, fix the parser before
     scaling. This is the "overfit one example" step.
  3. scale up downloads, then parse the full set into train.jsonl.

API NOTES (verified against smogon WEB-API.md)
----------------------------------------------
  - Replay search:  https://replay.pokemonshowdown.com/search.json?format=gen9ou
        paginate with &before=<uploadtime of last result>; 51 results/page.
  - Single replay:  https://replay.pokemonshowdown.com/<id>.json
        returns {"id","format","players","log","uploadtime","rating",...}
  - Be polite: sleep between requests. Do NOT hammer the public server.
"""

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

SEARCH_URL = "https://replay.pokemonshowdown.com/search.json"
REPLAY_URL = "https://replay.pokemonshowdown.com/{id}.json"
RAW_DIR = Path("replays_raw")
OUT_JSONL = Path("sft_train.jsonl")
HEADERS = {"User-Agent": "sft-data-prep-research/0.1 (educational use)"}


# Stage 1: download
def _get_json(url: str):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def download(args):
    RAW_DIR.mkdir(exist_ok=True)
    before = None
    n_saved = 0
    n_skipped_rating = 0

    for page in range(args.pages):
        url = f"{SEARCH_URL}?format={args.format}"
        if before is not None:
            url += f"&before={before}"
        try:
            results = _get_json(url)
        except Exception as e:
            print(f"[download] search page {page} failed: {e}")
            break

        if not results:
            print("[download] no more results.")
            break

        # The 51st result only signals 'more pages exist'; use it for the
        # cursor but don't double-process it.
        page_items = results[:50]
        for meta in page_items:
            rid = meta.get("id")
            rating = meta.get("rating")  # may be None for some replays
            # Filter on rating when present. Replays without a rating are
            # unranked games ,  skip them for a high-Elo SFT set.
            if args.min_rating > 0:
                if rating is None or rating < args.min_rating:
                    n_skipped_rating += 1
                    continue
            out_path = RAW_DIR / f"{rid}.json"
            if out_path.exists():
                continue
            try:
                replay = _get_json(REPLAY_URL.format(id=rid))
                out_path.write_text(json.dumps(replay))
                n_saved += 1
                print(f"[download] saved {rid} (rating={rating})")
            except Exception as e:
                print(f"[download] replay {rid} failed: {e}")
            time.sleep(args.sleep)  # be polite to the public server

        # pagination cursor = uploadtime of the last item in the full results
        before = results[-1].get("uploadtime")
        time.sleep(args.sleep)

    print(f"\n[download] done. saved={n_saved}, "
          f"skipped_low_rating={n_skipped_rating}, dir={RAW_DIR}/")


# Stage 2: parse log -> state/action pairs
#
# Showdown battle-log lines we care about (pipe-delimited). A minimal subset:
#   |player|p1|Username|avatar|rating
#   |poke|p1|Dragonite, L50, M|item            (team preview: a Pokemon exists)
#   |teampreview
#   |start
#   |turn|3
#   |switch|p1a: Dragonite|Dragonite, L50, M|100/100
#   |move|p1a: Dragonite|Earthquake|p2a: Kingambit
#   |-damage|p2a: Kingambit|45/100
#   |faint|p2a: Kingambit
#   |win|Username
#
# The action label for a player on a given turn is the FIRST committed action
# we see from them after a |turn| marker: either a |move| or a |switch|.
# (Switches that are forced by a faint are excluded ,  they aren't a free
# decision and would pollute the labels.)

class BattleState:
    """Tracks an approximation of what is known, replaying the log forward.

    We keep it deliberately small and legible. This is NOT a full battle
    engine ,  it's just enough state to describe a decision point for SFT.
    """
    def __init__(self):
        self.turn = 0
        # active[side] = species string currently in play
        self.active = {"p1": None, "p2": None}
        # team[side] = list of species revealed so far (team preview gives all
        # for the player's own side; opponent's fill in as they appear)
        self.team = {"p1": [], "p2": []}
        # hp[side] = {species: "cur/max" string as shown in log}
        self.hp = {"p1": {}, "p2": {}}
        self.players = {"p1": None, "p2": None}
        self.ratings = {"p1": None, "p2": None}
        self.last_faint_side = None  # to detect forced switches

    def species_of(self, ident: str) -> str:
        # ident like "p1a: Dragonite" -> "Dragonite"
        return ident.split(": ", 1)[1] if ": " in ident else ident

    def side_of(self, ident: str) -> str:
        return ident[:2]  # "p1a: ..." -> "p1"


def _clean_species(detail: str) -> str:
    # "Dragonite, L50, M" -> "Dragonite"
    return detail.split(",", 1)[0].strip()


def parse_one(replay: dict, verbose: bool = False):
    """Return a list of (state_dict, action_dict) pairs from one replay."""
    log = replay.get("log", "")
    pairs = []
    st = BattleState()

    # We record an action only when we can attribute it to a turn AND a side.
    # A "decision point" is the first move/switch a side makes in a turn.
    seen_action_this_turn = {"p1": False, "p2": False}

    for line in log.split("\n"):
        if not line.startswith("|"):
            continue
        parts = line.split("|")[1:]  # drop leading ""
        if not parts:
            continue
        tag = parts[0]

        if tag == "player" and len(parts) >= 3:
            side = parts[1]
            st.players[side] = parts[2]
            if len(parts) >= 5 and parts[4]:
                try:
                    st.ratings[side] = int(parts[4])
                except ValueError:
                    pass

        elif tag == "poke" and len(parts) >= 3:
            side = parts[1]
            sp = _clean_species(parts[2])
            if sp not in st.team[side]:
                st.team[side].append(sp)

        elif tag == "turn":
            st.turn = int(parts[1])
            seen_action_this_turn = {"p1": False, "p2": False}

        elif tag == "switch" and len(parts) >= 4:
            ident, detail, hp = parts[1], parts[2], parts[3]
            side = st.side_of(ident)
            sp = _clean_species(detail)
            st.active[side] = sp
            st.hp[side][sp] = hp
            if sp not in st.team[side]:
                st.team[side].append(sp)
            # A switch right after a faint on the SAME side is forced -> skip
            # as a decision label. Otherwise it's a voluntary switch decision.
            forced = (st.last_faint_side == side)
            if not forced and not seen_action_this_turn.get(side):
                state = _snapshot(st, side)
                action = {"action": "switch", "value": sp}
                pairs.append((state, action))
                seen_action_this_turn[side] = True
            st.last_faint_side = None

        elif tag == "move" and len(parts) >= 3:
            ident, move = parts[1], parts[2]
            side = st.side_of(ident)
            if not seen_action_this_turn.get(side):
                state = _snapshot(st, side)
                action = {"action": "move", "value": move}
                pairs.append((state, action))
                seen_action_this_turn[side] = True

        elif tag == "-damage" and len(parts) >= 3:
            ident, hp = parts[1], parts[2]
            side = st.side_of(ident)
            sp = st.species_of(ident)
            st.hp[side][sp] = hp.split(" ")[0]  # strip status like "45/100 brn"

        elif tag == "faint" and len(parts) >= 2:
            ident = parts[1]
            st.last_faint_side = st.side_of(ident)

    if verbose:
        _pretty_print(replay, pairs)
    return pairs


def _snapshot(st: BattleState, deciding_side: str) -> dict:
    """Describe the decision point from the deciding player's perspective."""
    opp = "p2" if deciding_side == "p1" else "p1"
    return {
        "turn": st.turn,
        "my_active": st.active[deciding_side],
        "my_active_hp": st.hp[deciding_side].get(st.active[deciding_side], "?"),
        "my_team_revealed": list(st.team[deciding_side]),
        "opp_active": st.active[opp],
        "opp_active_hp": st.hp[opp].get(st.active[opp], "?"),
        "opp_team_revealed": list(st.team[opp]),
        "my_rating": st.ratings[deciding_side],
    }


def _pretty_print(replay, pairs):
    print("=" * 70)
    print(f"Replay: {replay.get('id')}  format={replay.get('format')}  "
          f"rating={replay.get('rating')}")
    print(f"Players: {replay.get('players')}")
    print(f"Extracted {len(pairs)} decision points. First few:\n")
    for i, (state, action) in enumerate(pairs[:6]):
        print(f"--- decision {i} (turn {state['turn']}) ---")
        print(f"  STATE: {json.dumps(state)}")
        print(f"  ACTION: {json.dumps(action)}")
    print("=" * 70)
    print("\nINSPECT THIS: does each ACTION match what the player did, and is")
    print("the STATE a plausible description of what they could see? If a")
    print("'switch' looks forced (right after their own faint) it should NOT")
    print("be here. Fix the parser before scaling.\n")


def _state_to_prompt(state: dict) -> str:
    """Render a state dict into the prompt string the model will train on.
    Keep this identical to what you'll use at inference / in the RL env.

    NOTE: deliberately does NOT end with "Decision: ". An earlier version did,
    which created a second answer-boundary that collided with the chat
    template's assistant-turn marker and broke training (model emitted stubs /
    chatty prose). The chat template's assistant turn is now the SOLE answer
    boundary. Do not re-add a trailing cue here."""
    return (
        "You are an expert Gen 9 OU Pokemon battler. Given the state, respond "
        'with ONE line of JSON: {"action":"move"|"switch","value":"<name>"}.\n'
        f"Turn: {state['turn']}\n"
        f"Your active: {state['my_active']} (HP {state['my_active_hp']})\n"
        f"Your revealed team: {', '.join(state['my_team_revealed'])}\n"
        f"Opponent active: {state['opp_active']} (HP {state['opp_active_hp']})\n"
        f"Opponent revealed: {', '.join(state['opp_team_revealed'])}"
    )


def parse(args):
    files = sorted(RAW_DIR.glob("*.json"))
    if not files:
        print(f"[parse] no replays in {RAW_DIR}/ ,  run the download stage first.")
        return

    if args.limit == 1:
        # inspection mode: parse one, pretty-print, write nothing
        replay = json.loads(files[0].read_text())
        parse_one(replay, verbose=True)
        print("[parse] inspection mode ,  nothing written. "
              "Re-run without --limit 1 to write JSONL.")
        return

    n_pairs = 0
    n_replays = 0
    files_to_do = files if args.limit <= 0 else files[:args.limit]
    with OUT_JSONL.open("w") as f:
        for path in files_to_do:
            try:
                replay = json.loads(path.read_text())
            except Exception:
                continue
            # only keep sufficiently-rated games if rating is present
            rating = replay.get("rating")
            if args.min_rating > 0 and (rating is None or rating < args.min_rating):
                continue
            pairs = parse_one(replay)
            for state, action in pairs:
                record = {
                    "prompt": _state_to_prompt(state),
                    "completion": json.dumps(action),
                    "meta": {"replay_id": replay.get("id"), "rating": rating},
                }
                f.write(json.dumps(record) + "\n")
                n_pairs += 1
            n_replays += 1
    print(f"[parse] wrote {n_pairs} pairs from {n_replays} replays -> {OUT_JSONL}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download")
    d.add_argument("--format", default="gen9ou")
    d.add_argument("--pages", type=int, default=2, help="search pages (50 each)")
    d.add_argument("--min-rating", type=int, default=1500,
                   help="skip replays below this Elo (0 = keep all)")
    d.add_argument("--sleep", type=float, default=1.0,
                   help="seconds between requests; be polite")
    d.set_defaults(func=download)

    p = sub.add_parser("parse")
    p.add_argument("--limit", type=int, default=0,
                   help="1 = inspect one replay (prints, writes nothing); "
                        "0 = parse all; N>1 = first N replays")
    p.add_argument("--min-rating", type=int, default=1500)
    p.set_defaults(func=parse)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
