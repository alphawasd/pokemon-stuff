"""
Shared Gen 9 OU config for matched SFT/RL experiments.
=====================================================

The RL phase originally defaulted to gen9randombattle (poke-env's default when
no battle_format is given), which did NOT match the gen9ou SFT distribution.
This module pins BOTH the format and a fixed, legal Gen 9 OU team so the RL
phase runs on the same format the policy was supervised on.

Single fixed team on both sides => a mirror match. This removes team imbalance
as a variable and matches the OU format (mechanics, Tera, banlist), at the cost
of not covering OU's full team diversity (noted as a limitation in the paper).

The team below is a standard, legal Gen 9 OU sample (balance archetype).
"""

BATTLE_FORMAT = "gen9ou"

# A legal Gen 9 OU team in Showdown export format.
OU_TEAM = """
Great Tusk @ Heavy-Duty Boots
Ability: Protosynthesis
Tera Type: Steel
EVs: 252 Atk / 4 Def / 252 Spe
Jolly Nature
- Headlong Rush
- Close Combat
- Rapid Spin
- Stealth Rock

Kingambit @ Leftovers
Ability: Supreme Overlord
Tera Type: Fire
EVs: 252 Atk / 4 Def / 252 Spe
Adamant Nature
- Swords Dance
- Kowtow Cleave
- Sucker Punch
- Iron Head

Gholdengo @ Choice Scarf
Ability: Good as Gold
Tera Type: Steel
EVs: 252 SpA / 4 Def / 252 Spe
Timid Nature
- Make It Rain
- Shadow Ball
- Focus Blast
- Trick

Dragapult @ Choice Specs
Ability: Infiltrator
Tera Type: Ghost
EVs: 252 SpA / 4 Def / 252 Spe
Timid Nature
- Shadow Ball
- Draco Meteor
- Flamethrower
- U-turn

Slowking-Galar @ Assault Vest
Ability: Regenerator
Tera Type: Water
EVs: 252 HP / 16 SpA / 240 SpD
Calm Nature
- Future Sight
- Sludge Bomb
- Flamethrower
- Chilly Reception

Corviknight @ Leftovers
Ability: Pressure
Tera Type: Dragon
EVs: 252 HP / 4 Atk / 252 Def
Impish Nature
- Body Press
- Brave Bird
- Roost
- Defog
""".strip()

# A SECOND, distinct legal Gen 9 OU team (different archetype: rain/offense)
# for the generalization head-to-head. If step5 beats SFT on THIS team too
# (a matchup it was never trained on), the GRPO gain is transferable skill,
# not memorised lines for OU_TEAM.
OU_TEAM_2 = """
Pelipper @ Damp Rock
Ability: Drizzle
Tera Type: Ground
EVs: 248 HP / 252 Def / 8 SpD
Bold Nature
- Hurricane
- Surf
- U-turn
- Roost

Barraskewda @ Choice Band
Ability: Swift Swim
Tera Type: Water
EVs: 252 Atk / 4 Def / 252 Spe
Adamant Nature
- Liquidation
- Flip Turn
- Close Combat
- Aqua Jet

Zapdos @ Heavy-Duty Boots
Ability: Static
Tera Type: Steel
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Hurricane
- Volt Switch
- Heat Wave
- Roost

Iron Valiant @ Booster Energy
Ability: Quark Drive
Tera Type: Dark
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Moonblast
- Knock Off
- Close Combat
- Encore

Raging Bolt @ Leftovers
Ability: Protosynthesis
Tera Type: Electric
EVs: 252 HP / 100 SpA / 156 SpD
Modest Nature
- Thunderclap
- Draco Meteor
- Calm Mind
- Thunderbolt

Great Tusk @ Heavy-Duty Boots
Ability: Protosynthesis
Tera Type: Water
EVs: 252 HP / 4 Def / 252 Spe
Jolly Nature
- Headlong Rush
- Ice Spinner
- Rapid Spin
- Stealth Rock
""".strip()

TEAMS = {"team1": OU_TEAM, "team2": OU_TEAM_2}
