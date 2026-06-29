"""Generate paper figures from results/*.json. Run from repo root."""
import json, os
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"figure.dpi": 150, "font.size": 11,
                     "axes.grid": True, "grid.alpha": 0.3})

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "..", "figures")
os.makedirs(OUT, exist_ok=True)

ou = json.load(open(os.path.join(RESULTS, "ou_final_result.json")))
final = json.load(open(os.path.join(RESULTS, "FINAL_results.json")))

# Fig 1: OU GRPO training - rise then collapse
wc = np.array(ou["ou_train_curve_pct"])
steps = np.arange(1, len(wc) + 1)
gn = [None,418,830,248,510,585,379,480,270,488,453,1077,1526,1055,1589,1357,264,362]
gn = np.array([g if g is not None else np.nan for g in gn], dtype=float)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)
ax1.plot(steps, wc, marker="o", ms=3, lw=1, color="#2a6f97", label="win rate")
k = 3
roll = np.convolve(wc, np.ones(k)/k, mode="valid")
ax1.plot(steps[k-1:], roll, lw=2.2, color="#e36414", label=f"{k}-step moving avg")
ax1.axhline(50, ls="--", lw=1, color="gray", alpha=0.7)
ax1.axvspan(12, len(wc), color="red", alpha=0.07)
ax1.annotate("peak 88%", xy=(8, 88), xytext=(8, 70), ha="center", fontsize=8,
             arrowprops=dict(arrowstyle="->", lw=0.8))
ax1.set_ylabel("win rate vs\nMaxBasePower (%)"); ax1.set_ylim(0, 100)
ax1.set_title("GRPO on matched Gen 9 OU: rise to a peak, then collapse")
ax1.legend(loc="upper right", fontsize=8)
ax2.plot(steps[:len(gn)], gn, marker="o", ms=3, lw=1.2, color="#9d0208")
ax2.axvspan(12, len(wc), color="red", alpha=0.07)
ax2.axhline(1000, ls=":", lw=1, color="gray")
ax2.set_yscale("log"); ax2.set_ylabel("grad norm (log)")
ax2.set_xlabel("GRPO step")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig1_ou_training.png")); plt.close()

# Fig 2: SFT action bias
v = final["sft_static_eval"]["vanilla"]; b = final["sft_static_eval"]["balanced"]
models = ["Vanilla SFT", "Balanced SFT"]
switch_em = [v["switch_EM"]*100, b["switch_EM"]*100]
move_em   = [v["move_EM"]*100,   b["move_EM"]*100]
x = np.arange(2); w = 0.35
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(x-w/2, switch_em, w, label="switch-EM", color="#2a9d8f")
ax.bar(x+w/2, move_em,   w, label="move-EM",   color="#e76f51")
ax.set_xticks(x); ax.set_xticklabels(models)
ax.set_ylabel("exact-match accuracy (%)"); ax.set_ylim(0, 100)
ax.set_title("SFT swaps one action bias for another")
for i, (s, m) in enumerate(zip(switch_em, move_em)):
    ax.text(i-w/2, s+1.5, f"{s:.0f}", ha="center", fontsize=9)
    ax.text(i+w/2, m+1.5, f"{m:.0f}", ha="center", fontsize=9)
ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig2_sft_bias.png")); plt.close()

# Fig 3: generalization
g = ou["ou_generalization_test"]
teams = ["team1\n(trained on)", "team2\n(unseen)"]
grpo = [g["team1_trained"]["step5"], g["team2_unseen"]["step5"]]
sft  = [g["team1_trained"]["sft"],   g["team2_unseen"]["sft"]]
x = np.arange(2); w = 0.35
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(x-w/2, grpo, w, label="GRPO (step 5)", color="#2a6f97")
ax.bar(x+w/2, sft,  w, label="SFT (start)",   color="#bc6c25")
ax.axhline(50, ls="--", lw=1, color="gray", alpha=0.7)
ax.set_xticks(x); ax.set_xticklabels(teams)
ax.set_ylabel("head-to-head wins / 100"); ax.set_ylim(0, 100)
ax.set_title("GRPO gain is matchup-specific")
for i, (gg, ss) in enumerate(zip(grpo, sft)):
    ax.text(i-w/2, gg+1.5, f"{gg}", ha="center", fontsize=9)
    ax.text(i+w/2, ss+1.5, f"{ss}", ha="center", fontsize=9)
ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig3_generalization.png")); plt.close()
print("wrote fig1_ou_training.png, fig2_sft_bias.png, fig3_generalization.png")
