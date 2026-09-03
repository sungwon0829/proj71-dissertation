"""Analysis 6 (Amendment 24.1, POST-HOC EXPLORATORY): safety-helpfulness frontier figure.

Plots all six arms (b0, b1, b2, b3, t, t_ctrl) on:
  x = ASR primary (180 non-crisis attack prompts, lower is better)
  y = helpfulness mean reward (PsychoCounsel-Llama3-8B-Reward, higher is better)

ASR error bars are the 95% percentile bootstrap CIs ALREADY COMPUTED in
results/tables_final/tables_final.json (n_boot=10000, seed 0) -- reused verbatim,
never recomputed here. t_ctrl has no pre-computed CI in that file, so it is drawn
without an error bar (stated in the caption). Helpfulness has no CI anywhere in
the pipeline; none is drawn.

Over-refusal is annotated per point (mixed instruments: hand labels for b2/b3/t,
rubric-judge cross-check only for b0/b1/t_ctrl, kappa~0.075). Judge-only values
carry a dagger. NOT size-encoded, because the two instruments are not comparable.

Deterministic: pure matplotlib from pinned input files; no RNG (no --seed needed).
Outputs (written next to this script):
  frontier.png (300 dpi), frontier.pdf, frontier_data.json
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
TABLES_FINAL = r"C:\proj71\results\tables_final\tables_final.json"
T_CTRL_SUMMARY = r"C:\proj71\results\t_ctrl_seed42\summary_realsuite.json"

# ---------------------------------------------------------------- load inputs
with open(TABLES_FINAL, encoding="utf-8") as f:
    tf = json.load(f)
with open(T_CTRL_SUMMARY, encoding="utf-8") as f:
    tc = json.load(f)

arms = {}
for arm in ["b0", "b1", "b2", "b3", "t"]:
    row = tf["table1"][arm]
    ci = tf["per_arm_primary_asr_ci_NEW"][arm]
    assert ci["n_prompts"] == 180
    arms[arm] = {
        "asr_pct": row["asr_primary_pct"],
        "asr_ci95_pct": [ci["ci95_bootstrap_over_prompts"][0] * 100.0,
                         ci["ci95_bootstrap_over_prompts"][1] * 100.0],
        "asr_n": ci["n_prompts"],
        "helpfulness": row["helpfulness_mean_reward"],
        "over_refusal_pct": row["over_refusal_pct"],
        "over_refusal_n": 60,
        "over_refusal_source": row["over_refusal_source"],
    }
    # consistency: table1 CI text must match the reused CI block
    a, b = row["asr_ci95_over_prompts_pct"]
    assert abs(a - arms[arm]["asr_ci95_pct"][0]) < 1e-9
    assert abs(b - arms[arm]["asr_ci95_pct"][1]) < 1e-9

m = tc["metrics"]
assert m["asr_primary_non_crisis"]["n_items"] == 180
assert m["over_refusal"]["n_items"] == 60
arms["t_ctrl"] = {
    "asr_pct": m["asr_primary_non_crisis"]["value"] * 100.0,
    "asr_ci95_pct": None,  # no pre-computed bootstrap CI exists in tables_final.json
    "asr_n": 180,
    "helpfulness": m["helpfulness"]["value"],
    "over_refusal_pct": m["over_refusal"]["value"] * 100.0,
    "over_refusal_n": 60,
    "over_refusal_source": m["over_refusal"]["source"],  # rubric_judge
}

# ------------------------------------------------------------- figure styling
INK = "#222222"          # near-black for all text
GRAY_ARM = "#8a8a8a"     # neutral: context arms b0, b1
BLUE_ARM = "#4477AA"     # baseline family: b2, b3
ORANGE_ARM = "#EE7733"   # treatment family: t, t_ctrl

STYLE = {
    "b0":     dict(color=GRAY_ARM,   marker="o"),
    "b1":     dict(color=GRAY_ARM,   marker="s"),
    "b2":     dict(color=BLUE_ARM,   marker="o"),
    "b3":     dict(color=BLUE_ARM,   marker="s"),
    "t":      dict(color=ORANGE_ARM, marker="o"),
    "t_ctrl": dict(color=ORANGE_ARM, marker="s"),
}

DISPLAY = {"b0": "B0", "b1": "B1", "b2": "B2", "b3": "B3", "t": "T", "t_ctrl": "T_ctrl"}


def or_annot(arm):
    d = arms[arm]
    dag = "\u2020" if d["over_refusal_source"] != "hand_label" else ""
    return f"OR {d['over_refusal_pct']:.1f}%{dag}"


plt.rcParams.update({
    "font.size": 9.5,
    "text.color": INK,
    "axes.edgecolor": "#999999",
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
})

fig, ax = plt.subplots(figsize=(7.6, 5.4))

# recessive grid, behind everything
ax.grid(True, color="#e6e6e6", linewidth=0.6, zorder=0)
ax.set_axisbelow(True)

# ------------------------------------------------------------- draw the arms
for arm, d in arms.items():
    st = STYLE[arm]
    x, y = d["asr_pct"], d["helpfulness"]
    if d["asr_ci95_pct"] is not None:
        lo, hi = d["asr_ci95_pct"]
        ax.errorbar(x, y, xerr=[[x - lo], [hi - x]],
                    fmt="none", ecolor=st["color"], elinewidth=1.0,
                    capsize=2.5, capthick=1.0, alpha=0.85, zorder=2)
    ax.plot(x, y, marker=st["marker"], markersize=9,
            markerfacecolor=st["color"], markeredgecolor="white",
            markeredgewidth=0.8, linestyle="none", zorder=3)

# ------------------------------------------- direct labels (manual placement)
# single-line labels; offsets in points, tuned to avoid collisions
# (B2/B3 share y exactly; T/T_ctrl nearly coincide)
LEADER = dict(arrowstyle="-", color="#aaaaaa", linewidth=0.7, shrinkA=2, shrinkB=4)
LABEL = {
    #   arm: (dx, dy, ha, va, leader)
    "b0":     (0, -20, "center", "top", False),
    "b1":     (0, -16, "center", "top", False),
    "b2":     (34, 34, "left", "bottom", True),
    "b3":     (-52, 34, "right", "bottom", True),
    "t":      (-16, -20, "right", "top", True),
    "t_ctrl": (20, -20, "left", "top", True),
}
for arm, (dx, dy, ha, va, leader) in LABEL.items():
    d = arms[arm]
    name = DISPLAY[arm]
    style = "italic" if arm == "t_ctrl" else "normal"
    txt = f"{name} \u2014 {or_annot(arm)}"
    ax.annotate(txt, (d["asr_pct"], d["helpfulness"]),
                textcoords="offset points", xytext=(dx, dy),
                ha=ha, va=va, fontsize=9, color=INK, fontstyle=style,
                arrowprops=LEADER if leader else None, zorder=4)

# B3 ~ B2 note (filter fired on 2/300 suite items; helpfulness identical)
# placed in the empty lower-right region, well clear of all markers/labels
ax.text(0.975, 0.56, "B3 \u2248 B2: guardrail fired on 2/300 items;\n"
        "helpfulness identical, markers overlap",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=8, color="#555555", zorder=4)

# "better" corner arrow (upper left: low ASR, high helpfulness)
ax.annotate("better", xy=(0.035, 0.965), xycoords="axes fraction",
            xytext=(0.115, 0.865), textcoords="axes fraction",
            ha="left", va="top", fontsize=10, color="#555555",
            arrowprops=dict(arrowstyle="->", color="#555555", linewidth=1.1))

# legend for hue families (identity is direct-labelled; hues group families)
handles = [
    plt.Line2D([], [], marker="o", linestyle="none", markersize=8,
               markerfacecolor=BLUE_ARM, markeredgecolor="white", label="baseline family (B2, B3)"),
    plt.Line2D([], [], marker="o", linestyle="none", markersize=8,
               markerfacecolor=ORANGE_ARM, markeredgecolor="white", label="treatment family (T, T_ctrl)"),
    plt.Line2D([], [], marker="o", linestyle="none", markersize=8,
               markerfacecolor=GRAY_ARM, markeredgecolor="white", label="context arms (B0, B1)"),
]
leg = ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=8.5,
                handletextpad=0.4, borderaxespad=0.4, labelcolor=INK)

ax.set_xlabel("Attack success rate, primary endpoint (%)  \u2014  lower is better\n"
              "(180 non-crisis attack prompts; error bars: 95% bootstrap CI over prompts)")
ax.set_ylabel("Helpfulness (mean reward,\nPsychoCounsel-Llama3-8B-Reward)  \u2014  higher is better")
ax.set_xlim(15, 62)
ax.set_ylim(-13, 12.5)
ax.tick_params(length=3, width=0.7)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

ax.set_title("Safety\u2013helpfulness frontier, all six arms \u2014 "
             "post-hoc exploratory (Amendment 24.1)", fontsize=10.5, color=INK, pad=10)

fig.tight_layout()
fig.savefig(os.path.join(HERE, "frontier.png"), dpi=300)
fig.savefig(os.path.join(HERE, "frontier.pdf"))

# ------------------------------------------------------------ data sidecar
caption = (
    "Safety-helpfulness frontier across all six arms (post-hoc exploratory, Amendment 24.1). "
    "x: ASR on the primary endpoint (n=180 non-crisis attack prompts: prefilling, persona, many_shot; "
    "crisis_adjacent is a co-primary judged under refusal-is-failure semantics and is never pooled). "
    "y: mean reward from PsychoCounsel-Llama3-8B-Reward over the 120 benign_sensitive + crisis_adjacent items; "
    "helpfulness has no CI (single deterministic pass, no uncertainty estimate computed). "
    "All values are training seed 1 (eval seed 42), single run per arm. "
    "ASR error bars are the 95% percentile bootstrap CIs over the prompt set (n_boot=10000, seed 0) reused "
    "verbatim from results/tables_final/tables_final.json; T_ctrl (italic; Amendment 9 single-run weak control) "
    "has no pre-computed CI in that file and is shown without an error bar rather than recomputing one. "
    "Over-refusal (OR, n=60 benign_sensitive) is annotated per point rather than size-encoded because the "
    "instruments are mixed: B2/B3/T are hand labels (Revision 4 primary instrument); daggered values (B0, B1, "
    "T_ctrl) come from the rubric-judge cross-check only (kappa~0.075 vs hand labels, failed validation) and are "
    "not comparable to the hand-labelled rows. B3 is near-identical to B2: the Llama Guard filter fired on 2/300 "
    "suite items (both prefilling), so the two arms' helpfulness values are identical and their markers overlap. "
    "ASR is a conservative lower bound (judge recall 0.55, precision 1.00). Exploratory figure; confirmatory "
    "claims are confined to the pre-registered analyses."
)

sidecar = {
    "record_type": "exploratory_frontier_figure_data",
    "analysis": "Amendment 24.1 Analysis 6 (post-hoc exploratory)",
    "script": os.path.join(HERE, "plot_frontier.py"),
    "deterministic": True,
    "seed": None,
    "sources": {
        "tables_final": TABLES_FINAL,
        "t_ctrl_summary": T_CTRL_SUMMARY,
    },
    "small_cell_rule": "denominators < 10 would be reported as counts only; all denominators here are >= 60",
    "arms": arms,
    "caption": caption,
}
with open(os.path.join(HERE, "frontier_data.json"), "w", encoding="utf-8") as f:
    json.dump(sidecar, f, indent=2)

print("written:", os.path.join(HERE, "frontier.png"))
print("written:", os.path.join(HERE, "frontier.pdf"))
print("written:", os.path.join(HERE, "frontier_data.json"))
for arm, d in arms.items():
    print(f"{arm:7s} asr={d['asr_pct']:6.2f}% ci={d['asr_ci95_pct']} help={d['helpfulness']:8.3f} "
          f"or={d['over_refusal_pct']:5.2f}% ({d['over_refusal_source']})")
