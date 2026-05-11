"""
Per-seed dot plot: M2 vs M4 σ=4.0 for each seed.
Pairs connected by lines; seed 44 failure highlighted.
Saves: analysis/plots/fig3_per_seed.pdf  (and .png)
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS = Path("/data/lizhiwei/dfl_v2/v5/exp2/results")
OUT     = Path(__file__).parent / "plots"
OUT.mkdir(exist_ok=True)

SEEDS = [42, 43, 44]
SEED_COLORS = {42: "#4393c3", 43: "#2166ac", 44: "#d73027"}  # red = stuck seed

m2   = {s: json.load(open(RESULTS / f"M2_seed{s}_metrics.json"))["C_norm"] for s in SEEDS}
m4   = {s: json.load(open(RESULTS / f"M4_sigma4.0_seed{s}_metrics.json"))["C_norm"] for s in SEEDS}

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 4.5))

x_m2, x_m4 = 0, 1

for s in SEEDS:
    color = SEED_COLORS[s]
    improved = m4[s] < m2[s]
    ls = "-" if improved else "--"
    lw = 1.8

    ax.plot([x_m2, x_m4], [m2[s], m4[s]], ls=ls, color=color, lw=lw, zorder=3)
    ax.scatter([x_m2], [m2[s]], color=color, s=70, zorder=5)
    ax.scatter([x_m4], [m4[s]], color=color, s=70, zorder=5,
               marker="D" if not improved else "o")

    # Seed label on M2 side
    ax.text(x_m2 - 0.06, m2[s], f"seed {s}", ha="right", va="center",
            fontsize=9, color=color, fontweight="bold")

    # Delta label
    delta = m4[s] - m2[s]
    mid_y = (m2[s] + m4[s]) / 2
    sign  = "+" if delta >= 0 else ""
    ax.text(0.5, mid_y + 0.001, f"{sign}{delta:.4f}",
            ha="center", va="bottom", fontsize=8, color=color)

# Formatting
ax.set_xlim(-0.4, 1.4)
ax.set_xticks([x_m2, x_m4])
ax.set_xticklabels(["M2\n(CE)", "M4 σ=4.0\n(DFL)"], fontsize=11)
ax.set_ylabel(r"$C_{\mathrm{norm}}$ (↓ better)", fontsize=11)
ax.set_title("Per-Seed: M2 vs M4 σ=4.0\n(dashed line + ◆ = Stage 3 did not improve)", fontsize=11)
ax.yaxis.grid(True, lw=0.5, alpha=0.5, zorder=0)
ax.set_axisbelow(True)

# Legend for stuck seed
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color="#4393c3", lw=1.8, label="seed 42 (improved)"),
    Line2D([0], [0], color="#2166ac", lw=1.8, label="seed 43 (improved)"),
    Line2D([0], [0], color="#d73027", lw=1.8, ls="--", label="seed 44 (Stage 3 stuck)"),
]
ax.legend(handles=legend_elements, fontsize=9, loc="upper right")

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig3_per_seed.{ext}", dpi=150, bbox_inches="tight")
print(f"Saved → {OUT}/fig3_per_seed.{{pdf,png}}")
