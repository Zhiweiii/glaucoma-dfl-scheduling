"""
Per-seed dot plot: M2 → M3 → M4 σ=4.0 for each seed.
Pairs connected by lines; seed 44 DFL failure highlighted.
Saves: analysis/plots/fig3_per_seed.pdf  (and .png)
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

RESULTS = Path("/data/lizhiwei/dfl_v2/v5/exp2/results")
OUT     = Path(__file__).parent / "plots"
OUT.mkdir(exist_ok=True)

SEEDS = [42, 43, 44]
SEED_COLORS = {42: "#4393c3", 43: "#2166ac", 44: "#d73027"}  # red = stuck seed

m2 = {s: json.load(open(RESULTS / f"M2_seed{s}_metrics.json"))["C_norm"] for s in SEEDS}
m3 = {s: json.load(open(RESULTS / f"M3_seed{s}_metrics.json"))["C_norm"] for s in SEEDS}
m4 = {s: json.load(open(RESULTS / f"M4_sigma4.0_seed{s}_metrics.json"))["C_norm"] for s in SEEDS}

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 4.5))

xs = [0, 1, 2]  # M2, M3, M4

for s in SEEDS:
    color   = SEED_COLORS[s]
    vals    = [m2[s], m3[s], m4[s]]
    m4_improved = m4[s] < m2[s]

    # M2→M3 segment (always solid — just CE with different checkpoint)
    ax.plot(xs[:2], vals[:2], "-", color=color, lw=1.8, zorder=3)
    # M3→M4 segment (dashed if seed44 Stage 3 stuck)
    ax.plot(xs[1:], vals[1:], "-" if m4_improved else "--", color=color, lw=1.8, zorder=3)

    # Dots: circle for all M2/M3; diamond on M4 if stuck
    ax.scatter([xs[0], xs[1]], [vals[0], vals[1]], color=color, s=60, zorder=5)
    ax.scatter([xs[2]], [vals[2]], color=color, s=60, zorder=5,
               marker="D" if not m4_improved else "o")

    # Seed label on M2 side
    ax.text(xs[0] - 0.07, vals[0], f"seed {s}", ha="right", va="center",
            fontsize=9, color=color, fontweight="bold")

    # Delta labels
    for seg_x, (y_a, y_b) in zip([0.5, 1.5], [(vals[0], vals[1]), (vals[1], vals[2])]):
        delta = y_b - y_a
        sign  = "+" if delta >= 0 else ""
        ax.text(seg_x, (y_a + y_b) / 2 + 0.0005, f"{sign}{delta:.4f}",
                ha="center", va="bottom", fontsize=7.5, color=color)

# Formatting
ax.set_xlim(-0.45, 2.45)
ax.set_xticks(xs)
ax.set_xticklabels(["M2\n(CE)", "M3\n(cost ckpt)", "M4\n(DFL, σ=4)"], fontsize=11)
ax.set_ylabel(r"$C_{\mathrm{norm}}$ (↓ better)", fontsize=11)
ax.set_title("Per-Seed: M2 → M3 → M4 σ=4.0\n(dashed + ◆ = Stage 3 did not improve)", fontsize=11)
ax.yaxis.grid(True, lw=0.5, alpha=0.5, zorder=0)
ax.set_axisbelow(True)

legend_elements = [
    Line2D([0], [0], color="#4393c3", lw=1.8, label="seed 42"),
    Line2D([0], [0], color="#2166ac", lw=1.8, label="seed 43"),
    Line2D([0], [0], color="#d73027", lw=1.8, ls="--", label="seed 44 (Stage 3 stuck)"),
]
ax.legend(handles=legend_elements, fontsize=7.5, loc="center right",
          handlelength=1.2, handletextpad=0.4, borderpad=0.4, labelspacing=0.3)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig3_per_seed.{ext}", dpi=150, bbox_inches="tight")
print(f"Saved → {OUT}/fig3_per_seed.{{pdf,png}}")
