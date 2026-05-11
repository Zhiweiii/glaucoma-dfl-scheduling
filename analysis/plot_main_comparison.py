"""
Hero figure: C_norm bar chart for M1, M2, M3_fix, M4 σ=4.0 across seeds 42/43/44.
Saves: analysis/plots/fig1_main_comparison.pdf  (and .png)
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

RESULTS = Path("/data/lizhiwei/dfl_v2/v5/exp2/results")
OUT     = Path(__file__).parent / "plots"
OUT.mkdir(exist_ok=True)

SEEDS = [42, 43, 44]

METHODS = {
    "M1":         ("M1_seed{s}",          "M1\n(binary only)"),
    "M2":         ("M2_seed{s}",          "M2\n(CE)"),
    "M3":         ("M3_seed{s}",          "M3\n(cost ckpt)"),
    "M4 σ=4.0":   ("M4_sigma4.0_seed{s}", "M4\n(DFL, σ=4)"),
}

C_ORACLE = 15742.5
C_RANDOM = 20180.422

# ── Load data ─────────────────────────────────────────────────────────────────
means, stds, all_vals = {}, {}, {}
for key, (pattern, _) in METHODS.items():
    vals = [json.load(open(RESULTS / f"{pattern.format(s=s)}_metrics.json"))["C_norm"]
            for s in SEEDS]
    all_vals[key] = vals
    means[key]    = np.mean(vals)
    stds[key]     = np.std(vals)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))

labels   = list(METHODS.keys())
x        = np.arange(len(labels))
width    = 0.55
colors   = ["#9ecae1", "#4393c3", "#2166ac", "#d6604d"]

bars = ax.bar(x, [means[k] for k in labels], width,
              yerr=[stds[k] for k in labels],
              color=colors, capsize=5, error_kw=dict(lw=1.5, capthick=1.5),
              zorder=3)

# Individual seed dots
for i, key in enumerate(labels):
    ax.scatter([x[i]] * len(SEEDS), all_vals[key],
               color="black", s=20, zorder=5, alpha=0.7)

# Oracle and random reference lines
ax.axhline(C_ORACLE / C_RANDOM, color="#1a9641", lw=1.5, ls="--", zorder=2,
           label=f"Oracle  ({C_ORACLE/C_RANDOM:.3f})")
ax.axhline(1.0, color="#d73027", lw=1.5, ls=":", zorder=2, label="Random  (1.000)")

# Formatting
ax.set_xticks(x)
ax.set_xticklabels([METHODS[k][1] for k in labels], fontsize=11)
ax.set_ylabel(r"$C_{\mathrm{norm}}$ = $C_{\mathrm{total}}$ / $C_{\mathrm{random}}$  (↓ better)", fontsize=11)
ax.set_title("Scheduling Performance — Main Method Comparison\n(test set, N=333, seeds 42/43/44)",
             fontsize=12)
ax.set_ylim(0.76, 1.02)
ax.yaxis.grid(True, lw=0.5, alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.legend(fontsize=10, loc="upper right")

# Value labels on bars
for bar, key in zip(bars, labels):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + stds[key] + 0.003,
            f"{means[key]:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig1_main_comparison.{ext}", dpi=150, bbox_inches="tight")
print(f"Saved → {OUT}/fig1_main_comparison.{{pdf,png}}")
