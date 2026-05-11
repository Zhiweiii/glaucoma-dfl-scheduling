"""
Sigma sweep line plot: mean C_norm vs σ with ±std shaded band.
Horizontal dashed line for M2 baseline. Highlights sweet spot at σ=4.0.
Saves: analysis/plots/fig2_sigma_sweep.pdf  (and .png)
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS = Path("/data/lizhiwei/dfl_v2/v5/exp2/results")
OUT     = Path(__file__).parent / "plots"
OUT.mkdir(exist_ok=True)

SEEDS  = [42, 43, 44]
SIGMAS = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0]

def stem(sigma):
    return "M4_seed{s}" if sigma == 0.5 else f"M4_sigma{sigma}_seed{{s}}"

# ── Load sigma sweep data ─────────────────────────────────────────────────────
means, stds = [], []
for sigma in SIGMAS:
    vals = [json.load(open(RESULTS / f"{stem(sigma).format(s=s)}_metrics.json"))["C_norm"]
            for s in SEEDS]
    means.append(np.mean(vals))
    stds.append(np.std(vals))

means = np.array(means)
stds  = np.array(stds)

# ── M2 baseline ───────────────────────────────────────────────────────────────
m2_vals = [json.load(open(RESULTS / f"M2_seed{s}_metrics.json"))["C_norm"] for s in SEEDS]
m2_mean = np.mean(m2_vals)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(SIGMAS, means, "o-", color="#d6604d", lw=2, ms=7, zorder=4, label="M4 (DFL)")
ax.fill_between(SIGMAS, means - stds, means + stds,
                color="#d6604d", alpha=0.18, zorder=3)

# M2 baseline
ax.axhline(m2_mean, color="#4393c3", lw=1.8, ls="--", zorder=2,
           label=f"M2 baseline  ({m2_mean:.4f})")

# Sweet spot annotation
best_idx = int(np.argmin(means))
ax.annotate(f"σ=4.0\n({means[best_idx]:.4f})",
            xy=(SIGMAS[best_idx], means[best_idx]),
            xytext=(SIGMAS[best_idx] + 0.5, means[best_idx] - 0.006),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
            fontsize=9, ha="left")

ax.set_xlabel("σ (DFL perturbation std)", fontsize=11)
ax.set_ylabel(r"$C_{\mathrm{norm}}$ (↓ better)", fontsize=11)
ax.set_title("M4 DFL Performance vs. Perturbation Scale σ\n(mean ± std across seeds 42/43/44)",
             fontsize=12)
ax.set_xticks(SIGMAS)
ax.yaxis.grid(True, lw=0.5, alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.legend(fontsize=10)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig2_sigma_sweep.{ext}", dpi=150, bbox_inches="tight")
print(f"Saved → {OUT}/fig2_sigma_sweep.{{pdf,png}}")
