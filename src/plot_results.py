import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ── Data ──────────────────────────────────────────────────────────────────────
models   = ['M1', 'M2b', 'M3']
c_norm   = np.array([0.9599, 0.8877, 0.8882])
c_regret = np.array([3209.0, 1766.0, 1775.0])
recall_k = np.array([0.3777, 0.4734, 0.4628])
pw_acc   = np.array([0.5614, 0.6934, 0.6971])

BLUE   = '#3B82F6'
ORANGE = '#F97316'
out_dir = Path.home() / 'repos'
out_dir.mkdir(parents=True, exist_ok=True)

x     = np.arange(len(models))
width = 0.35

# ── Chart 1: Decision Quality ─────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()

bars1 = ax1.bar(x - width / 2, c_norm,   width, color=BLUE,   alpha=0.85, label='C_norm ↓',   zorder=3)
bars2 = ax2.bar(x + width / 2, c_regret, width, color=ORANGE, alpha=0.85, label='C_regret ↓', zorder=3)

# Annotations: value above bar, [% vs M1] inside bar
pct_cn = [None, -7.5, -7.5]
pct_cr = [None, -45.0, -44.7]

for i, (v, pct) in enumerate(zip(c_norm, pct_cn)):
    ax1.text(x[i] - width / 2, v + 0.003, f'{v:.3f}',
             ha='center', va='bottom', fontsize=9, fontweight='bold', color='black', zorder=4)
    if pct is not None:
        ax1.text(x[i] - width / 2, v - 0.025, f'[{pct:.1f}%]',
                 ha='center', va='top', fontsize=8, color='white', fontweight='bold', zorder=4)

for i, (v, pct) in enumerate(zip(c_regret, pct_cr)):
    ax2.text(x[i] + width / 2, v + 40, f'{v:,.0f}',
             ha='center', va='bottom', fontsize=9, fontweight='bold', color='black', zorder=4)
    if pct is not None:
        ax2.text(x[i] + width / 2, v - 270, f'[{pct:.1f}%]',
                 ha='center', va='top', fontsize=8, color='white', fontweight='bold', zorder=4)

# Reference lines — no label (annotated inline instead)
ax1.axhline(1.000, color='#1A56DB', linestyle=':',  alpha=0.85, linewidth=2.0, zorder=1)
ax2.axhline(4012,  color='#B45309', linestyle=':',  alpha=0.85, linewidth=2.0, zorder=1)

ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=12)
ax1.set_ylabel('C_norm', color=BLUE, fontsize=11)
ax2.set_ylabel('C_regret', color=ORANGE, fontsize=11)
ax1.tick_params(axis='y', labelcolor=BLUE)
ax2.tick_params(axis='y', labelcolor=ORANGE)
ax1.set_ylim(0.82, 1.06)
ax2.set_ylim(0, 4800)

# Inline labels on reference lines (left edge, inside axes)
ax1.text(-0.48, 1.000 + 0.003, 'Random C_norm', color='#1A56DB', fontsize=7.5, va='bottom', fontstyle='italic')
ax2.text(-0.48, 4012  + 60,    'Random C_regret', color='#B45309', fontsize=7.5, va='bottom', fontstyle='italic')

ax1.set_title('Chart 1 — Decision Quality  (seed 42, p_available=0.7)', fontsize=12, fontweight='bold', pad=10)
ax1.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)

# Small legend inside — bars only
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
leg = ax2.legend(lines1 + lines2, labels1 + labels2,
                 loc='upper right', fontsize=9, framealpha=1.0)

fig.tight_layout()
fig.savefig(out_dir / 'chart1_decision_quality.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved chart1_decision_quality.png')

# ── Chart 2: Severity Discrimination ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(x - width / 2, recall_k, width, color=BLUE,   alpha=0.85, label='Recall@K ↑',      zorder=3)
ax.bar(x + width / 2, pw_acc,   width, color=ORANGE, alpha=0.85, label='Pairwise Acc ↑',  zorder=3)

pct_rk = [None, +25.3, +22.5]
pct_pa = [None, +23.5, +24.2]

for i, (v, pct) in enumerate(zip(recall_k, pct_rk)):
    ax.text(x[i] - width / 2, v + 0.005, f'{v:.3f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold', color='black', zorder=4)
    if pct is not None:
        ax.text(x[i] - width / 2, v - 0.04, f'[+{pct:.1f}%]',
                ha='center', va='top', fontsize=8, color='white', fontweight='bold', zorder=4)

for i, (v, pct) in enumerate(zip(pw_acc, pct_pa)):
    ax.text(x[i] + width / 2, v + 0.005, f'{v:.3f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold', color='black', zorder=4)
    if pct is not None:
        ax.text(x[i] + width / 2, v - 0.04, f'[+{pct:.1f}%]',
                ha='center', va='top', fontsize=8, color='white', fontweight='bold', zorder=4)

ax.axhline(0.345, color='#1A56DB', linestyle=':', alpha=0.85, linewidth=2.0, label='Recall@K random (~0.345)')
ax.axhline(0.500, color='#B45309', linestyle=':', alpha=0.85, linewidth=2.0, label='Pairwise Acc random (0.500)')

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)
ax.set_ylabel('Score', fontsize=11)
ax.set_ylim(0, 0.85)
ax.legend(fontsize=9, framealpha=0.85)
ax.set_title('Chart 2 — Severity Discrimination  (seed 42, p_available=0.7)', fontsize=12, fontweight='bold', pad=10)
ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)

fig.tight_layout()
fig.savefig(out_dir / 'chart2_severity_discrimination.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved chart2_severity_discrimination.png')
