"""
For every (model size × dataset split) run, parse the evaluation checkpoints
from the log files and plot train loss vs. validation loss over training steps.
Saves figures/fig_train_vs_val.png
"""

import re
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

LOG_DIR  = Path(__file__).parent / "logs"
FIGS     = Path(__file__).parent / "figures"
FIGS.mkdir(exist_ok=True)

SIZE_ORDER  = ["XS", "S", "M", "L"]
SPLIT_ORDER = [10, 25, 50, 100]

SIZE_COLOR  = {"XS": "#4e79a7", "S": "#f28e2b", "M": "#e15759", "L": "#76b7b2"}

RE_STEP = re.compile(
    r"step\s+(\d+):\s+train loss\s+([\d.]+),\s+val loss\s+([\d.]+)"
)


def parse_curves(path: Path):
    """Return (steps, train_losses, val_losses) lists from a log file."""
    steps, train_losses, val_losses = [], [], []
    for m in RE_STEP.finditer(path.read_text(encoding="utf-8")):
        steps.append(int(m.group(1)))
        train_losses.append(float(m.group(2)))
        val_losses.append(float(m.group(3)))
    return steps, train_losses, val_losses


# ── build subplot grid: rows = model size, cols = dataset split ──────────────
n_rows = len(SIZE_ORDER)
n_cols = len(SPLIT_ORDER)

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(4.5 * n_cols, 3.5 * n_rows),
    sharex=False, sharey=False,
)
fig.patch.set_facecolor("#1a1a2e")

BG   = "#0f3460"
GRID = "#1e3a5f"

for row, size in enumerate(SIZE_ORDER):
    for col, split in enumerate(SPLIT_ORDER):
        ax = axes[row][col]
        ax.set_facecolor(BG)
        for spine in ax.spines.values():
            spine.set_edgecolor("#334")
        ax.tick_params(colors="white", labelsize=8)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.grid(True, color=GRID, linestyle="--", linewidth=0.6)

        log_path = LOG_DIR / f"train_{size}_split{split}.log"
        if not log_path.exists():
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="#888", fontsize=10)
            ax.set_title(f"{size} / {split}%", color="white", fontsize=9, pad=4)
            continue

        steps, train_l, val_l = parse_curves(log_path)
        if not steps:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="#888", fontsize=10)
            ax.set_title(f"{size} / {split}%", color="white", fontsize=9, pad=4)
            continue

        color = SIZE_COLOR[size]
        ax.plot(steps, train_l, color=color,   linewidth=1.8,
                label="train", zorder=3)
        ax.plot(steps, val_l,   color="white", linewidth=1.8,
                linestyle="--", label="val", zorder=3, alpha=0.85)

        ax.set_title(f"{size} / {split}%", color="white", fontsize=9, pad=4)
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x)))
        )

        # column label (split) on top row only
        if row == 0:
            ax.set_xlabel("")   # title already shows split
        # row label (size) on first column only
        if col == 0:
            ax.set_ylabel("loss", color="white", fontsize=8)
        if col > 0:
            ax.set_ylabel("")

        # shared legend only in first subplot
        if row == 0 and col == 0:
            leg = ax.legend(fontsize=8, facecolor="#1a1a2e",
                            labelcolor="white", edgecolor="#555",
                            loc="upper right")

# row labels on the left outside the axes
for row, size in enumerate(SIZE_ORDER):
    axes[row][0].annotate(
        size,
        xy=(-0.22, 0.5), xycoords="axes fraction",
        ha="right", va="center",
        color=SIZE_COLOR[size], fontsize=13, fontweight="bold",
    )

# column headers on top
for col, split in enumerate(SPLIT_ORDER):
    axes[0][col].set_title(
        f"{split}% data  ·  {SIZE_ORDER[0]}",
        color="white", fontsize=9, pad=4,
    )
# re-do all titles cleanly
for row, size in enumerate(SIZE_ORDER):
    for col, split in enumerate(SPLIT_ORDER):
        axes[row][col].set_title(f"{split}% data", color="white",
                                 fontsize=9, pad=4)

fig.suptitle("Train loss  vs.  Validation loss", color="white",
             fontsize=15, y=1.01)

# x-axis label only on bottom row
for col in range(n_cols):
    axes[-1][col].set_xlabel("step", color="white", fontsize=8)

plt.tight_layout()
out = FIGS / "fig_train_vs_val.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved {out}")
