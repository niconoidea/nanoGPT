"""
Generates four figures from results.json:
  1. figures/fig1_loss_grid.png       – heatmap table of final val loss
  2. figures/fig2_scaling_plots.png   – (a) loss vs params, (b) loss vs data, (c) loss vs FLOPs
  3. figures/fig3_compute_frontier.png – compute-optimal frontier
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ── data ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
data = json.loads((ROOT / "results.json").read_text())
complete = [r for r in data if r["final_val_loss"] is not None]

SIZE_ORDER  = ["XS", "S", "M"]   # L run is incomplete
SPLIT_ORDER = [10, 25, 50, 100]
SIZE_PARAMS = {"XS": 119_168, "S": 828_672, "M": 10_745_088}
SIZE_FLOPS  = {"XS": 58_573_455_360_000,
               "S":  407_308_861_440_000,
               "M":  5_281_425_653_760_000}

SIZE_COLOR  = {"XS": "#4e79a7", "S": "#f28e2b", "M": "#e15759"}
SPLIT_COLOR = {10: "#76b7b2", 25: "#59a14f", 50: "#edc948", 100: "#b07aa1"}

FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)

# Helper: look up val loss for a (size, split) pair
def val_loss(size, split):
    for r in complete:
        if r["size"] == size and r["split_pct"] == split:
            return r["final_val_loss"]
    return None


# ── fig 1 · loss grid heatmap ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 3.2))
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#1a1a2e")

grid = np.array([[val_loss(s, sp) or np.nan
                  for sp in SPLIT_ORDER]
                 for s in SIZE_ORDER], dtype=float)

im = ax.imshow(grid, cmap="RdYlGn_r", aspect="auto",
               vmin=np.nanmin(grid), vmax=min(np.nanmax(grid), 3.0))

# annotate cells
for i, size in enumerate(SIZE_ORDER):
    for j, split in enumerate(SPLIT_ORDER):
        v = grid[i, j]
        txt = f"{v:.4f}" if not np.isnan(v) else "—"
        color = "black" if 1.5 < v < 2.8 else "white"
        ax.text(j, i, txt, ha="center", va="center",
                fontsize=11, fontweight="bold", color=color)

ax.set_xticks(range(len(SPLIT_ORDER)))
ax.set_xticklabels([f"{s}%" for s in SPLIT_ORDER], color="white", fontsize=11)
ax.set_yticks(range(len(SIZE_ORDER)))
ax.set_yticklabels(SIZE_ORDER, color="white", fontsize=11)
ax.set_xlabel("Dataset split", color="white", fontsize=12)
ax.set_ylabel("Model size", color="white", fontsize=12)
ax.set_title("Final validation loss", color="white", fontsize=13, pad=10)
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_edgecolor("#444")

cb = fig.colorbar(im, ax=ax, pad=0.02)
cb.ax.yaxis.set_tick_params(color="white")
plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
cb.set_label("val loss", color="white")

plt.tight_layout()
fig.savefig(FIGS / "fig1_loss_grid.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("Saved fig1_loss_grid.png")


# ── fig 2 · three scaling plots ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.patch.set_facecolor("#1a1a2e")
for ax in axes:
    ax.set_facecolor("#0f3460")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334")

# (a) val loss vs. model params — one curve per split
ax = axes[0]
for split in SPLIT_ORDER:
    xs = [SIZE_PARAMS[s] for s in SIZE_ORDER]
    ys = [val_loss(s, split) for s in SIZE_ORDER]
    ax.plot(xs, ys, "o-", color=SPLIT_COLOR[split],
            label=f"{split}%", linewidth=2, markersize=7)
ax.set_xscale("log")
ax.set_xlabel("Parameters")
ax.set_ylabel("Final val loss")
ax.set_title("(a) Loss vs. model size")
ax.xaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x/1e6:.2g}M"))
ax.legend(title="Dataset split", title_fontsize=9,
          facecolor="#1a1a2e", labelcolor="white",
          edgecolor="#555", fontsize=9)
ax.grid(True, color="#334", linestyle="--", linewidth=0.6)

# (b) val loss vs. dataset size — one curve per model
ax = axes[1]
for size in SIZE_ORDER:
    xs = SPLIT_ORDER
    ys = [val_loss(size, sp) for sp in SPLIT_ORDER]
    ax.plot(xs, ys, "o-", color=SIZE_COLOR[size],
            label=size, linewidth=2, markersize=7)
ax.set_xlabel("Dataset split (%)")
ax.set_ylabel("Final val loss")
ax.set_title("(b) Loss vs. dataset size")
ax.set_xticks(SPLIT_ORDER)
ax.set_xticklabels([f"{s}%" for s in SPLIT_ORDER])
ax.legend(title="Model size", title_fontsize=9,
          facecolor="#1a1a2e", labelcolor="white",
          edgecolor="#555", fontsize=9)
ax.grid(True, color="#334", linestyle="--", linewidth=0.6)

# (c) val loss vs. FLOPs — all (size, split) points coloured by model size
ax = axes[2]
for size in SIZE_ORDER:
    xs, ys = [], []
    for sp in SPLIT_ORDER:
        v = val_loss(size, sp)
        if v is not None:
            xs.append(SIZE_FLOPS[size])
            ys.append(v)
    ax.scatter(xs, ys, color=SIZE_COLOR[size], label=size,
               s=80, zorder=3, edgecolors="white", linewidths=0.5)
ax.set_xscale("log")
ax.set_xlabel("Total FLOPs")
ax.set_ylabel("Final val loss")
ax.set_title("(c) Loss vs. FLOPs")
ax.xaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x:.0e}"))
ax.legend(title="Model size", title_fontsize=9,
          facecolor="#1a1a2e", labelcolor="white",
          edgecolor="#555", fontsize=9)
ax.grid(True, color="#334", linestyle="--", linewidth=0.6)

plt.suptitle("Scaling plots", color="white", fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(FIGS / "fig2_scaling_plots.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("Saved fig2_scaling_plots.png")


# ── fig 3 · compute-optimal frontier ─────────────────────────────────────────
# For each compute budget (FLOPs = model size proxy), find the (model, split)
# pair with the lowest val loss.
frontier = []
for size in SIZE_ORDER:
    candidates = [(val_loss(size, sp), sp)
                  for sp in SPLIT_ORDER
                  if val_loss(size, sp) is not None]
    if not candidates:
        continue
    best_loss, best_split = min(candidates)
    frontier.append({
        "size": size,
        "split": best_split,
        "flops": SIZE_FLOPS[size],
        "val_loss": best_loss,
    })

fig, ax = plt.subplots(figsize=(6.5, 4.5))
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#0f3460")

flops  = [p["flops"]    for p in frontier]
losses = [p["val_loss"] for p in frontier]
sizes  = [p["size"]     for p in frontier]
splits = [p["split"]    for p in frontier]
colors = [SIZE_COLOR[s] for s in sizes]

ax.plot(flops, losses, "--", color="#aaa", linewidth=1.2, zorder=1)
sc = ax.scatter(flops, losses, c=colors, s=150, zorder=3,
                edgecolors="white", linewidths=0.8)

for flop, loss, size, split in zip(flops, losses, sizes, splits):
    ax.annotate(f"{size} / {split}%",
                xy=(flop, loss), xytext=(8, 6),
                textcoords="offset points",
                color="white", fontsize=9)

ax.set_xscale("log")
ax.set_xlabel("Total FLOPs (compute budget)", color="white")
ax.set_ylabel("Best val loss (optimal split)", color="white")
ax.set_title("Compute-optimal frontier\n"
             "(lowest val loss per compute budget)",
             color="white", fontsize=12)
ax.tick_params(colors="white")
ax.xaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x:.0e}"))
ax.grid(True, color="#334", linestyle="--", linewidth=0.6)
for spine in ax.spines.values():
    spine.set_edgecolor("#334")

# custom legend patches
from matplotlib.patches import Patch
legend_handles = [Patch(facecolor=SIZE_COLOR[s], label=s,
                        edgecolor="white") for s in SIZE_ORDER]
ax.legend(handles=legend_handles, title="Model size", title_fontsize=9,
          facecolor="#1a1a2e", labelcolor="white", edgecolor="#555", fontsize=9)

plt.tight_layout()
fig.savefig(FIGS / "fig3_compute_frontier.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("Saved fig3_compute_frontier.png")
