# plot_eval.py
# Plot the in-training learning curve. Defaults to eval_history.json (the cp run);
# pass another file for other runs, e.g.:
#   python plot_eval.py eval_history_wdl.json   ->  eval_curve_wdl.png
import sys
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "eval_history.json"
out = src.replace("eval_history", "eval_curve").replace(".json", ".png")

hist = json.load(open(src))
steps = [h["step"] for h in hist]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
panels = [
    (axes[0], "top1",       "top-1 match (higher = better)",        True),
    (axes[1], "legal_rate", "legal-move rate (higher = better)",    True),
    (axes[2], "cp_loss",    "avg centipawn loss (lower = better)",  False),
]
for ax, key, title, is_pct in panels:
    ax.plot(steps, [h[key] for h in hist], "-o")
    ax.set(xlabel="step", title=title)
    if is_pct:
        ax.set_ylim(0, 1)

fig.suptitle(f"Qwen3 chess GRPO — held-out eval ({src})")
fig.tight_layout()
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"wrote {out}")
