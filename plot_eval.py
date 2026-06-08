# plot_eval.py
# Plots the in-training learning curve from eval_history.json -> eval_curve.png
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

hist = json.load(open("eval_history.json"))
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

fig.suptitle("Qwen3-1.7B chess GRPO — held-out eval")
fig.tight_layout()
fig.savefig("eval_curve.png", dpi=120, bbox_inches="tight")
print("wrote eval_curve.png")
