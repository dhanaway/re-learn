# Chess RL — Qwen3 + GRPO + Stockfish

Reinforcement-learning a Qwen3 model to play chess. Stockfish supplies the reward
(centipawn loss); GRPO reinforces moves that beat the group average. Prove the
method at 1.7B, then scale to 14B.

## Files
- `chess_reward.py`  — parses the model's move, scores it with Stockfish (the training reward). `DEPTH=6`.
- `chess_eval.py`    — shared eval: build a prompt, score a move vs Stockfish (`EVAL_DEPTH=8`). Used by training + benchmark so they match.
- `make_dataset.py`  — generates the TRAIN set (`chess_positions/`) and a fixed VAL set (`val_fens.json`).
- `train.py`         — proper 1.7B GRPO run: 400 steps, constant LR, eval on the fixed val set every 25 steps -> `eval_history.json`. vLLM on (TRITON_ATTN backend).
- `benchmark.py`     — before/after benchmark on the fixed val set (greedy, deterministic).
- `plot_eval.py`     — plots the learning curve from `eval_history.json` -> `eval_curve.png`.
- `train_smoke.py`   — the original 30-step smoke test (kept for reference).

## One-time environment setup (vLLM fix)
vLLM 0.11.0's default FlashInfer backend asserts (`_sm_scale`) when unsloth enables LoRA,
so remove it once per pod:
```bash
pip uninstall -y flashinfer-python flashinfer
```
`train.py` and `benchmark.py` set `VLLM_ATTENTION_BACKEND=TRITON_ATTN` at the top, so vLLM
uses the Triton backend (bundled, no extra install) instead.

## Run order (on the H100 pod)
```bash
python make_dataset.py                          # train set + fixed val_fens.json
python benchmark.py Qwen/Qwen3-1.7B             # baseline on the fixed val set
python train.py                                 # ~30-60 min; prints [eval @ step N] every 25 steps
python plot_eval.py                             # -> eval_curve.png
python benchmark.py ./qwen3-1.7b-chess-merged   # trained, directly comparable to baseline
runpodctl send eval_curve.png                   # pull the curve to your Mac
```

## What success looks like
On the held-out curve (`eval_curve.png`) and the before/after benchmark, all measured
on the SAME fixed positions every time:
- **top-1 match** rises, **avg centipawn loss** falls, **legal-move rate** trends toward ~100%.

## Scaling to 14B (after the 1.7B run proves the method)
1. In `train.py`: apply the `[14B]` lines (`Qwen/Qwen3-14B`, `load_in_4bit=True`). vLLM is already on; the same flashinfer-uninstall + TRITON_ATTN fix applies. Lower `gpu_memory_utilization` if you hit OOM.
2. In `make_dataset.py`: remove `/no_think`, bump `build_train(n=...)`.
3. In `chess_reward.py`: set `DEPTH=10`.
