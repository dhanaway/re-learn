# make_dataset.py
# Generates a TRAIN set (for GRPO) and a fixed VAL set (for benchmarking + the
# in-training eval curve). The val set is saved to val_fens.json so every
# evaluation runs on the exact same positions -> trustworthy before/after numbers.
import json
import random
import chess
from datasets import Dataset

# "/no_think" makes Qwen3 answer directly (no long reasoning) = fast.
# For the full 14B run, REMOVE "/no_think" so the model is allowed to reason.
SYSTEM = ("You are a strong chess engine. Give your single best move in SAN "
          "notation inside \\boxed{}. Example: \\boxed{Nf3} /no_think")


def random_position(max_plies=40):
    board = chess.Board()
    for _ in range(random.randint(6, max_plies)):
        if board.is_game_over():
            break
        board.push(random.choice(list(board.legal_moves)))
    if board.is_game_over():
        return random_position(max_plies)
    return board


def build_train(n=2000, seed=0):
    random.seed(seed)
    rows = []
    for _ in range(n):
        b = random_position()
        legal = [b.san(m) for m in b.legal_moves]
        user = (f"FEN: {b.fen()}\n"
                f"Side to move: {'White' if b.turn else 'Black'}\n"
                f"Legal moves: {', '.join(legal)}\n"
                "Choose the best move.")
        rows.append({
            "prompt": [{"role": "system", "content": SYSTEM},
                       {"role": "user",   "content": user}],
            "fen": b.fen(),
        })
    return Dataset.from_list(rows)


def build_val(n=24, seed=999):
    random.seed(seed)                         # different seed -> no overlap with train
    return [random_position().fen() for _ in range(n)]


if __name__ == "__main__":
    train = build_train()
    train.save_to_disk("chess_positions")
    val = build_val()
    json.dump(val, open("val_fens.json", "w"), indent=2)
    print(f"saved {len(train)} train positions to ./chess_positions")
    print(f"saved {len(val)} val positions to ./val_fens.json")
