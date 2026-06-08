#!/usr/bin/env python
# uci_engine.py
# Wraps the model as a standard UCI engine so the whole computer-chess toolchain
# (python-chess, cutechess/fastchess, ordo, lichess-bot) can drive it unchanged.
#
#   python -u uci_engine.py ./qwen3-1.7b-chess-merged    # play with a trained model
#   python -u uci_engine.py Qwen/Qwen3-1.7B              # play with the base model
#   python -u uci_engine.py random                       # a free random-move opponent (no GPU)
#
# Speak UCI on stdin/stdout. vLLM's chatty init is redirected to stderr so stdout
# stays a clean UCI channel. GPU fraction via UCI_GPU_MEM (default 0.35 so two
# model engines fit on one card for head-to-head matches).
import os
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")   # flashinfer removed
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

import sys
import contextlib
import random
import chess

from make_dataset import SYSTEM
from chess_eval import prompt_for
from chess_reward import _extract_move

MODEL  = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-1.7B"
RANDOM = (MODEL == "random")
GMU    = float(os.environ.get("UCI_GPU_MEM", "0.35"))

_llm = _tok = _sp = None


def load():
    global _llm, _tok, _sp
    if RANDOM or _llm is not None:
        return
    from vllm import LLM, SamplingParams
    kwargs = dict(model=MODEL, max_model_len=2048,
                  gpu_memory_utilization=GMU, disable_log_stats=True)
    tok_override = os.environ.get("UCI_TOKENIZER")        # fallback if a local tokenizer is broken
    if tok_override:
        kwargs["tokenizer"] = tok_override
    with contextlib.redirect_stdout(sys.stderr):          # keep stdout clean for UCI
        _llm = LLM(**kwargs)
    _tok = _llm.get_tokenizer()
    _sp  = SamplingParams(temperature=float(os.environ.get("UCI_TEMP", "0.7")),
                          max_tokens=64)


def best_move(board):
    if RANDOM:
        return random.choice(list(board.legal_moves))
    prompt = prompt_for(_tok, board.fen(), SYSTEM)
    with contextlib.redirect_stdout(sys.stderr):
        text = _llm.generate([prompt], _sp)[0].outputs[0].text
    s = _extract_move(text)
    if s:
        for parse in (board.parse_san, board.parse_uci):
            try:
                mv = parse(s)
                if mv in board.legal_moves:
                    return mv
            except Exception:
                pass
    return next(iter(board.legal_moves))                  # deterministic legal fallback


def set_position(tokens):
    board = chess.Board()
    if tokens and tokens[0] == "fen":
        board.set_fen(" ".join(tokens[1:7]))
    moves_idx = tokens.index("moves") + 1 if "moves" in tokens else len(tokens)
    for mv in tokens[moves_idx:]:
        board.push_uci(mv)
    return board


def main():
    board = chess.Board()
    for line in sys.stdin:
        line = line.strip()
        if line == "uci":
            print(f"id name QwenChess({MODEL})")
            print("id author re-learn")
            print("uciok", flush=True)
        elif line == "isready":
            load()
            print("readyok", flush=True)
        elif line == "ucinewgame":
            board = chess.Board()
        elif line.startswith("position"):
            board = set_position(line.split()[1:])
        elif line.startswith("go"):
            print(f"bestmove {board.uci(best_move(board))}", flush=True)
        elif line == "quit":
            break
        # ignore setoption / debug / anything else


if __name__ == "__main__":
    main()
