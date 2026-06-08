# chess_eval.py
# Shared evaluation logic: build a prompt for a position, and score a model's raw
# text output against Stockfish. Used by BOTH the in-training eval callback and
# benchmark.py, so the during-training curve and the final benchmark are consistent.
import json
import chess
import chess.engine
from chess_reward import _extract_move, SF_PATH

EVAL_DEPTH = 8     # reference engine depth for "best move" (deeper than the reward's DEPTH)


def load_val_fens(path="val_fens.json"):
    with open(path) as f:
        return json.load(f)


def make_eval_engine():
    return chess.engine.SimpleEngine.popen_uci(SF_PATH)


def prompt_for(tokenizer, fen, system):
    board = chess.Board(fen)
    legal = [board.san(m) for m in board.legal_moves]
    user = (f"FEN: {fen}\nSide to move: {'White' if board.turn else 'Black'}\n"
            f"Legal moves: {', '.join(legal)}\nChoose the best move.")
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True)


def score_completion(fen, text, eng, depth=EVAL_DEPTH):
    """Return dict(legal, top1, cp_loss) for the model's output `text` on `fen`.
    Illegal/unparseable -> deterministic fallback (first legal move) so eval is
    fully reproducible: same positions + greedy decoding + fixed depth = no noise."""
    board = chess.Board(fen)
    info = eng.analyse(board, chess.engine.Limit(depth=depth))
    best_cp = info["score"].pov(board.turn).score(mate_score=10000)
    best_mv = info["pv"][0] if info.get("pv") else None
    turn = board.turn

    mv_str = _extract_move(text)
    mv, was_legal = None, False
    if mv_str:
        for parse in (board.parse_san, board.parse_uci):
            try:
                m = parse(mv_str)
                if m in board.legal_moves:
                    mv, was_legal = m, True
                    break
            except Exception:
                pass
    if mv is None:
        mv = next(iter(board.legal_moves))          # deterministic fallback

    matched = (best_mv is not None and mv == best_mv)
    board.push(mv)
    if board.is_game_over():
        after_cp = 10000 if board.is_checkmate() else 0
    else:
        after_cp = eng.analyse(board, chess.engine.Limit(depth=depth))["score"].pov(turn).score(mate_score=10000)
    return {"legal": was_legal, "top1": matched, "cp_loss": max(0, best_cp - after_cp)}


def summarize(rows, step=None):
    n = len(rows)
    out = {
        "legal_rate": sum(r["legal"] for r in rows) / n,
        "top1":       sum(r["top1"] for r in rows) / n,
        "cp_loss":    sum(r["cp_loss"] for r in rows) / n,
    }
    if step is not None:
        out["step"] = step
    return out
