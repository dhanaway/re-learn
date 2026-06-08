# chess_reward.py
# The reward function. For each move the model proposes, Stockfish scores how
# good it was (centipawn loss). This is the entire training signal.
#
# DEPTH=6 here is tuned for the SMOKE TEST (fast). For the full 14B run, bump to 10.
import re
import chess
import chess.engine

SF_PATH = "/usr/games/stockfish"   # from `which stockfish` on the pod
DEPTH   = 6                        # engine search depth for reward  ([14B] use 10)

_engine = None
_best_cache = {}                   # cache best-eval per FEN (same for all G completions)


def _get_engine():
    global _engine
    if _engine is None:
        _engine = chess.engine.SimpleEngine.popen_uci(SF_PATH)
    return _engine


def _best_eval(fen):
    """Stockfish's evaluation of the position + its top move, from the mover's POV."""
    if fen in _best_cache:
        return _best_cache[fen]
    board = chess.Board(fen)
    info = _get_engine().analyse(board, chess.engine.Limit(depth=DEPTH))
    cp   = info["score"].pov(board.turn).score(mate_score=10000)
    best = info["pv"][0] if info.get("pv") else None
    _best_cache[fen] = (cp, best)
    return cp, best


def _text(c):
    # GRPO passes completions as a list of chat messages (or sometimes a string).
    if isinstance(c, str):
        return c
    if isinstance(c, list) and c and isinstance(c[-1], dict):
        return c[-1].get("content", "")
    return str(c)


def _extract_move(text):
    m = re.search(r"\\boxed\{([^}]*)\}", text)           # primary format
    if not m:
        m = re.search(r"<answer>(.*?)</answer>", text, re.S)
    return m.group(1).strip() if m else None


def chess_reward(completions, fen, **kwargs):
    """GRPO calls this with one entry per generated completion.
    `fen` is the dataset column, auto-aligned (repeated once per generation)."""
    rewards = []
    for comp, f in zip(completions, fen):
        mv_str = _extract_move(_text(comp))
        if mv_str is None:
            rewards.append(-1.0)            # produced no parseable move
            continue

        board = chess.Board(f)
        move = None
        for parse in (board.parse_san, board.parse_uci):
            try:
                move = parse(mv_str)
                break
            except Exception:
                pass
        if move is None or move not in board.legal_moves:
            rewards.append(-0.5)            # illegal / unparseable move
            continue

        r = 0.2                              # reward: legal + parseable
        best_cp, best_move = _best_eval(f)
        original = board.turn
        board.push(move)
        if board.is_game_over():
            # Stockfish can't analyse a finished game. Checkmate by the mover is
            # the best possible outcome; a draw is neutral.
            after_cp = 10000 if board.is_checkmate() else 0
        else:
            info = _get_engine().analyse(board, chess.engine.Limit(depth=DEPTH))
            after_cp = info["score"].pov(original).score(mate_score=10000)
        cp_loss  = max(0, best_cp - after_cp)

        r += max(0.0, 1.0 - cp_loss / 200.0)         # quality: closer to best = higher
        if best_move is not None and move == best_move:
            r += 0.3                                 # bonus for the engine's top move
        rewards.append(r)
    return rewards
