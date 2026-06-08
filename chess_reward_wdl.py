# chess_reward_wdl.py
# Alternative reward: optimize WIN PROBABILITY (WDL) instead of raw centipawn loss.
# Literature favors this as the better strength/Elo proxy:
#   - Leela Chess Zero's value head is WDL (a single eval can't tell "dead drawn"
#     from "50% win / 50% loss").
#   - DeepMind's searchless_chess trains on win-% targets, not centipawns.
# Win-prob is also ~linear in what decides games (and thus Elo); centipawns are a
# nonlinear transform that over-weights swings in already-decided positions.
#
# To A/B against the centipawn reward, edit train.py:
#     from chess_reward_wdl import chess_reward_wdl
#     ...
#     reward_funcs = [chess_reward_wdl],
#
# Validate with elo.py which reward actually yields higher Elo — that's the point
# of building the Elo monitor first.
import chess
import chess.engine
from chess_reward import SF_PATH, _get_engine, _text, _extract_move

DEPTH    = 6
BLEND_CP = 0.0      # >0 adds a small centipawn term (keeps gradient in decided positions)

_best_cache = {}


def _win_prob(score_pov, ply):
    """cp/mate Score (from a fixed POV) -> win expectation in [0,1]."""
    try:
        return score_pov.wdl(model="sf", ply=ply).expectation()
    except Exception:
        cp = score_pov.score(mate_score=10000)
        return 1.0 / (1.0 + 10 ** (-cp / 400.0))          # logistic fallback


def _best(fen):
    if fen in _best_cache:
        return _best_cache[fen]
    board = chess.Board(fen)
    info = _get_engine().analyse(board, chess.engine.Limit(depth=DEPTH))
    wp  = _win_prob(info["score"].pov(board.turn), board.ply())
    cp  = info["score"].pov(board.turn).score(mate_score=10000)
    mv  = info["pv"][0] if info.get("pv") else None
    _best_cache[fen] = (wp, cp, mv)
    return _best_cache[fen]


def chess_reward_wdl(completions, fen, **kwargs):
    rewards = []
    for comp, f in zip(completions, fen):
        mv_str = _extract_move(_text(comp))
        if mv_str is None:
            rewards.append(-1.0)
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
            rewards.append(-0.5)
            continue

        r = 0.2
        best_wp, best_cp, best_move = _best(f)
        original = board.turn
        board.push(move)
        if board.is_game_over():
            wp_after = 1.0 if board.is_checkmate() else 0.5
            cp_after = 10000 if board.is_checkmate() else 0
        else:
            info = _get_engine().analyse(board, chess.engine.Limit(depth=DEPTH))
            wp_after = _win_prob(info["score"].pov(original), board.ply())
            cp_after = info["score"].pov(original).score(mate_score=10000)

        r += max(0.0, 1.0 - (best_wp - wp_after))          # win-probability preserved
        if BLEND_CP > 0:
            r += BLEND_CP * max(0.0, 1.0 - max(0, best_cp - cp_after) / 200.0)
        if best_move is not None and move == best_move:
            r += 0.3
        rewards.append(r)
    return rewards
