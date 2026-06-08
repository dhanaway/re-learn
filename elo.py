#!/usr/bin/env python
# elo.py
# Measure playing strength by running full games through the standard UCI
# interface (uci_engine.py) and converting results to Elo.
#
#   python elo.py ./qwen3-1.7b-chess-merged
#
# Opponents (gauntlet, trained model alternates colors, each opening played both ways):
#   - base model            -> RELATIVE Elo gain (headline "did RL help" number)
#   - random mover          -> a floor: can it beat random?
#   - Stockfish Skill Level 0 -> a weak calibrated-ish anchor
#
# A random OPENING BOOK makes every game distinct (otherwise near-deterministic
# engines replay the same game and the result is meaningless). Every game is also
# written to elo_games.pgn for official ordo/bayeselo rating later.
import os
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")
os.environ.setdefault("UCI_GPU_MEM", "0.35")               # so trained+base fit together
os.environ.setdefault("UCI_TEMP", "0.5")                   # some stochasticity on top of openings

import sys
import math
import random
import chess
import chess.engine
import chess.pgn

SF_PATH    = "/usr/games/stockfish"
TRAINED    = sys.argv[1] if len(sys.argv) > 1 else "./qwen3-1.7b-chess-merged"
GAMES      = int(os.environ.get("ELO_GAMES", "12"))        # per opponent (even; slow, tune up)
MOVE_LIMIT = 160                                           # ply cap before adjudication
PGN_OUT    = "elo_games.pgn"


def model_engine(path):
    # Generous timeout: the first move triggers a one-time ~80s vLLM load, far
    # longer than python-chess's 10s default (which times out before it's ready).
    return chess.engine.SimpleEngine.popen_uci(
        ["python", "-u", "uci_engine.py", path], timeout=600)


def safe_quit(engine):
    """vLLM workers are slow to die; never let cleanup crash the script."""
    for fn in ("quit", "close"):
        try:
            getattr(engine, fn)()
        except Exception:
            pass


def random_opening(plies=8):
    board = chess.Board()
    for _ in range(plies):
        if board.is_game_over():
            break
        board.push(random.choice(list(board.legal_moves)))
    if board.is_game_over():
        return random_opening(plies)
    return [m.uci() for m in board.move_stack]


def adjudicate(board, judge):
    cp = judge.analyse(board, chess.engine.Limit(depth=12))["score"].white().score(mate_score=10000)
    if cp is None:
        return 0.5
    return 1.0 if cp > 150 else (0.0 if cp < -150 else 0.5)


def play_game(white, black, judge, limit, opening):
    board = chess.Board()
    for mv in opening:
        board.push_uci(mv)
    while not board.is_game_over(claim_draw=True) and board.ply() < MOVE_LIMIT:
        engine = white if board.turn == chess.WHITE else black
        try:
            mv = engine.play(board, limit).move
        except Exception:
            mv = None
        if mv is None or mv not in board.legal_moves:
            return (0.0 if board.turn == chess.WHITE else 1.0), board   # that side forfeits
        board.push(mv)
    oc = board.outcome(claim_draw=True)
    if oc and oc.winner is not None:
        return (1.0 if oc.winner == chess.WHITE else 0.0), board
    if board.ply() >= MOVE_LIMIT:
        return adjudicate(board, judge), board
    return 0.5, board


def elo_diff(score, n):
    p = min(max(score, 0.5 / n), 1 - 0.5 / n)              # clamp so 0%/100% stay finite
    return -400.0 * math.log10(1.0 / p - 1.0)


def bootstrap_ci(results, iters=2000):
    n = len(results)
    deltas = sorted(elo_diff(sum(random.choice(results) for _ in range(n)) / n, n)
                    for _ in range(iters))
    return deltas[int(0.025 * iters)], deltas[int(0.975 * iters)]


def match(trained, opp, opp_name, judge, limit, pgn_fh):
    results = []
    for opening in [random_opening() for _ in range(max(1, GAMES // 2))]:
        for trained_white in (True, False):                # play each opening both ways (fair)
            white, black = (trained, opp) if trained_white else (opp, trained)
            wscore, board = play_game(white, black, judge, limit, opening)
            results.append(wscore if trained_white else 1.0 - wscore)
            game = chess.pgn.Game.from_board(board)
            game.headers["White"] = "trained" if trained_white else opp_name
            game.headers["Black"] = opp_name if trained_white else "trained"
            print(game, file=pgn_fh, end="\n\n", flush=True)
    n = len(results)
    w, d, l = results.count(1.0), results.count(0.5), results.count(0.0)
    score = sum(results) / n
    lo, hi = bootstrap_ci(results)
    line = (f"vs {opp_name:<14} {n} games  W-D-L = {w}-{d}-{l}  "
            f"score = {score:5.1%}  ΔElo = {elo_diff(score, n):+.0f}  [{lo:+.0f}, {hi:+.0f}]")
    print(line)
    return line


def selftest(engine, name, limit):
    try:
        mv = engine.play(chess.Board(), limit).move
        print(f"  [{name}] OK (first move {mv})")
        return True
    except Exception as e:
        print(f"  [{name}] FAILED to produce a move: {e}")
        return False


def main():
    judge = chess.engine.SimpleEngine.popen_uci(SF_PATH)   # full-strength adjudicator
    limit = chess.engine.Limit(time=0.1)                   # think time (model ignores it)
    pgn_fh = open(PGN_OUT, "w")
    summary = []

    print(f"=== Elo gauntlet for {TRAINED} ({GAMES} games/opponent) ===")
    print("First move per model engine takes ~80s while vLLM loads...\n")

    trained = model_engine(TRAINED)
    if not selftest(trained, "trained", limit):
        print("Trained engine can't move — fix the model load first (see standalone test).")
        safe_quit(trained); safe_quit(judge); pgn_fh.close(); return

    # 1) head-to-head vs base (the headline relative number)
    try:
        base = model_engine("Qwen/Qwen3-1.7B")
        if selftest(base, "base", limit):
            summary.append(match(trained, base, "base", judge, limit, pgn_fh))
        else:
            summary.append("vs base           skipped: engine self-test failed")
        safe_quit(base)
    except Exception as e:
        summary.append(f"vs base           skipped: {e}")

    # 2) vs random mover (floor)
    try:
        rnd = model_engine("random")
        summary.append(match(trained, rnd, "random", judge, limit, pgn_fh))
        safe_quit(rnd)
    except Exception as e:
        summary.append(f"vs random         skipped: {e}")

    # 3) vs weak Stockfish (Skill Level 0)
    try:
        sf0 = chess.engine.SimpleEngine.popen_uci(SF_PATH)
        sf0.configure({"Skill Level": 0})
        summary.append(match(trained, sf0, "stockfish-sk0", judge, limit, pgn_fh))
        safe_quit(sf0)
    except Exception as e:
        summary.append(f"vs stockfish-sk0  skipped: {e}")

    safe_quit(trained)
    safe_quit(judge)
    pgn_fh.close()

    # clean recap at the very end, so results aren't buried in vLLM startup noise
    header = f"RESULTS  —  {TRAINED}  ({GAMES} games/opponent)"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for line in summary:
        print(line)
    with open("elo_results.txt", "a") as f:                 # append -> both runs accumulate
        f.write(f"\n=== {TRAINED} ({GAMES} games/opponent) ===\n")
        for line in summary:
            f.write(line + "\n")
    print(f"\nGames -> {PGN_OUT};  results appended to elo_results.txt (cat it to compare runs)")


if __name__ == "__main__":
    main()
