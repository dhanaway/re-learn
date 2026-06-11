#!/usr/bin/env python
"""
chess_microgpt.py — a from-scratch, pure-Python (stdlib-only) scalar-autograd GPT,
in the style of Karpathy's microGPT, adapted into a chess engine.

It learns board -> best move (Stockfish-labelled), plays whole games, and is
Elo-measurable by reusing elo.py.  The MODEL is pure Python (no torch/numpy):
a micrograd-style `Value` autograd + a tiny transformer.  python-chess + Stockfish
are only the "environment" (board rules + move labels + Elo judge) — exactly like
the original gist used urllib for its data.

HONEST EXPECTATIONS: this is CPU-only, single-threaded, slow, and weak. It does NOT
use a GPU. Target outcome: beats random modestly, loses heavily to Stockfish, and
ALWAYS plays a legal move (decoding is masked to the legal set).

Subcommands:
  gen    generate + Stockfish-label positions -> positions.json   (needs python-chess + Stockfish)
  train  train the pure-Python GPT on the cached ints -> weights.json   (STDLIB ONLY, no chess)
  uci    run as a UCI engine (legal-masked decode)                 (needs python-chess)
  play   play a quick game vs a random mover (sanity)              (needs python-chess)
  elo    Elo gauntlet vs random + Stockfish-sk0, reusing elo.py    (needs python-chess + Stockfish)
  selftest  encode/decode round-trip check                        (needs python-chess)

Examples:
  python chess_microgpt.py gen   --n 300 --out positions.json
  python chess_microgpt.py train --data positions.json --num-steps 1000 --out weights.json
  python chess_microgpt.py uci   weights.json
  ELO_GAMES=12 python chess_microgpt.py elo --weights weights.json
"""
import os
import sys
import math
import json
import random

# ----------------------------------------------------------------------------
# 1. AUTOGRAD CORE  (micrograd-style scalar Value)
# ----------------------------------------------------------------------------
class Value:
    __slots__ = ("data", "grad", "_backward", "_prev")

    def __init__(self, data, _children=()):
        self.data = data
        self.grad = 0.0
        self._backward = None
        self._prev = _children

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))
        def _b():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _b
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))
        def _b():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _b
        return out

    def __pow__(self, p):  # p is a plain number
        out = Value(self.data ** p, (self,))
        def _b():
            self.grad += (p * self.data ** (p - 1)) * out.grad
        out._backward = _b
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,))
        def _b():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _b
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,))
        def _b():
            self.grad += e * out.grad
        out._backward = _b
        return out

    def log(self):
        out = Value(math.log(self.data), (self,))
        def _b():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _b
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other if isinstance(other, Value) else Value(-other))

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other ** -1.0

    def backward(self):
        # iterative post-order topo sort (avoids recursion-depth limits on big graphs)
        topo = []
        visited = set()
        stack = [(self, False)]
        while stack:
            v, processed = stack.pop()
            if processed:
                topo.append(v)
                continue
            if id(v) in visited:
                continue
            visited.add(id(v))
            stack.append((v, True))
            for child in v._prev:
                if id(child) not in visited:
                    stack.append((child, False))
        self.grad = 1.0
        for v in reversed(topo):
            if v._backward is not None:
                v._backward()


# ----------------------------------------------------------------------------
# 2. NN HELPERS (over Value)
# ----------------------------------------------------------------------------
def linear(x, W):  # x: list[Value] (in), W: list (out) of list (in) -> list[Value] (out)
    out = []
    for row in W:
        acc = row[0] * x[0]
        for i in range(1, len(x)):
            acc = acc + row[i] * x[i]
        out.append(acc)
    return out


def dot(a, b):
    acc = a[0] * b[0]
    for i in range(1, len(a)):
        acc = acc + a[i] * b[i]
    return acc


def softmax(logits):
    mx = max(l.data for l in logits)
    exps = [(l - mx).exp() for l in logits]
    s = exps[0]
    for e in exps[1:]:
        s = s + e
    inv = s ** -1.0
    return [e * inv for e in exps]


def rmsnorm(x):
    n = len(x)
    ss = x[0] * x[0]
    for i in range(1, n):
        ss = ss + x[i] * x[i]
    ss = ss * (1.0 / n)
    inv = (ss + 1e-5) ** -0.5
    return [xi * inv for xi in x]


# ----------------------------------------------------------------------------
# 3. VOCAB & ENCODING  (replaces the gist's char vocab)
# ----------------------------------------------------------------------------
PIECE_BASE  = 0     # 0=empty, 1..6 white PNBRQK, 7..12 black pnbrqk
STM_BASE    = 13    # 13 white-to-move, 14 black-to-move
CASTLE_BASE = 15    # 15..30  (4-bit mask K,Q,k,q)
SEP         = 31
SQ_BASE     = 32    # 32..95  square index (reused for from and to)
PROMO_BASE  = 96    # 96 none, 97 N, 98 B, 99 R, 100 Q
VOCAB_SIZE  = 101
PREFIX_LEN  = 67    # 64 squares + side + castle + SEP
MOVE_LEN    = 3     # from, to, promo
BLOCK_SIZE  = 72


def encode_board(board):
    import chess
    seq = []
    for sq in range(64):  # chess.A1==0 .. chess.H8==63
        p = board.piece_at(sq)
        if p is None:
            seq.append(0)
        else:
            base = 0 if p.color == chess.WHITE else 6
            seq.append(PIECE_BASE + base + p.piece_type)
    seq.append(STM_BASE + (0 if board.turn == chess.WHITE else 1))
    mask = (int(board.has_kingside_castling_rights(chess.WHITE)) << 3) \
         | (int(board.has_queenside_castling_rights(chess.WHITE)) << 2) \
         | (int(board.has_kingside_castling_rights(chess.BLACK)) << 1) \
         |  int(board.has_queenside_castling_rights(chess.BLACK))
    seq.append(CASTLE_BASE + mask)
    seq.append(SEP)
    return seq  # length PREFIX_LEN


def encode_move(move):
    f = SQ_BASE + move.from_square
    t = SQ_BASE + move.to_square
    promo = PROMO_BASE if not move.promotion else PROMO_BASE + (move.promotion - 1)
    return [f, t, promo]


def decode_move_tokens(ft, tt, pt):
    import chess
    promo = None if pt == PROMO_BASE else (pt - PROMO_BASE + 1)
    return chess.Move(ft - SQ_BASE, tt - SQ_BASE, promotion=promo)


# ----------------------------------------------------------------------------
# 4. MODEL — params + training forward (Value) + inference forward (float)
# ----------------------------------------------------------------------------
def init_params(cfg):
    E, V, B, nL = cfg["n_embd"], cfg["vocab_size"], cfg["block_size"], cfg["n_layer"]

    def mat(out_d, in_d, std):
        return [[Value(random.gauss(0, std)) for _ in range(in_d)] for _ in range(out_d)]

    P = {}
    P["tok_emb"] = [[Value(random.gauss(0, 0.02)) for _ in range(E)] for _ in range(V)]
    P["pos_emb"] = [[Value(random.gauss(0, 0.02)) for _ in range(E)] for _ in range(B)]
    P["layers"] = []
    for _ in range(nL):
        s = 1.0 / math.sqrt(E)
        P["layers"].append({
            "wq": mat(E, E, s), "wk": mat(E, E, s), "wv": mat(E, E, s), "wo": mat(E, E, s),
            "w1": mat(4 * E, E, s), "w2": mat(E, 4 * E, 1.0 / math.sqrt(4 * E)),
        })
    P["head"] = mat(V, E, 1.0 / math.sqrt(E))
    return P


def collect_params(P):
    out = []
    for row in P["tok_emb"]:
        out.extend(row)
    for row in P["pos_emb"]:
        out.extend(row)
    for lp in P["layers"]:
        for key in ("wq", "wk", "wv", "wo", "w1", "w2"):
            for row in lp[key]:
                out.extend(row)
    for row in P["head"]:
        out.extend(row)
    return out


def forward_train(token_ids, loss_positions, P, cfg):
    """Teacher-forced forward; only builds logits at loss_positions (the rest just
    populate the KV cache, the big pure-Python speedup)."""
    E, H, nL = cfg["n_embd"], cfg["n_head"], cfg["n_layer"]
    D = E // H
    inv_sqrt_d = 1.0 / math.sqrt(D)
    keys = [[] for _ in range(nL)]
    values = [[] for _ in range(nL)]
    logits_at = {}
    for t, tok in enumerate(token_ids):
        x = [P["tok_emb"][tok][i] + P["pos_emb"][t][i] for i in range(E)]
        for L in range(nL):
            lp = P["layers"][L]
            xn = rmsnorm(x)
            keys[L].append(linear(xn, lp["wk"]))
            values[L].append(linear(xn, lp["wv"]))
            need_block = (L < nL - 1) or (t in loss_positions)
            if not need_block:
                continue
            q = linear(xn, lp["wq"])
            attn = [None] * E
            for h in range(H):
                qs = q[h * D:(h + 1) * D]
                scores = []
                for kj in keys[L]:
                    scores.append(dot(qs, kj[h * D:(h + 1) * D]) * inv_sqrt_d)
                w = softmax(scores)
                for d in range(D):
                    acc = w[0] * values[L][0][h * D + d]
                    for j in range(1, len(w)):
                        acc = acc + w[j] * values[L][j][h * D + d]
                    attn[h * D + d] = acc
            o = linear(attn, lp["wo"])
            x = [x[i] + o[i] for i in range(E)]
            hmid = [hh.relu() for hh in linear(rmsnorm(x), lp["w1"])]
            h2 = linear(hmid, lp["w2"])
            x = [x[i] + h2[i] for i in range(E)]
        if t in loss_positions:
            logits_at[t] = linear(rmsnorm(x), P["head"])
    return logits_at


def cross_entropy(logits, target):
    return -(softmax(logits)[target].log())


def sequence_loss(row, prefix_len, P, cfg):
    # position i predicts token i+1; move tokens sit at prefix_len, +1, +2
    loss_positions = {prefix_len - 1, prefix_len, prefix_len + 1}
    inputs = row[:prefix_len + 2]  # feed up to the to-square token; promo is a target only
    logits_at = forward_train(inputs, loss_positions, P, cfg)
    total = None
    n = 0
    for i in sorted(loss_positions):
        ce = cross_entropy(logits_at[i], row[i + 1])
        total = ce if total is None else total + ce
        n += 1
    return total * (1.0 / n)


# ---- inference float fast-path (no autograd) ----
class FloatGPT:
    def __init__(self, P, cfg):
        self.P, self.cfg = P, cfg
        self.E = cfg["n_embd"]; self.H = cfg["n_head"]; self.nL = cfg["n_layer"]
        self.D = self.E // self.H
        self.keys = [[] for _ in range(self.nL)]
        self.values = [[] for _ in range(self.nL)]
        self.pos = 0

    @staticmethod
    def _rms(x):
        n = len(x)
        ss = sum(v * v for v in x) / n
        inv = 1.0 / math.sqrt(ss + 1e-5)
        return [v * inv for v in x]

    @staticmethod
    def _lin(x, W):
        return [sum(row[i] * x[i] for i in range(len(x))) for row in W]

    def step(self, tok):
        P, E, H, D = self.P, self.E, self.H, self.D
        inv_sqrt_d = 1.0 / math.sqrt(D)
        x = [P["tok_emb"][tok][i] + P["pos_emb"][self.pos][i] for i in range(E)]
        for L in range(self.nL):
            lp = P["layers"][L]
            xn = self._rms(x)
            k = self._lin(xn, lp["wk"]); v = self._lin(xn, lp["wv"])
            self.keys[L].append(k); self.values[L].append(v)
            q = self._lin(xn, lp["wq"])
            attn = [0.0] * E
            for h in range(H):
                qs = q[h * D:(h + 1) * D]
                scores = [sum(qs[d] * kj[h * D + d] for d in range(D)) * inv_sqrt_d
                          for kj in self.keys[L]]
                mx = max(scores)
                ex = [math.exp(s - mx) for s in scores]
                Z = sum(ex)
                for d in range(D):
                    attn[h * D + d] = sum((ex[j] / Z) * self.values[L][j][h * D + d]
                                          for j in range(len(ex)))
            o = self._lin(attn, lp["wo"])
            x = [x[i] + o[i] for i in range(E)]
            hmid = [hh if hh > 0 else 0.0 for hh in self._lin(self._rms(x), lp["w1"])]
            h2 = self._lin(hmid, lp["w2"])
            x = [x[i] + h2[i] for i in range(E)]
        self.pos += 1
        return self._lin(self._rms(x), P["head"])


def sample_masked(logits, allowed, temp):
    vals = [logits[t] for t in allowed]
    if temp <= 1e-6 or len(vals) == 1:
        return allowed[max(range(len(vals)), key=lambda i: vals[i])]
    mx = max(vals)
    ws = [math.exp((x - mx) / temp) for x in vals]
    r = random.random() * sum(ws)
    acc = 0.0
    for tid, w in zip(allowed, ws):
        acc += w
        if r <= acc:
            return tid
    return allowed[-1]


def legal_masked_decode(board, P, cfg, temp):
    import chess
    legal = list(board.legal_moves)
    if not legal:
        return None
    g = FloatGPT(P, cfg)
    logits = None
    for tok in encode_board(board):
        logits = g.step(tok)
    # FROM square
    from_choices = sorted({m.from_square for m in legal})
    from_tok = sample_masked(logits, [SQ_BASE + s for s in from_choices], temp)
    from_sq = from_tok - SQ_BASE
    logits = g.step(from_tok)
    # TO square (legal destinations from that square)
    to_choices = sorted({m.to_square for m in legal if m.from_square == from_sq})
    to_tok = sample_masked(logits, [SQ_BASE + s for s in to_choices], temp)
    to_sq = to_tok - SQ_BASE
    logits = g.step(to_tok)
    # PROMOTION
    promo_moves = [m for m in legal if m.from_square == from_sq and m.to_square == to_sq]
    promo_set = sorted({(m.promotion or 0) for m in promo_moves})
    if promo_set == [0]:
        promo = None
    else:
        allowed = [PROMO_BASE if pt == 0 else PROMO_BASE + (pt - 1) for pt in promo_set]
        ptok = sample_masked(logits, allowed, temp)
        promo = None if ptok == PROMO_BASE else (ptok - PROMO_BASE + 1)
    mv = chess.Move(from_sq, to_sq, promotion=promo)
    if mv not in board.legal_moves:  # belt-and-suspenders; never emit illegal
        mv = promo_moves[0] if promo_moves else legal[0]
    return mv


# ----------------------------------------------------------------------------
# 5. WEIGHTS I/O
# ----------------------------------------------------------------------------
def _to_floats(x):
    if isinstance(x, Value):
        return x.data
    if isinstance(x, list):
        return [_to_floats(e) for e in x]
    if isinstance(x, dict):
        return {k: _to_floats(v) for k, v in x.items()}
    return x


def save_weights(P, cfg, path):
    json.dump({"cfg": cfg, "params": _to_floats(P)}, open(path, "w"))


def load_weights(path):
    d = json.load(open(path))
    return d["params"], d["cfg"]  # params are nested floats; FloatGPT uses them directly


# ----------------------------------------------------------------------------
# 6. SHARED: random position (lazy chess import)
# ----------------------------------------------------------------------------
def random_position(max_plies=40):
    import chess
    b = chess.Board()
    for _ in range(random.randint(6, max_plies)):
        if b.is_game_over():
            break
        b.push(random.choice(list(b.legal_moves)))
    if b.is_game_over():
        return random_position(max_plies)
    return b


# ----------------------------------------------------------------------------
# 7. SUBCOMMANDS
# ----------------------------------------------------------------------------
def _stockfish_path():
    """Find Stockfish: $STOCKFISH_PATH, then common pod/Mac/Linux locations, then PATH."""
    import shutil
    env = os.environ.get("STOCKFISH_PATH")
    if env:
        return env
    for p in ("/usr/games/stockfish", "/opt/homebrew/bin/stockfish", "/usr/local/bin/stockfish"):
        if os.path.exists(p):
            return p
    w = shutil.which("stockfish")
    if w:
        return w
    raise SystemExit("Stockfish not found. Install it (`brew install stockfish` on Mac, "
                     "`apt-get install stockfish` on Linux) or set STOCKFISH_PATH.")


def cmd_gen(args):
    import chess
    import chess.engine
    sf = chess.engine.SimpleEngine.popen_uci(_stockfish_path())
    random.seed(args.seed)
    seen, rows = set(), []
    try:
        while len(rows) < args.n:
            b = random_position()
            key = b.board_fen() + (" w" if b.turn else " b")
            if key in seen:
                continue
            seen.add(key)
            info = sf.analyse(b, chess.engine.Limit(depth=args.depth))
            best = info["pv"][0] if info.get("pv") else None
            if best is None or best not in b.legal_moves:
                continue
            rows.append(encode_board(b) + encode_move(best))
            if len(rows) % 50 == 0:
                print(f"gen {len(rows)}/{args.n}", file=sys.stderr)
    finally:
        sf.quit()
    json.dump({"block_size": BLOCK_SIZE, "vocab_size": VOCAB_SIZE,
               "prefix_len": PREFIX_LEN, "rows": rows}, open(args.out, "w"))
    print(f"wrote {len(rows)} rows -> {args.out}", file=sys.stderr)


def cmd_train(args):
    data = json.load(open(args.data))
    cfg = {"n_layer": args.n_layer, "n_embd": args.n_embd, "n_head": args.n_head,
           "block_size": data["block_size"], "vocab_size": data["vocab_size"]}
    prefix_len, rows = data["prefix_len"], data["rows"]
    random.seed(args.seed)
    P = init_params(cfg)
    flat = collect_params(P)
    m = [0.0] * len(flat)
    v = [0.0] * len(flat)
    b1, b2, eps = args.beta1, args.beta2, 1e-8
    print(f"params={len(flat)}  rows={len(rows)}  cfg={cfg}", file=sys.stderr)
    for step in range(args.num_steps):
        row = rows[step % len(rows)]
        for p in flat:
            p.grad = 0.0
        loss = sequence_loss(row, prefix_len, P, cfg)
        loss.backward()
        lr_t = args.lr * (1 - step / args.num_steps)
        t = step + 1
        for idx in range(len(flat)):
            p = flat[idx]
            g = p.grad
            m[idx] = b1 * m[idx] + (1 - b1) * g
            v[idx] = b2 * v[idx] + (1 - b2) * g * g
            mhat = m[idx] / (1 - b1 ** t)
            vhat = v[idx] / (1 - b2 ** t)
            p.data -= lr_t * mhat / (math.sqrt(vhat) + eps)
        if step % 50 == 0:
            print(f"step {step} loss {loss.data:.4f} lr {lr_t:.5f}", file=sys.stderr)
    save_weights(P, cfg, args.out)
    print(f"saved weights -> {args.out}", file=sys.stderr)


def set_position_uci(tokens):
    import chess
    board = chess.Board()
    if tokens and tokens[0] == "fen":
        board.set_fen(" ".join(tokens[1:7]))
    moves_idx = tokens.index("moves") + 1 if "moves" in tokens else len(tokens)
    for mv in tokens[moves_idx:]:
        board.push_uci(mv)
    return board


def cmd_uci(args):
    import chess
    is_random = (args.weights == "random")          # built-in random UCI opponent (no model)
    P = cfg = None
    if not is_random:
        P, cfg = load_weights(args.weights)
    temp = float(os.environ.get("UCI_TEMP", "0.5"))
    board = chess.Board()
    for line in sys.stdin:
        line = line.strip()
        if line == "uci":
            print("id name microGPT-chess" + ("-random" if is_random else ""))
            print("id author re-learn")
            print("uciok", flush=True)
        elif line == "isready":
            print("readyok", flush=True)
        elif line == "ucinewgame":
            board = chess.Board()
        elif line.startswith("position"):
            board = set_position_uci(line.split()[1:])
        elif line.startswith("go"):
            if is_random:
                mv = random.choice(list(board.legal_moves))
            else:
                mv = legal_masked_decode(board, P, cfg, temp)
            if mv is None or mv not in board.legal_moves:
                mv = next(iter(board.legal_moves))
            print(f"bestmove {board.uci(mv)}", flush=True)
        elif line == "quit":
            break


def cmd_play(args):
    import chess
    P, cfg = load_weights(args.weights)
    temp = float(os.environ.get("UCI_TEMP", "0.5"))
    board = chess.Board()
    while not board.is_game_over(claim_draw=True) and board.ply() < args.max_plies:
        if board.turn == chess.WHITE:
            mv = legal_masked_decode(board, P, cfg, temp)
        else:
            mv = random.choice(list(board.legal_moves))
        if mv is None or mv not in board.legal_moves:
            mv = next(iter(board.legal_moves))
        print(("microGPT" if board.turn == chess.WHITE else "random") + ": " + board.san(mv),
              file=sys.stderr)
        board.push(mv)
    print("result:", board.result(claim_draw=True), file=sys.stderr)


def cmd_elo(args):
    import chess
    import chess.engine
    import elo as E  # reuse the existing gauntlet/Elo math unchanged
    weights = args.weights
    self_path = os.path.abspath(__file__)
    py = sys.executable                              # same interpreter (works on Mac + pod)
    sf_path = _stockfish_path()

    judge = chess.engine.SimpleEngine.popen_uci(sf_path)
    limit = chess.engine.Limit(time=0.1)
    pgn_fh = open("elo_games.pgn", "w")
    summary = []
    trained = chess.engine.SimpleEngine.popen_uci(
        [py, "-u", self_path, "uci", weights], timeout=600)
    if not E.selftest(trained, "microgpt", limit):
        print("microGPT engine self-test failed", file=sys.stderr)
        E.safe_quit(trained); E.safe_quit(judge); pgn_fh.close(); return

    try:
        rnd = chess.engine.SimpleEngine.popen_uci(
            [py, "-u", self_path, "uci", "random"], timeout=600)   # self-contained random opponent
        summary.append(E.match(trained, rnd, "random", judge, limit, pgn_fh))
        E.safe_quit(rnd)
    except Exception as e:
        summary.append(f"vs random         skipped: {e}")

    try:
        sf0 = chess.engine.SimpleEngine.popen_uci(sf_path)
        sf0.configure({"Skill Level": 0})
        summary.append(E.match(trained, sf0, "stockfish-sk0", judge, limit, pgn_fh))
        E.safe_quit(sf0)
    except Exception as e:
        summary.append(f"vs stockfish-sk0  skipped: {e}")

    E.safe_quit(trained); E.safe_quit(judge); pgn_fh.close()
    print("\n=== microGPT Elo ===")
    for line in summary:
        print(line)


def cmd_selftest(args):
    import chess
    random.seed(0)
    ok = 0
    for _ in range(50):
        b = random_position()
        for mv in list(b.legal_moves)[:3]:
            toks = encode_board(b) + encode_move(mv)
            assert len(toks) == PREFIX_LEN + MOVE_LEN, len(toks)
            assert all(0 <= tk < VOCAB_SIZE for tk in toks)
            dec = decode_move_tokens(toks[PREFIX_LEN], toks[PREFIX_LEN + 1], toks[PREFIX_LEN + 2])
            assert dec == mv, (b.fen(), mv, dec)
            assert dec in b.legal_moves
            ok += 1
    print(f"selftest OK: {ok} encode/decode round-trips passed")


# ----------------------------------------------------------------------------
# 8. main
# ----------------------------------------------------------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser(description="pure-Python microGPT chess engine")
    sub = ap.add_subparsers(dest="cmd")

    g = sub.add_parser("gen")
    g.add_argument("--n", type=int, default=300)
    g.add_argument("--out", default="positions.json")
    g.add_argument("--depth", type=int, default=6)
    g.add_argument("--seed", type=int, default=0)

    t = sub.add_parser("train")
    t.add_argument("--data", default="positions.json")
    t.add_argument("--out", default="weights.json")
    t.add_argument("--n-embd", dest="n_embd", type=int, default=16)
    t.add_argument("--n-layer", dest="n_layer", type=int, default=1)
    t.add_argument("--n-head", dest="n_head", type=int, default=4)
    t.add_argument("--num-steps", dest="num_steps", type=int, default=1000)
    t.add_argument("--lr", type=float, default=0.01)
    t.add_argument("--beta1", type=float, default=0.85)
    t.add_argument("--beta2", type=float, default=0.99)
    t.add_argument("--seed", type=int, default=0)

    u = sub.add_parser("uci")
    u.add_argument("weights")

    p = sub.add_parser("play")
    p.add_argument("--weights", default="weights.json")
    p.add_argument("--max-plies", dest="max_plies", type=int, default=160)

    e = sub.add_parser("elo")
    e.add_argument("--weights", default="weights.json")

    sub.add_parser("selftest")

    args = ap.parse_args()
    if args.cmd == "gen":
        cmd_gen(args)
    elif args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "uci":
        cmd_uci(args)
    elif args.cmd == "play":
        cmd_play(args)
    elif args.cmd == "elo":
        cmd_elo(args)
    elif args.cmd == "selftest":
        cmd_selftest(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
