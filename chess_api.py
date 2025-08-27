#!/usr/bin/env python3
"""
Chess API in Python (Flask + python-chess)

Features
--------
- Create and manage multiple chess games in memory
- Make moves via UCI (e2e4) or SAN (e4) notation
- Get current state (FEN, turn, checks, result, legal moves, simple material score)
- Undo the last move
- Export PGN
- List legal moves
- Bot move (random or capture-priority, or simple minimax hint)
- Perft helper (node count) for a given FEN
- Light auth via per-game secret (optional)

Quickstart
----------
1) Install deps:
   pip install flask python-chess

2) Run server:
   python chess_api.py

3) Use with curl/httpie, e.g.:
   http POST :5000/game
   http POST :5000/game/<gid>/move move=e2e4
   http GET  :5000/game/<gid>
   http POST :5000/game/<gid>/undo
   http GET  :5000/game/<gid>/pgn

Notes
-----
- This service keeps games in memory only. For production, add persistence.
- The built-in bot and hint use a tiny evaluation + shallow search; it is NOT
  a strong engine, but works as an example without Stockfish.
- If you want a stronger engine, you can integrate an external engine via
  python-chess's UCI interface, but that adds complexity and a native binary.
"""
from __future__ import annotations

import os
import json
import time
import uuid
import random
import logging
import functools
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

from flask import Flask, request, jsonify, abort

try:
    import chess
    import chess.pgn
    import chess.polyglot
except Exception as e:
    raise SystemExit("python-chess is required. Install with: pip install python-chess")

# ----------------------------------------------------------------------------
# Config & Utilities
# ----------------------------------------------------------------------------

PORT = int(os.getenv("PORT", "5000"))
HOST = os.getenv("HOST", "0.0.0.0")
DEBUG = bool(int(os.getenv("DEBUG", "1")))

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


def json_response(data, status=200):
    """Consistent JSON responses with CORS headers."""
    resp = jsonify(data)
    resp.status_code = status
    # Very light CORS for demos; adjust for production.
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return resp


@app.after_request
def add_no_cache_headers(resp):
    # Avoid caching responses while developing/testing.
    resp.headers['Cache-Control'] = 'no-store'
    return resp


# ----------------------------------------------------------------------------
# Models & Game Store
# ----------------------------------------------------------------------------

@dataclass
class MoveRecord:
    move_uci: str
    san: str
    fen_before: str
    fen_after: str
    clock: float


@dataclass
class Game:
    id: str
    secret: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    board: chess.Board = field(default_factory=chess.Board)
    history: List[MoveRecord] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self, include_legal: bool = False, include_history: bool = False) -> Dict:
        board = self.board
        legal_moves = [m.uci() for m in board.legal_moves] if include_legal else None
        result = None
        if board.is_game_over():
            result = board.result(claim_draw=True)
        return {
            "id": self.id,
            "created_at": self.created_at,
            "turn": 'w' if board.turn else 'b',
            "fen": board.fen(),
            "pgn": self.to_pgn_string(max_headers=12),
            "is_check": board.is_check(),
            "is_game_over": board.is_game_over(),
            "result": result,  # "1-0", "0-1", "1/2-1/2" or None
            "reason": game_over_reason(board) if board.is_game_over() else None,
            "material_score": material_score(board),
            "legal_moves": legal_moves,
            "history": [asdict(h) for h in self.history] if include_history else None,
            "tags": self.tags,
        }

    # -- PGN helpers ----------------------------------------------------------
    def to_pgn_game(self) -> chess.pgn.Game:
        game = chess.pgn.Game()
        # Minimal headers; you can add more via self.tags
        game.headers["Event"] = self.tags.get("Event", "Casual Game")
        game.headers["Site"] = self.tags.get("Site", "Localhost")
        game.headers["Date"] = self.tags.get("Date", time.strftime("%Y.%m.%d"))
        game.headers["Round"] = self.tags.get("Round", "?")
        game.headers["White"] = self.tags.get("White", "White")
        game.headers["Black"] = self.tags.get("Black", "Black")
        game.headers["Result"] = self.board.result(claim_draw=True) if self.board.is_game_over() else "*"
        node = game
        board = chess.Board()
        for rec in self.history:
            move = chess.Move.from_uci(rec.move_uci)
            node = node.add_variation(move)
            board.push(move)
        return game

    def to_pgn_string(self, max_headers: int = 10) -> str:
        game = self.to_pgn_game()
        # Limit long headers to keep payload small
        # python-chess prints headers then movetext
        exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
        text = game.accept(exporter)
        # Optionally trim headers (simple approach)
        lines = text.splitlines()
        trimmed = []
        hdr_count = 0
        for line in lines:
            if line.startswith("[") and "]" in line:
                hdr_count += 1
                if hdr_count <= max_headers:
                    trimmed.append(line)
            else:
                trimmed.append(line)
        return "\n".join(trimmed)

    # -- Move & state ops -----------------------------------------------------
    def apply_move(self, move_str: str) -> Tuple[bool, str]:
        """Apply a move in UCI or SAN. Returns (ok, message)."""
        board = self.board
        fen_before = board.fen()

        move = None
        # Try UCI first
        try:
            if len(move_str) in (4, 5):
                move = chess.Move.from_uci(move_str)
        except Exception:
            move = None

        if move is None:
            # Try SAN
            try:
                move = board.parse_san(move_str)
            except Exception:
                return False, f"Invalid move notation: {move_str}"

        if move not in board.legal_moves:
            return False, f"Illegal move: {move_str}"

        san = board.san(move)
        board.push(move)
        fen_after = board.fen()
        self.history.append(MoveRecord(move.uci(), san, fen_before, fen_after, time.time()))
        return True, san

    def undo(self) -> bool:
        if not self.history:
            return False
        self.board.pop()
        self.history.pop()
        return True


class GameStore:
    """Simple in-memory store for games."""

    def __init__(self):
        self._games: Dict[str, Game] = {}

    def create(self, secret: Optional[str] = None, tags: Optional[Dict[str, str]] = None, fen: Optional[str] = None) -> Game:
        gid = uuid.uuid4().hex[:12]
        game = Game(id=gid, secret=secret, board=chess.Board(fen) if fen else chess.Board())
        if tags:
            game.tags.update(tags)
        self._games[gid] = game
        return game

    def get(self, gid: str) -> Game:
        game = self._games.get(gid)
        if not game:
            raise KeyError("Game not found")
        return game

    def delete(self, gid: str) -> bool:
        return self._games.pop(gid, None) is not None

    def list_ids(self) -> List[str]:
        return list(self._games.keys())


GAMES = GameStore()


# ----------------------------------------------------------------------------
# Chess helpers: evaluation, search, perft, reasons
# ----------------------------------------------------------------------------

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,  # King value handled via mate scores
}

MATE_SCORE = 10_000


def material_score(board: chess.Board) -> int:
    """Simple material count: positive for side-to-move advantage."""
    score = 0
    for piece_type, value in PIECE_VALUES.items():
        score += len(board.pieces(piece_type, chess.WHITE)) * value
        score -= len(board.pieces(piece_type, chess.BLACK)) * value
    # Perspective: side to move
    return score if board.turn == chess.WHITE else -score


def game_over_reason(board: chess.Board) -> Optional[str]:
    if not board.is_game_over():
        return None
    if board.is_checkmate():
        return "checkmate"
    if board.is_stalemate():
        return "stalemate"
    if board.is_insufficient_material():
        return "insufficient_material"
    if board.can_claim_fifty_moves() or board.is_fifty_moves():
        return "fifty_move_rule"
    if board.can_claim_threefold_repetition() or board.is_repetition(3):
        return "threefold_repetition"
    if board.is_seventyfive_moves():
        return "seventy_five_move_rule"
    if board.is_fivefold_repetition():
        return "fivefold_repetition"
    return "game_over"


@functools.lru_cache(maxsize=2048)
def _perft_cached(fen: str, depth: int) -> int:
    board = chess.Board(fen)
    return perft(board, depth)


def perft(board: chess.Board, depth: int) -> int:
    """Node counter for testing move generator correctness."""
    if depth == 0:
        return 1
    nodes = 0
    for move in board.legal_moves:
        board.push(move)
        nodes += perft(board, depth - 1)
        board.pop()
    return nodes


# -- Search (very small) -----------------------------------------------------

def evaluate(board: chess.Board) -> int:
    if board.is_game_over():
        if board.is_checkmate():
            return -MATE_SCORE if board.turn else MATE_SCORE
        return 0
    return material_score(board)


def minimax(board: chess.Board, depth: int) -> Tuple[int, Optional[chess.Move]]:
    """Very small minimax without alpha-beta. Depth 2–3 is OK for demo."""
    if depth == 0 or board.is_game_over():
        return evaluate(board), None

    best_score = -10**9
    best_move = None
    for move in board.legal_moves:
        board.push(move)
        score, _ = minimax(board, depth - 1)
        board.pop()
        score = -score  # perspective switch (negamax style)
        if score > best_score:
            best_score = score
            best_move = move
    return best_score, best_move


# ----------------------------------------------------------------------------
# Decorators & Auth helpers
# ----------------------------------------------------------------------------

def require_game(func):
    @functools.wraps(func)
    def wrapper(gid: str, *args, **kwargs):
        try:
            game = GAMES.get(gid)
        except KeyError:
            return json_response({"error": "game_not_found"}, 404)
        request.game = game  # type: ignore[attr-defined]
        return func(gid, *args, **kwargs)
    return wrapper


def check_secret(game: Game, req: dict) -> Optional[Dict]:
    if game.secret is None:
        return None
    provided = req.get("secret") or request.headers.get("X-Game-Secret")
    if provided != game.secret:
        return {"error": "unauthorized", "detail": "Secret mismatch for this game."}
    return None


# ----------------------------------------------------------------------------
# Routes: Games
# ----------------------------------------------------------------------------

@app.route("/", methods=["GET"])  # Healthcheck & docs pointer
def root():
    return json_response({
        "ok": True,
        "service": "chess-api",
        "docs": {
            "create": {"method": "POST", "path": "/game"},
            "state": {"method": "GET", "path": "/game/<gid>"},
            "move": {"method": "POST", "path": "/game/<gid>/move"},
            "undo": {"method": "POST", "path": "/game/<gid>/undo"},
            "pgn": {"method": "GET", "path": "/game/<gid>/pgn"},
            "legal": {"method": "GET", "path": "/game/<gid>/legal"},
            "bot_move": {"method": "POST", "path": "/game/<gid>/bot_move"},
            "hint": {"method": "POST", "path": "/game/<gid>/hint"},
            "perft": {"method": "GET", "path": "/perft"},
        }
    })


@app.route("/game", methods=["POST"])  # Create a new game
def create_game():
    payload = request.get_json(silent=True) or {}
    fen = payload.get("fen")
    tags = payload.get("tags") or {}
    secret = payload.get("secret")  # optional shared secret per-game

    try:
        game = GAMES.create(secret=secret, tags=tags, fen=fen)
    except ValueError as e:
        return json_response({"error": "bad_fen", "detail": str(e)}, 400)

    logging.info("Created game %s", game.id)
    return json_response({"id": game.id, "secret": game.secret, "fen": game.board.fen()})


@app.route("/games", methods=["GET"])  # List ids
def list_games():
    return json_response({"games": GAMES.list_ids()})


@app.route("/game/<gid>", methods=["GET"])  # Get state
@require_game
def game_state(gid: str):
    game: Game = request.game  # type: ignore[attr-defined]
    include_legal = request.args.get("legal") == "1"
    include_history = request.args.get("history") == "1"
    return json_response(game.to_dict(include_legal=include_legal, include_history=include_history))


@app.route("/game/<gid>", methods=["DELETE"])  # Delete game
@require_game
def game_delete(gid: str):
    game: Game = request.game  # type: ignore[attr-defined]
    if game.secret is not None:
        err = check_secret(game, request.args)
        if err:
            return json_response(err, 403)
    ok = GAMES.delete(gid)
    return json_response({"deleted": ok})


@app.route("/game/<gid>/move", methods=["POST"])  # Make a move
@require_game
def game_move(gid: str):
    game: Game = request.game  # type: ignore[attr-defined]
    payload = request.get_json(silent=True) or {}

    if game.secret is not None:
        err = check_secret(game, payload)
        if err:
            return json_response(err, 403)

    move_str = (payload.get("move") or "").strip()
    if not move_str:
        return json_response({"error": "missing_move"}, 400)

    ok, msg = game.apply_move(move_str)
    if not ok:
        return json_response({"error": "illegal_move", "detail": msg}, 400)

    return json_response({"ok": True, "san": msg, "fen": game.board.fen(), "turn": 'w' if game.board.turn else 'b'})


@app.route("/game/<gid>/undo", methods=["POST"])  # Undo last move
@require_game
def game_undo(gid: str):
    game: Game = request.game  # type: ignore[attr-defined]
    payload = request.get_json(silent=True) or {}

    if game.secret is not None:
        err = check_secret(game, payload)
        if err:
            return json_response(err, 403)

    if not game.undo():
        return json_response({"error": "nothing_to_undo"}, 400)
    return json_response({"ok": True, "fen": game.board.fen()})


@app.route("/game/<gid>/pgn", methods=["GET"])  # Export PGN
@require_game
def game_pgn(gid: str):
    game: Game = request.game  # type: ignore[attr-defined]
    text = game.to_pgn_string()
    return json_response({"pgn": text})


@app.route("/game/<gid>/legal", methods=["GET"])  # List legal moves
@require_game
def game_legal(gid: str):
    game: Game = request.game  # type: ignore[attr-defined]
    uci = [m.uci() for m in game.board.legal_moves]
    san = [game.board.san(chess.Move.from_uci(u)) for u in uci]
    return json_response({"uci": uci, "san": san})


@app.route("/game/<gid>/hint", methods=["POST"])  # Suggest a move
@require_game
def game_hint(gid: str):
    game: Game = request.game  # type: ignore[attr-defined]
    payload = request.get_json(silent=True) or {}

    depth = int(payload.get("depth", 2))
    depth = max(1, min(depth, 4))  # keep it quick

    board = game.board.copy()
    score, best = minimax(board, depth)
    if best is None:
        return json_response({"hint": None, "score": score, "note": "no legal moves"})

    return json_response({
        "hint": best.uci(),
        "san": board.san(best),
        "score": score,
        "depth": depth,
    })


@app.route("/game/<gid>/bot_move", methods=["POST"])  # Make a bot move
@require_game
def game_bot_move(gid: str):
    game: Game = request.game  # type: ignore[attr-defined]
    payload = request.get_json(silent=True) or {}

    if game.secret is not None:
        err = check_secret(game, payload)
        if err:
            return json_response(err, 403)

    style = (payload.get("style") or "random").lower()

    board = game.board
    legal = list(board.legal_moves)
    if not legal:
        return json_response({"error": "no_legal_moves"}, 400)

    chosen = None
    if style == "random":
        chosen = random.choice(legal)
    elif style == "capture":
        captures = [m for m in legal if board.is_capture(m)]
        chosen = random.choice(captures) if captures else random.choice(legal)
    elif style == "minimax":
        _, chosen = minimax(board.copy(), depth=2)
        if chosen is None:
            chosen = random.choice(legal)
    else:
        return json_response({"error": "unknown_style", "detail": "Use random|capture|minimax"}, 400)

    san = board.san(chosen)
    board.push(chosen)
    game.history.append(MoveRecord(chosen.uci(), san, "", game.board.fen(), time.time()))
    return json_response({"move": chosen.uci(), "san": san, "fen": board.fen()})


# ----------------------------------------------------------------------------
# Routes: Perft & Tools
# ----------------------------------------------------------------------------

@app.route("/perft", methods=["GET"])  # Count nodes from a FEN at depth
def perft_route():
    fen = request.args.get("fen")
    depth = int(request.args.get("depth", "2"))
    if not fen:
        return json_response({"error": "missing_fen"}, 400)
    try:
        board = chess.Board(fen)
    except Exception as e:
        return json_response({"error": "bad_fen", "detail": str(e)}, 400)

    depth = max(0, min(depth, 6))  # limit for demo safety
    nodes = _perft_cached(board.fen(), depth)
    return json_response({"fen": board.fen(), "depth": depth, "nodes": nodes})


@app.route("/validate_move", methods=["GET"])  # Quick move check for a FEN
def validate_move():
    fen = request.args.get("fen")
    move_str = request.args.get("move")
    if not fen or not move_str:
        return json_response({"error": "missing_params", "need": ["fen", "move"]}, 400)
    try:
        board = chess.Board(fen)
    except Exception as e:
        return json_response({"error": "bad_fen", "detail": str(e)}, 400)

    try:
        if len(move_str) in (4, 5):
            move = chess.Move.from_uci(move_str)
        else:
            move = board.parse_san(move_str)
    except Exception:
        return json_response({"ok": False, "reason": "bad_notation"})

    ok = move in board.legal_moves
    reason = None if ok else "illegal"
    return json_response({"ok": ok, "reason": reason})


# ----------------------------------------------------------------------------
# Error handlers & OPTIONS
# ----------------------------------------------------------------------------

@app.errorhandler(404)
def not_found(e):
    return json_response({"error": "not_found"}, 404)


@app.errorhandler(405)
def not_allowed(e):
    return json_response({"error": "method_not_allowed"}, 405)


@app.route('/<path:path>', methods=['OPTIONS'])
@app.route('/', methods=['OPTIONS'])
def options(path: str = ''):
    resp = json_response({"ok": True})
    return resp


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\nStarting Chess API on http://{HOST}:{PORT} (debug={DEBUG})\n")
    app.run(host=HOST, port=PORT, debug=DEBUG)
