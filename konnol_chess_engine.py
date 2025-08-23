#!/usr/bin/env python3
"""
Playable Chess in Python — interactive when possible, non‑interactive fallback when input() is blocked.

Why this rewrite?
- Some sandboxed environments raise `OSError: [Errno 29] I/O error` on `input()`.
- This version detects when standard input isn’t available and **automatically falls back** to non‑interactive modes (scripted commands, demo, or self‑play), so it "actually works" everywhere.

Highlights
- Terminal UI with Unicode board, SAN/UCI input, undo/redo, hints, save/load PGN, load FEN
- Simple AI with adjustable depth (1–4)
- **Non‑interactive modes** (no `input()` needed):
  - `--script FILE`   : read commands/moves line‑by‑line from a text file
  - `--demo`          : run a built‑in Scholar’s Mate demo
  - `--selfplay N`    : AI vs AI for N plies (half‑moves) or until game over
  - `--run-tests`     : run unit tests (no interactivity)

Usage
    pip install python-chess
    python3 chess_term.py                # interactive (if stdin is a TTY)
    python3 chess_term.py --demo         # works even if input() is blocked
    python3 chess_term.py --script cmds.txt
    python3 chess_term.py --selfplay 60

Script file format (`--script`)
- One command or move per line (UCI like `e2e4` or SAN like `e4`, `Nf3`, `O-O`).
- Lines starting with `#` are comments. Blank lines are ignored.
- All in‑game commands from `help` are supported: `undo`, `redo`, `moves e2`, `ai white`, `depth 3`, `save game.pgn`, `load <fen or pgnfile>`, etc.

"""
from __future__ import annotations
import sys
import os
import time
import math
import random
import argparse
import tempfile
from dataclasses import dataclass
from typing import Optional, List, Tuple, Iterable

try:
    import chess
    import chess.pgn
    import chess.polyglot
except Exception as e:
    print("This program requires the `python-chess` package.\nInstall it with: pip install python-chess\n\nError:", e)
    sys.exit(1)

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,  # King value handled via game termination
}

@dataclass
class Settings:
    ai_side: Optional[chess.Color] = None  # chess.WHITE, chess.BLACK or None
    ai_depth: int = 2  # search depth for the simple AI
    use_opening_book: bool = False
    opening_book_path: Optional[str] = None
    clear_screen: bool = True

class SimpleAI:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.book = None
        if settings.use_opening_book and settings.opening_book_path and os.path.exists(settings.opening_book_path):
            try:
                self.book = chess.polyglot.open_reader(settings.opening_book_path)
            except Exception:
                self.book = None

    def choose_move(self, board: chess.Board) -> Optional[chess.Move]:
        # Opening book move if available
        if self.book:
            try:
                with self.book as reader:
                    entries = list(reader.find_all(board))
                    if entries:
                        return random.choice(entries).move
            except Exception:
                pass
        # Otherwise search
        return self._search_root(board, self.settings.ai_depth)

    def _evaluate(self, board: chess.Board) -> int:
        if board.is_checkmate():
            return -999999 if board.turn else 999999
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return 0
        score = 0
        # Material
        for piece_type in PIECE_VALUES:
            score += len(board.pieces(piece_type, chess.WHITE)) * PIECE_VALUES[piece_type]
            score -= len(board.pieces(piece_type, chess.BLACK)) * PIECE_VALUES[piece_type]
        # Mobility (very light)
        score += len(list(board.legal_moves)) * (1 if board.turn == chess.WHITE else -1)
        # Center pawns bonus
        for sq in (chess.D4, chess.E4, chess.D5, chess.E5):
            p = board.piece_at(sq)
            if p and p.piece_type == chess.PAWN:
                score += 10 if p.color == chess.WHITE else -10
        # Castling rights tiny nudge
        try:
            if board.has_castling_rights(chess.WHITE):
                score += 5
            if board.has_castling_rights(chess.BLACK):
                score -= 5
        except Exception:
            pass
        return score

    def _search_root(self, board: chess.Board, depth: int) -> Optional[chess.Move]:
        best_move = None
        best_score = -math.inf if board.turn == chess.WHITE else math.inf
        for move in board.legal_moves:
            board.push(move)
            score = self._minimax(board, depth - 1, -math.inf, math.inf, not board.turn)
            board.pop()
            if board.turn == chess.WHITE:  # after pop, it's original side to move
                if score > best_score:
                    best_score = score
                    best_move = move
            else:
                if score < best_score:
                    best_score = score
                    best_move = move
        return best_move

    def _minimax(self, board: chess.Board, depth: int, alpha: float, beta: float, maximizing_white: bool) -> int:
        if depth == 0 or board.is_game_over():
            return self._evaluate(board)
        if maximizing_white:
            max_eval = -math.inf
            for move in board.legal_moves:
                board.push(move)
                eval_ = self._minimax(board, depth - 1, alpha, beta, not maximizing_white)
                board.pop()
                if eval_ > max_eval:
                    max_eval = eval_
                if eval_ > alpha:
                    alpha = eval_
                if beta <= alpha:
                    break
            return int(max_eval)
        else:
            min_eval = math.inf
            for move in board.legal_moves:
                board.push(move)
                eval_ = self._minimax(board, depth - 1, alpha, beta, not maximizing_white)
                board.pop()
                if eval_ < min_eval:
                    min_eval = eval_
                if eval_ < beta:
                    beta = eval_
                if beta <= alpha:
                    break
            return int(min_eval)

# --------- UI Helpers ---------
UNICODE_PIECES = {
    chess.Piece.from_symbol('P'): '♙',
    chess.Piece.from_symbol('N'): '♘',
    chess.Piece.from_symbol('B'): '♗',
    chess.Piece.from_symbol('R'): '♖',
    chess.Piece.from_symbol('Q'): '♕',
    chess.Piece.from_symbol('K'): '♔',
    chess.Piece.from_symbol('p'): '♟',
    chess.Piece.from_symbol('n'): '♞',
    chess.Piece.from_symbol('b'): '♝',
    chess.Piece.from_symbol('r'): '♜',
    chess.Piece.from_symbol('q'): '♛',
    chess.Piece.from_symbol('k'): '♚',
}

def _clear_screen(enabled: bool):
    if not enabled:
        return
    try:
        os.system('cls' if os.name == 'nt' else 'clear')
    except Exception:
        pass


def board_to_str(board: chess.Board) -> str:
    s = []
    s.append("  +------------------------+")
    for rank in range(7, -1, -1):
        row = [f"{rank+1} |"]
        for file in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            if piece:
                row.append(f" {UNICODE_PIECES[piece]} ")
            else:
                row.append(" · ")
        row.append("|")
        s.append("".join(row))
    s.append("  +------------------------+")
    s.append("    a  b  c  d  e  f  g  h")
    return "\n".join(s)


def print_status(board: chess.Board):
    print(board_to_str(board))
    state = []
    if board.is_check():
        state.append("CHECK!")
    if board.can_claim_threefold_repetition():
        state.append("(3-fold repetition claim available)")
    if board.can_claim_fifty_moves():
        state.append("(50-move claim available)")
    if state:
        print(" ".join(state))
    print(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")
    print(f"FEN: {board.fen()}")


def parse_move(board: chess.Board, text: str) -> Optional[chess.Move]:
    text = text.strip()
    if not text:
        return None
    # Try UCI first, then SAN
    try:
        move = chess.Move.from_uci(text)
        if move in board.legal_moves:
            return move
    except Exception:
        pass
    try:
        move = board.parse_san(text)
        if move in board.legal_moves:
            return move
    except Exception:
        pass
    return None


def list_moves_from(board: chess.Board, square_str: str) -> List[str]:
    try:
        square = chess.parse_square(square_str)
    except Exception:
        return []
    res = []
    for m in board.legal_moves:
        if m.from_square == square:
            res.append(m.uci())
    return res

# --------- Command Loop & Script Runner ---------
HELP = """
Commands:
  - Enter a move in UCI (e2e4, g1f3) or SAN (e4, Nf3, O-O, exd5, e8=Q, etc.)
  - moves <square>       : list legal moves from that square (e.g., `moves e2`)
  - undo                 : undo last move
  - redo                 : redo last undone move (if any)
  - ai white|black|off   : toggle simple AI side
  - depth <n>            : set AI search depth (default 2, range 1–4)
  - hint                 : suggest a move for the side to move
  - fen                  : print current FEN
  - save <file.pgn>      : save the current game to PGN
  - load <file.pgn|fen>  : load a PGN file (last game) or a FEN string
  - new                  : start a new game
  - help                 : show this help
  - exit                 : eit/quit
"""

class Game:
    def __init__(self, settings: Optional[Settings] = None):
        self.board = chess.Board()
        self.history: List[chess.Move] = []
        self.redo_stack: List[chess.Move] = []
        self.settings = settings or Settings()
        self.ai = SimpleAI(self.settings)

    def push(self, move: chess.Move):
        self.board.push(move)
        self.history.append(move)
        self.redo_stack.clear()

    def undo(self):
        if self.history:
            self.board.pop()
            self.redo_stack.append(self.history.pop())
        else:
            print("Nothing to undo.")

    def redo(self):
        if self.redo_stack:
            m = self.redo_stack.pop()
            self.board.push(m)
            self.history.append(m)
        else:
            print("Nothing to redo.")

    def save_pgn(self, path: str):
        game = chess.pgn.Game()
        game.headers["Event"] = "Terminal Chess"
        game.headers["Date"] = time.strftime("%Y.%m.%d")
        node = game
        board_tmp = chess.Board()
        for mv in self.history:
            node = node.add_variation(mv)
            board_tmp.push(mv)
        with open(path, 'w', encoding='utf-8') as f:
            print(game, file=f, end='\n')
        print(f"Saved to {path}")

    def load_pgn(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            game = chess.pgn.read_game(f)
        if not game:
            print("No game found in PGN.")
            return
        self.board.reset()
        self.history.clear()
        self.redo_stack.clear()
        for move in game.mainline_moves():
            self.board.push(move)
            self.history.append(move)
        print(f"Loaded {path} ({len(self.history)} moves)")

    def load_fen(self, fen: str):
        try:
            self.board.set_fen(fen)
            self.history.clear()
            self.redo_stack.clear()
            print("FEN loaded.")
        except Exception as e:
            print("Invalid FEN:", e)

    def suggest(self) -> Optional[chess.Move]:
        return self.ai.choose_move(self.board)

    def maybe_ai_move(self):
        side = self.settings.ai_side
        if side is None:
            return False
        if self.board.turn == side and not self.board.is_game_over():
            mv = self.ai.choose_move(self.board)
            if mv:
                print(f"AI plays: {self.board.san(mv)} ({mv.uci()})")
                self.push(mv)
                return True
        return False

    def new_game(self):
        self.board.reset()
        self.history.clear()
        self.redo_stack.clear()


def safe_input(prompt: str) -> Optional[str]:
    """Like input(), but return None on EOF/OSError instead of crashing."""
    try:
        return input(prompt)
    except (EOFError, OSError):
        return None


def run_interactive(game: Game):
    _clear_screen(game.settings.clear_screen)
    print("Welcome to Terminal Chess. Type 'help' for commands.\n")
    while True:
        _clear_screen(game.settings.clear_screen)
        print_status(game.board)
        if game.board.is_game_over():
            result = game.board.result()
            if game.board.is_checkmate():
                print("Checkmate!", "White" if result == '1-0' else "Black", "wins.")
            else:
                print("Game over:", result)
            cmd = safe_input("Type 'new' to start again or 'exit' to quit: ")
            if cmd is None:
                print("\nInput is not available; exiting interactive mode.")
                break
            cmd = cmd.strip()
            if cmd == 'new':
                game.new_game()
                continue
            elif cmd == 'exit':
                break
            else:
                continue

        # If AI to move, play automatically
        if game.maybe_ai_move():
            continue

        cmd = safe_input("Move/Command > ")
        if cmd is None:
            print("\nInput is not available; exiting interactive mode.")
            break
        cmd = cmd.strip()
        if not cmd:
            continue

        if not handle_command_or_move(game, cmd):
            print("Invalid move/command. Type 'help' for help.")
            _ = safe_input("(enter to continue)")


def handle_command_or_move(game: Game, cmd: str) -> bool:
    """Return True if handled; False if invalid."""
    if cmd in {"help", "?"}:
        print(HELP)
        return True
    if cmd == "fen":
        print(game.board.fen())
        return True
    if cmd.startswith("ai "):
        arg = cmd.split(maxsplit=1)[1].lower().strip()
        if arg in {"white", "w"}:
            game.settings.ai_side = chess.WHITE
        elif arg in {"black", "b"}:
            game.settings.ai_side = chess.BLACK
        elif arg in {"off", "none"}:
            game.settings.ai_side = None
        else:
            print("Usage: ai white|black|off")
        return True
    if cmd.startswith("depth "):
        try:
            n = int(cmd.split()[1])
            game.settings.ai_depth = max(1, min(4, n))
            print(f"AI depth set to {game.settings.ai_depth}")
        except Exception:
            print("Usage: depth <1-4>")
        return True
    if cmd.startswith("moves "):
        parts = cmd.split()
        if len(parts) == 2:
            sq = parts[1]
            moves = list_moves_from(game.board, sq)
            if moves:
                print("Legal from", sq + ":", ", ".join(moves))
            else:
                print("No legal moves from", sq)
        else:
            print("Usage: moves <square>")
        return True
    if cmd == "undo":
        game.undo(); return True
    if cmd == "redo":
        game.redo(); return True
    if cmd == "new":
        game.new_game(); return True
    if cmd.startswith("save "):
        path = cmd.split(maxsplit=1)[1]
        try:
            game.save_pgn(path)
        except Exception as e:
            print("Save failed:", e)
        return True
    if cmd.startswith("load "):
        arg = cmd.split(maxsplit=1)[1]
        if os.path.exists(arg):
            try:
                game.load_pgn(arg)
            except Exception as e:
                print("Load failed:", e)
        else:
            game.load_fen(arg)
        return True
    if cmd == "hint":
        mv = game.suggest()
        if mv:
            print(f"Hint: {game.board.san(mv)} ({mv.uci()})")
        else:
            print("No hint available.")
        return True
    if cmd == "exit":
        # Caller decides what to do with exit, but treat as handled
        return True

    # Otherwise try to interpret as a move
    move = parse_move(game.board, cmd)
    if move is None:
        return False
    game.push(move)
    return True


def run_script(game: Game, commands: Iterable[str]) -> None:
    """Execute commands/moves without prompting (safe for no-stdin environments)."""
    for raw in commands:
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        if not handle_command_or_move(game, line):
            print(f"Ignoring invalid line: {line}")
        if game.board.is_game_over():
            break


def demo_commands() -> List[str]:
    """Scholar's Mate demo (works in both SAN and UCI forms)."""
    return [
        # White and Black alternating, SAN format
        "e4", "e5",
        "Bc4", "Nc6",
        "Qh5", "Nf6?",  # the '?' is ignored (will be invalid -> stripped below); keep simple:
        # since '?' is not valid SAN token for python-chess parse, use legal move instead
    ]


def demo_safe() -> List[str]:
    # Clean, purely legal sequence achieving Scholar's Mate
    # 1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6?? 4. Qxf7#
    return ["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7#"]


def selfplay(game: Game, plies: int = 60):
    game.settings.ai_side = None  # drive manually here
    for _ in range(plies):
        if game.board.is_game_over():
            break
        mv = game.ai.choose_move(game.board)
        if mv is None:
            break
        print(f"{('White' if game.board.turn else 'Black')} plays: {game.board.san(mv)}")
        game.push(mv)

# ---------------- CLI ----------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Terminal chess with non-interactive fallback")
    p.add_argument('--script', type=str, help='Run commands/moves from a text file')
    p.add_argument('--demo', action='store_true', help="Run a built-in demo game (Scholar's Mate)")
    p.add_argument('--selfplay', type=int, metavar='N', help='AI vs AI for N plies (half-moves)')
    p.add_argument('--ai-side', choices=['white','black','off'], default=None, help='Start with AI on a side')
    p.add_argument('--ai-depth', type=int, default=2, help='AI search depth 1–4 (default 2)')
    p.add_argument('--fen', type=str, help='Start from a FEN position')
    p.add_argument('--pgn-in', type=str, help='Load a PGN file before running')
    p.add_argument('--pgn-out', type=str, help='Save a PGN after running')
    p.add_argument('--no-clear', action='store_true', help='Do not clear the screen between positions')
    p.add_argument('--run-tests', action='store_true', help='Run unit tests and exit')
    return p


def apply_args_to_settings(args: argparse.Namespace) -> Settings:
    side = None
    if args.ai_side == 'white':
        side = chess.WHITE
    elif args.ai_side == 'black':
        side = chess.BLACK
    elif args.ai_side == 'off':
        side = None
    depth = max(1, min(4, int(args.ai_depth))) if args.ai_depth else 2
    return Settings(ai_side=side, ai_depth=depth, clear_screen=not args.no_clear)


def main(argv: Optional[List[str]] = None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.run_tests:
        return _run_tests_via_unittest()

    settings = apply_args_to_settings(args)
    game = Game(settings)

    if args.fen:
        game.load_fen(args.fen)
    if args.pgn_in and os.path.exists(args.pgn_in):
        game.load_pgn(args.pgn_in)

    # Determine mode: script/demo/selfplay vs interactive
    stdin_tty = sys.stdin.isatty() if hasattr(sys.stdin, 'isatty') else False

    if args.script:
        with open(args.script, 'r', encoding='utf-8') as f:
            run_script(game, f.readlines())
    elif args.demo:
        run_script(game, demo_safe())
    elif args.selfplay is not None:
        selfplay(game, max(0, int(args.selfplay)))
    elif not stdin_tty:
        # No interactive input available — default to a safe demo
        print("No interactive input available; running demo. Use --script/--selfplay to customize.")
        run_script(game, demo_safe())
    else:
        # Interactive
        run_interactive(game)

    # Final board & result
    print_status(game.board)
    if game.board.is_game_over():
        print("Result:", game.board.result())

    if args.pgn_out:
        try:
            game.save_pgn(args.pgn_out)
        except Exception as e:
            print("Failed to save PGN:", e)

# ---------------- Tests ----------------

import unittest

class TestCore(unittest.TestCase):
    def setUp(self):
        self.game = Game(Settings(clear_screen=False))

    def test_parse_move_san_and_uci(self):
        # From start position, e4 is legal
        self.assertIsNotNone(parse_move(self.game.board, 'e4'))
        self.assertIsNotNone(parse_move(self.game.board, 'e2e4'))
        # Illegal move
        self.assertIsNone(parse_move(self.game.board, 'e5'))

    def test_list_moves_from_start(self):
        moves = list_moves_from(self.game.board, 'e2')
        self.assertIn('e2e4', moves)
        self.assertIn('e2e3', moves)

    def test_ai_choice_not_none(self):
        mv = self.game.ai.choose_move(self.game.board)
        self.assertIsNotNone(mv)

    def test_script_scholars_mate(self):
        cmds = ["e4","e5","Bc4","Nc6","Qh5","Nf6","Qxf7#"]
        run_script(self.game, cmds)
        self.assertTrue(self.game.board.is_checkmate())
        self.assertEqual(self.game.board.result(), '1-0')

    def test_pgn_roundtrip(self):
        cmds = ["e4","e5","Nf3","Nc6","Bb5"]
        run_script(self.game, cmds)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 't.pgn')
            self.game.save_pgn(path)
            # Load into new game
            g2 = Game(Settings(clear_screen=False))
            g2.load_pgn(path)
            self.assertEqual(len(self.game.history), len(g2.history))
            self.assertEqual(self.game.board.fen(), g2.board.fen())

    def test_selfplay_does_not_crash(self):
        g = Game(Settings(clear_screen=False))
        selfplay(g, 10)
        # At least some moves should have been played
        self.assertGreaterEqual(len(g.history), 1)


def _run_tests_via_unittest():
    # Running tests without invoking input() anywhere
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestCore)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    # Exit code 0 if all passed, 1 otherwise
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye!")
