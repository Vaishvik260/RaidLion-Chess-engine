#!/usr/bin/env python3
"""
A fully playable terminal chess game using the `python-chess` library.

Features
- Human vs Human or Human vs Simple AI (toggle at runtime)
- Accepts UCI-style moves (e2e4) or SAN (e4, Nf3, exd5, O-O, etc.)
- Shows legal moves for a square (e.g., `moves e2`)
- Undo/redo (single-step undo via `undo`; redo via `redo` if available)
- Save/load to FEN or PGN files
- Hints: simple evaluation search (depth 2) suggests a move
- Check, checkmate, stalemate and draw detection

Dependencies
- python-chess
    pip install python-chess

Run
    python3 chess_term.py

Type `help` in the game for commands.
"""
from __future__ import annotations
import sys
import os
import time
import math
import random
from dataclasses import dataclass
from typing import Optional, List, Tuple

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
        # Basic material evaluation + mobility + simple bonuses
        if board.is_checkmate():
            return -999999 if board.turn else 999999
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return 0
        score = 0
        # Material
        for piece_type in PIECE_VALUES:
            score += len(board.pieces(piece_type, chess.WHITE)) * PIECE_VALUES[piece_type]
            score -= len(board.pieces(piece_type, chess.BLACK)) * PIECE_VALUES[piece_type]
        # Mobility
        score += len(list(board.legal_moves)) * (1 if board.turn == chess.WHITE else -1)
        # Pawn structure (very light): center control
        center = [chess.D4, chess.E4, chess.D5, chess.E5]
        for sq in center:
            piece = board.piece_at(sq)
            if piece and piece.piece_type == chess.PAWN:
                score += 10 if piece.color == chess.WHITE else -10
        # King safety (tiny): encourage castling
        if board.has_castling_rights(chess.WHITE):
            score += 5
        if board.has_castling_rights(chess.BLACK):
            score -= 5
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
                max_eval = max(max_eval, eval_)
                alpha = max(alpha, eval_)
                if beta <= alpha:
                    break
            return int(max_eval)
        else:
            min_eval = math.inf
            for move in board.legal_moves:
                board.push(move)
                eval_ = self._minimax(board, depth - 1, alpha, beta, not maximizing_white)
                board.pop()
                min_eval = min(min_eval, eval_)
                beta = min(beta, eval_)
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

def clear_screen():
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

# --------- Command Loop ---------
HELP = """
Commands:
  - Enter a move in UCI (e2e4, g1f3) or SAN (e4, Nf3, O-O, exd5, e8=Q, etc.)
  - moves <square>       : list legal moves from that square (e.g., `moves e2`)
  - undo                 : undo last move
  - redo                 : redo last undone move (if any)
  - ai white|black|off   : toggle simple AI side
  - depth <n>            : set AI search depth (default 2)
  - hint                 : suggest a move for the side to move
  - fen                  : print current FEN
  - save <file.pgn>      : save the current game to PGN
  - load <file.pgn|fen>  : load a PGN file (last game) or a FEN string
  - new                  : start a new game
  - help                 : show this help
  - exit                 : quit
"""

class Game:
    def __init__(self):
        self.board = chess.Board()
        self.history: List[chess.Move] = []
        self.redo_stack: List[chess.Move] = []
        self.settings = Settings(ai_side=None, ai_depth=2)
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
        board = game.board()
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
            return
        if self.board.turn == side and not self.board.is_game_over():
            mv = self.ai.choose_move(self.board)
            if mv:
                print(f"AI plays: {self.board.san(mv)} ({mv.uci()})")
                self.push(mv)

    def new_game(self):
        self.board.reset()
        self.history.clear()
        self.redo_stack.clear()


def main():
    game = Game()
    clear_screen()
    print("Welcome to Terminal Chess. Type 'help' for commands.\n")
    while True:
        clear_screen()
        print_status(game.board)
        if game.board.is_game_over():
            result = game.board.result()
            if game.board.is_checkmate():
                print("Checkmate!", "White" if result == '1-0' else "Black", "wins.")
            else:
                print("Game over:", result)
            cmd = input("Type 'new' to start again or 'exit' to quit: ").strip()
            if cmd == 'new':
                game.new_game()
                continue
            elif cmd == 'exit':
                break
            else:
                continue

        # If AI to move, let it move automatically
        game.maybe_ai_move()
        if game.board.turn == game.settings.ai_side:
            # After AI move, loop to redraw
            continue

        cmd = input("Move/Command > ").strip()
        if not cmd:
            continue

        if cmd in {"help", "?"}:
            print(HELP)
            input("(enter to continue)")
            continue
        if cmd == "fen":
            print(game.board.fen())
            input("(enter to continue)")
            continue
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
                input("(enter to continue)")
                continue
            print(f"AI set to: { 'White' if game.settings.ai_side is True else 'Black' if game.settings.ai_side is False else 'Off' }")
            input("(enter to continue)")
            continue
        if cmd.startswith("depth "):
            try:
                n = int(cmd.split()[1])
                game.settings.ai_depth = max(1, min(4, n))
                print(f"AI depth set to {game.settings.ai_depth}")
            except Exception:
                print("Usage: depth <1-4>")
            input("(enter to continue)")
            continue
        if cmd.startswith("moves "):
            sq = cmd.split()[1]
            moves = list_moves_from(game.board, sq)
            if moves:
                print("Legal from", sq + ":", ", ".join(moves))
            else:
                print("No legal moves from", sq)
            input("(enter to continue)")
            continue
        if cmd == "undo":
            game.undo()
            continue
        if cmd == "redo":
            game.redo()
            continue
        if cmd == "new":
            game.new_game()
            continue
        if cmd.startswith("save "):
            path = cmd.split(maxsplit=1)[1]
            try:
                game.save_pgn(path)
            except Exception as e:
                print("Save failed:", e)
                input("(enter to continue)")
            continue
        if cmd.startswith("load "):
            arg = cmd.split(maxsplit=1)[1]
            if os.path.exists(arg):
                try:
                    game.load_pgn(arg)
                except Exception as e:
                    print("Load failed:", e)
                    input("(enter to continue)")
            else:
                game.load_fen(arg)
                input("(enter to continue)")
            continue
        if cmd == "hint":
            mv = game.suggest()
            if mv:
                print(f"Hint: {game.board.san(mv)} ({mv.uci()})")
            else:
                print("No hint available.")
            input("(enter to continue)")
            continue

        # Otherwise try to interpret as a move
        move = parse_move(game.board, cmd)
        if move is None:
            print("Invalid move/command. Type 'help' for help.")
            input("(enter to continue)")
            continue
        game.push(move)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye!")
