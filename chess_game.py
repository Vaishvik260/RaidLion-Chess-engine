
# chess_game.py
# Self-contained terminal chess game (no external libraries).
# Features: legal moves, check/checkmate, stalemate, castling, en passant,
# promotion, 50-move rule, and threefold repetition. Two-player (human vs human).
# Enter moves like e2e4, g7g8q (promotion to queen), castle: e1g1 or e1c1 etc.

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

FILES = 'abcdefgh'
RANKS = '12345678'

def in_bounds(r, c): return 0 <= r < 8 and 0 <= c < 8

def coord_to_rc(coord: str) -> Tuple[int,int]:
    file, rank = coord[0], coord[1]
    c = FILES.index(file)
    r = 8 - int(rank)
    return r, c

def rc_to_coord(r: int, c: int) -> str:
    return f"{FILES[c]}{8-r}"

def piece_color(p: str) -> Optional[str]:
    if not p: return None
    return 'w' if p.isupper() else 'b'

def opposite(color: str) -> str:
    return 'b' if color == 'w' else 'w'

def is_slider(p: str) -> bool:
    return p.upper() in ('B','R','Q')

@dataclass
class State:
    board: List[List[str]]
    side: str
    castling: Dict[str,bool]
    ep: Optional[Tuple[int,int]]
    halfmove: int
    move_no: int
    history: List[str]  # FEN-like positions for threefold

def start_position() -> State:
    s = State(
        board=[
            list('rnbqkbnr'),
            list('pppppppp'),
            ['']*8,
            ['']*8,
            ['']*8,
            ['']*8,
            list('PPPPPPPP'),
            list('RNBQKBNR'),
        ],
        side='w',
        castling={'K':True,'Q':True,'k':True,'q':True},
        ep=None,
        halfmove=0,
        move_no=1,
        history=[]
    )
    s.history.append(to_fen(s, include_clocks=False))
    return s

def clone_state(s: State) -> State:
    return State(
        board=[row[:] for row in s.board],
        side=s.side,
        castling=s.castling.copy(),
        ep=s.ep if s.ep is None else (s.ep[0], s.ep[1]),
        halfmove=s.halfmove,
        move_no=s.move_no,
        history=s.history[:]  # shallow is fine for strings
    )

def to_fen(s: State, include_clocks=True) -> str:
    # board
    parts = []
    for r in range(8):
        cnt = 0
        line = ''
        for c in range(8):
            p = s.board[r][c]
            if p == '':
                cnt += 1
            else:
                if cnt:
                    line += str(cnt); cnt = 0
                line += p
        if cnt: line += str(cnt)
        parts.append(line)
    board_part = '/'.join(parts)
    # castling
    castle = ''.join(k for k in 'KQkq' if s.castling[k])
    castle = castle if castle else '-'
    # ep
    ep = rc_to_coord(*s.ep) if s.ep else '-'
    if include_clocks:
        return f"{board_part} {s.side} {castle} {ep} {s.halfmove} {s.move_no}"
    else:
        return f"{board_part} {s.side} {castle} {ep}"

# Move generation utilities
KNIGHT_DIRS = [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]
KING_DIRS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
BISHOP_DIRS = [(-1,-1),(-1,1),(1,-1),(1,1)]
ROOK_DIRS = [(-1,0),(1,0),(0,-1),(0,1)]

def king_pos(s: State, color: str) -> Tuple[int,int]:
    for r in range(8):
        for c in range(8):
            p = s.board[r][c]
            if p == ('K' if color=='w' else 'k'):
                return (r,c)
    raise RuntimeError('King not found')

def attacks_square(s: State, r: int, c: int, by_color: str) -> bool:
    # Pawns
    dir = -1 if by_color == 'w' else 1
    pr = r + dir
    for dc in (-1, 1):
        pc = c + dc
        if in_bounds(pr, pc):
            p = s.board[pr][pc]
            if p == ('P' if by_color=='w' else 'p'):
                return True
    # Knights
    for dr, dc in KNIGHT_DIRS:
        rr, cc = r+dr, c+dc
        if in_bounds(rr,cc) and s.board[rr][cc] == ('N' if by_color=='w' else 'n'):
            return True
    # King
    for dr, dc in KING_DIRS:
        rr, cc = r+dr, c+dc
        if in_bounds(rr,cc) and s.board[rr][cc] == ('K' if by_color=='w' else 'k'):
            return True
    # Sliders
    # bishops/queens
    for dr, dc in BISHOP_DIRS:
        rr, cc = r+dr, c+dc
        while in_bounds(rr,cc):
            p = s.board[rr][cc]
            if p != '':
                if p.upper() in ('B','Q') and piece_color(p)==by_color:
                    return True
                break
            rr += dr; cc += dc
    # rooks/queens
    for dr, dc in ROOK_DIRS:
        rr, cc = r+dr, c+dc
        while in_bounds(rr,cc):
            p = s.board[rr][cc]
            if p != '':
                if p.upper() in ('R','Q') and piece_color(p)==by_color:
                    return True
                break
            rr += dr; cc += dc
    return False

def in_check(s: State, color: str) -> bool:
    kr, kc = king_pos(s, color)
    return attacks_square(s, kr, kc, opposite(color))

@dataclass
class Move:
    fr: Tuple[int,int]
    to: Tuple[int,int]
    promo: Optional[str] = None  # 'q','r','b','n' (lowercase for storing; we'll map case later)
    is_castle: bool = False
    is_ep: bool = False

def gen_pseudo_legal(s: State) -> List[Move]:
    color = s.side
    moves: List[Move] = []
    for r in range(8):
        for c in range(8):
            p = s.board[r][c]
            if p == '' or piece_color(p) != color: continue
            u = p.upper()
            if u == 'P':
                dir = -1 if color=='w' else 1
                start_rank = 6 if color=='w' else 1
                promo_rank = 0 if color=='w' else 7
                # forward 1
                rr, cc = r+dir, c
                if in_bounds(rr,cc) and s.board[rr][cc]=='':
                    if rr == promo_rank:
                        for pr in 'qrbn':
                            moves.append(Move((r,c),(rr,cc),promo=pr))
                    else:
                        moves.append(Move((r,c),(rr,cc)))
                    # forward 2
                    if r==start_rank:
                        rr2 = r+2*dir
                        if s.board[rr2][cc]=='':
                            moves.append(Move((r,c),(rr2,cc)))
                # captures
                for dc in (-1,1):
                    rr, cc = r+dir, c+dc
                    if in_bounds(rr,cc):
                        if s.board[rr][cc] != '' and piece_color(s.board[rr][cc])==opposite(color):
                            if rr == promo_rank:
                                for pr in 'qrbn':
                                    moves.append(Move((r,c),(rr,cc),promo=pr))
                            else:
                                moves.append(Move((r,c),(rr,cc)))
                # en passant
                if s.ep:
                    epr, epc = s.ep
                    if r+dir == epr and abs(c-epc)==1:
                        moves.append(Move((r,c),(epr,epc),is_ep=True))
            elif u == 'N':
                for dr, dc in KNIGHT_DIRS:
                    rr, cc = r+dr, c+dc
                    if in_bounds(rr,cc):
                        t = s.board[rr][cc]
                        if t=='' or piece_color(t)==opposite(color):
                            moves.append(Move((r,c),(rr,cc)))
            elif u == 'B' or u=='R' or u=='Q':
                dirs = BISHOP_DIRS if u=='B' else ROOK_DIRS if u=='R' else BISHOP_DIRS+ROOK_DIRS
                for dr, dc in dirs:
                    rr, cc = r+dr, c+dc
                    while in_bounds(rr,cc):
                        t = s.board[rr][cc]
                        if t=='':
                            moves.append(Move((r,c),(rr,cc)))
                        else:
                            if piece_color(t)==opposite(color):
                                moves.append(Move((r,c),(rr,cc)))
                            break
                        rr += dr; cc += dc
            elif u == 'K':
                for dr, dc in KING_DIRS:
                    rr, cc = r+dr, c+dc
                    if in_bounds(rr,cc):
                        t = s.board[rr][cc]
                        if t=='' or piece_color(t)==opposite(color):
                            moves.append(Move((r,c),(rr,cc)))
                # castling
                if color=='w' and r==7 and c==4:
                    if s.castling['K'] and s.board[7][5]=='' and s.board[7][6]==''                            and not attacks_square(s,7,4,'b') and not attacks_square(s,7,5,'b') and not attacks_square(s,7,6,'b'):
                        moves.append(Move((7,4),(7,6),is_castle=True))
                    if s.castling['Q'] and s.board[7][3]=='' and s.board[7][2]=='' and s.board[7][1]==''                            and not attacks_square(s,7,4,'b') and not attacks_square(s,7,3,'b') and not attacks_square(s,7,2,'b'):
                        moves.append(Move((7,4),(7,2),is_castle=True))
                if color=='b' and r==0 and c==4:
                    if s.castling['k'] and s.board[0][5]=='' and s.board[0][6]==''                            and not attacks_square(s,0,4,'w') and not attacks_square(s,0,5,'w') and not attacks_square(s,0,6,'w'):
                        moves.append(Move((0,4),(0,6),is_castle=True))
                    if s.castling['q'] and s.board[0][3]=='' and s.board[0][2]=='' and s.board[0][1]==''                            and not attacks_square(s,0,4,'w') and not attacks_square(s,0,3,'w') and not attacks_square(s,0,2,'w'):
                        moves.append(Move((0,4),(0,2),is_castle=True))
    return moves

def make_move(s: State, m: Move) -> State:
    ns = clone_state(s)
    fr, to = m.fr, m.to
    frp = ns.board[fr[0]][fr[1]]
    top = ns.board[to[0]][to[1]]
    ns.ep = None
    # halfmove clock
    if frp.upper()=='P' or top!='':
        ns.halfmove = 0
    else:
        ns.halfmove += 1

    # move piece
    ns.board[to[0]][to[1]] = frp
    ns.board[fr[0]][fr[1]] = ''

    # en passant capture
    if m.is_ep:
        cap_r = to[0] + (1 if s.side=='w' else -1)
        cap_c = to[1]
        ns.board[cap_r][cap_c] = ''

    # promotion
    if m.promo:
        ns.board[to[0]][to[1]] = (m.promo.upper() if s.side=='w' else m.promo.lower())

    # castling rook move
    if m.is_castle:
        if s.side=='w':
            if to==(7,6):  # king side
                ns.board[7][5] = 'R'; ns.board[7][7] = ''
            else:          # queen side
                ns.board[7][3] = 'R'; ns.board[7][0] = ''
        else:
            if to==(0,6):
                ns.board[0][5] = 'r'; ns.board[0][7] = ''
            else:
                ns.board[0][3] = 'r'; ns.board[0][0] = ''

    # set ep square if double pawn push
    if frp.upper()=='P' and abs(to[0]-fr[0])==2:
        mid_r = (to[0]+fr[0])//2
        ns.ep = (mid_r, fr[1])

    # update castling rights
    # If king moves
    if frp=='K':
        ns.castling['K']=False; ns.castling['Q']=False
    if frp=='k':
        ns.castling['k']=False; ns.castling['q']=False
    # If rooks move or are captured
    if fr==(7,0) or to==(7,0): ns.castling['Q']=False
    if fr==(7,7) or to==(7,7): ns.castling['K']=False
    if fr==(0,0) or to==(0,0): ns.castling['q']=False
    if fr==(0,7) or to==(0,7): ns.castling['k']=False

    # switch side and move number
    ns.side = opposite(s.side)
    if ns.side=='w':
        ns.move_no += 1

    ns.history.append(to_fen(ns, include_clocks=False))
    return ns

def legal_moves(s: State) -> List[Move]:
    candidates = gen_pseudo_legal(s)
    legal = []
    for m in candidates:
        ns = make_move(s, m)
        if not in_check(ns, opposite(ns.side)):  # after move, make sure mover's king isn't in check
            legal.append(m)
    return legal

def parse_move(txt: str, s: State) -> Optional[Move]:
    txt = txt.strip().lower()
    if len(txt) < 4: return None
    try:
        fr = coord_to_rc(txt[0:2])
        to = coord_to_rc(txt[2:4])
    except Exception:
        return None
    promo = None
    if len(txt) == 5 and txt[4] in 'qrbn':
        promo = txt[4]
    # find matching move
    for m in legal_moves(s):
        if m.fr == fr and m.to == to:
            if (promo or m.promo) and (promo != m.promo):
                continue
            if m.promo and not promo:
                m = Move(m.fr, m.to, promo='q', is_castle=m.is_castle, is_ep=m.is_ep)
            return m
    return None

def is_threefold(s: State) -> bool:
    current = to_fen(s, include_clocks=False)
    return s.history.count(current) >= 3

def insufficient_material(s: State) -> bool:
    pieces = []
    colors = []
    for r in range(8):
        for c in range(8):
            p = s.board[r][c]
            if p=='': continue
            pieces.append(p.upper())
            if p.upper()=='B':
                colors.append((r+c)%2)
    if pieces == ['K','K']: return True
    if sorted(pieces) in (['K','K','B'], ['K','K','N']): return True
    if all(p in ('K','B') for p in pieces) and len(set(colors))==1:
        return True
    return False

def print_board(s: State):
    print('  +-----------------+')
    for r in range(8):
        row = s.board[r]
        print(8-r, '|', end=' ')
        for c in range(8):
            ch = row[c] if row[c] else '.'
            print(ch, end=' ')
        print('|')
    print('  +-----------------+')
    print('    a b c d e f g h')
    print(f"Side to move: {'White' if s.side=='w' else 'Black'}  |  FEN: {to_fen(s)}\n")

def game_loop():
    s = start_position()
    while True:
        print_board(s)
        lm = legal_moves(s)
        if in_check(s, s.side):
            if not lm:
                print('Checkmate!', 'White' if s.side=='b' else 'Black', 'wins.')
                break
            else:
                print('Check!')
        else:
            if not lm:
                print('Stalemate! Draw.')
                break
        if s.halfmove >= 100:
            print('Draw by 50-move rule.')
            break
        if is_threefold(s):
            print('Draw by threefold repetition.')
            break
        if insufficient_material(s):
            print('Draw by insufficient material.')
            break
        mv = input(f"Enter move for {'White' if s.side=='w' else 'Black'} (e.g., e2e4, e7e8q): ").strip()
        if mv.lower() in ('quit','exit'):
            print('Game ended by user.'); break
        m = parse_move(mv, s)
        if not m:
            print('Invalid or illegal move. Try again.')
            continue
        s = make_move(s, m)

if __name__ == '__main__':
    game_loop()
