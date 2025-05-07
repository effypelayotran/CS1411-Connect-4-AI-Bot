from heuristic_adversarial_search_problem import HeuristicAdversarialSearchProblem
from .ttt_problem import TTTState, TTTProblem

SPACE = " "
X = "X"  # Player 0 is X
O = "O"  # Player 1 is 0
PLAYER_SYMBOLS = [X, O]

class HeuristicTTTProblem(TTTProblem, HeuristicAdversarialSearchProblem):
    def heuristic(self, state: TTTState) -> float:
        """
        TODO: Fill this out with your own heuristic function! You should make sure that this
        function works with boards of any size; if it only works for 3x3 boards, you won't be
        able to properly test ab-cutoff for larger board sizes!
        """
        board = state.board
        player = self.get_start_state().player_to_move()
        if player == 0:
            opponent = 1
        else: 
            opponent = 0
        size = len(board)
        value = 0

        # congregate all lines in the board (rows, cols, and diags) to check for wins later
        poss_wins = [] # list[list], where each inner list is a line in the board
        for i in range(size):
            # rows
            poss_wins.append([board[i][j] for j in range(size)])  
            # cols
            poss_wins.append([board[j][i] for j in range(size)])  
        # diags
        poss_wins.append([board[i][i] for i in range(size)])  
        poss_wins.append([board[i][size - 1 - i] for i in range(size)])  


        # for each row, col, and diag, check if it is a win.
        for line in poss_wins:
            if SPACE not in line:
                continue
            # if the line has only X's and no O's, add 1 to the value
            # if the line has only O's and no X's add -1 to the value
            if line.count(PLAYER_SYMBOLS[player]) > 0 and line.count(PLAYER_SYMBOLS[opponent]) == 0:
                value += 1  
            elif line.count(PLAYER_SYMBOLS[opponent]) > 0 and line.count(PLAYER_SYMBOLS[player]) == 0:
                value -= 1  
            # otherwise, don't add to the value
        return value