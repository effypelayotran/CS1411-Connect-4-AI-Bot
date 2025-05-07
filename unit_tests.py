import unittest
import numpy as np
from asps.game_dag import DAGState, GameDAG
from asps.heuristic_ttt_problem import HeuristicTTTProblem, TTTState, X, O, SPACE
from asps.heuristic_connect_four import HeuristicConnectFourProblem, ConnectFourState
from asps.ttt_problem import TTTProblem
from adversarial_search import alpha_beta, minimax

class IOTest(unittest.TestCase):
    """
    Tests IO for adversarial search implementations.
    Contains basic/trivial test cases.

    Each test function instantiates an adversarial search problem (DAG) and tests
    that the algorithm returns a valid action.

    It does NOT test whether the action is the "correct" action to take.
    """

    def _check_result(self, result, dag):
        """
        Tests whether the result is one of the possible actions of the dag.
        Input:
            result - the return value of an adversarial search problem.
                     This should be an action.
            dag - the GameDAG that was used to test the algorithm.
        """
        self.assertIsNotNone(result, "Output should not be None")
        start_state = dag.get_start_state()
        potential_actions = dag.get_available_actions(start_state)
        self.assertIn(result, potential_actions, "Output should be an available action")

    def test_minimax(self):
        """
        Test minimax on a basic GameDAG.
        """
        X = True
        _ = False
        matrix = [
            [_, X, X, _, _, _, _],
            [_, _, _, X, X, _, _],
            [_, _, _, _, _, X, X],
            [_, _, _, _, _, _, _],
            [_, _, _, _, _, _, _],
            [_, _, _, _, _, _, _],
            [_, _, _, _, _, _, _],
        ]
        start_state = DAGState(0, 0)
        terminal_evaluations = {3: -1, 4: -2, 5: -3, 6: -4}

        dag = GameDAG(matrix, start_state, terminal_evaluations)
        result, stats = minimax(dag)
        self._check_result(result, dag)

    def test_alpha_beta(self):
        """
        Test alpha-beta pruning on a basic GameDAG.
        """
        X = True
        _ = False
        matrix = [
            [_, X, X, _, _, _, _],
            [_, _, _, X, X, _, _],
            [_, _, _, _, _, X, X],
            [_, _, _, _, _, _, _],
            [_, _, _, _, _, _, _],
            [_, _, _, _, _, _, _],
            [_, _, _, _, _, _, _],
        ]
        start_state = DAGState(0, 0)
        terminal_evaluations = {3: -1, 4: -2, 5: -3, 6: -4}

        dag = GameDAG(matrix, start_state, terminal_evaluations)
        result, _ = alpha_beta(dag)
        self._check_result(result, dag)


class BestActionTest(unittest.TestCase):
    """
    Tests correctedness of minimax and aB-pruning implementations by checking for optimal actions.
    """

    def test_minimax_optimal_action(self):
        """
        Test minimax on a game tree (DAG) where the optimal action is known.
        """
        # simple GameDAG 
        matrix = [
            [False, True, True],  # root has two actions to nodes 1 or 2
            [False, False, False],  # node 1/action A is a terminal state with value 1
            [False, False, False],  # node 2/action B is a terminal state with value -1
        ]
        start_state = DAGState(0, 0)
        terminal_values = {1: 1, 2: -1}
        dag = GameDAG(matrix, start_state, terminal_values)
        result, _ = minimax(dag)
        opt_action = 1  # index of action to node 1 = 1, which is max terminal value
        self.assertEqual(result, opt_action, "Minimax DAG - not optimal")

    def test_alpha_beta_optimal_action(self):
        """
        Test alpha-beta pruning on the same simple game tree (DAG) where the optimal action is known.
        """
        matrix = [
            [False, True, True],
            [False, False, False],
            [False, False, False],
        ]
        start_state = DAGState(0, 0)
        terminal_evaluations = {1: 1, 2: -1}
        dag = GameDAG(matrix, start_state, terminal_evaluations)
        result, _ = alpha_beta(dag)
        opt_action = 1  
        self.assertEqual(result, opt_action, "Alpha-beta DAG - not optimal")

    def test_minimax_ttt_optimal_action(self):
        """
        Test minimax for optimal action for Player 1 (win) on Tic-Tac-Toe.
        """
        board = [
            [X, O, X],
            [O, X, SPACE],
            [O, SPACE, SPACE],
        ]
        # X's
        h_ttt_problem = HeuristicTTTProblem(dim=3, board=board,player_to_move=0)
        result, _ = minimax(h_ttt_problem)
        winning_action = (2, 2)
        self.assertEqual(result, winning_action, "Minimax TTT - not optimal")
    
    def test_alpha_beta_ttt_optimal_action(self):
        """
        Test alpha-beta pruning for optimal action for Player 1 (win) on Tic-Tac-Toe 
        """
        board = [
            [X, O, X],
            [O, X, SPACE],
            [O, SPACE, SPACE],
        ]
        # X's turn
        h_ttt_problem = HeuristicTTTProblem(dim=3, board=board,player_to_move=0)
        result, _ = alpha_beta(h_ttt_problem)
        winning_action = (2, 2)
        self.assertEqual(result, winning_action, "Alpha-beta TTT - not optimal")

    def test_minimax_connect_four_optimal_action(self):
        """
        Test minimax for optimal action for Player 2 on Connect 4.
        """
        board = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [2, 2, 1, 2, 2, 0, 0],
        ])

        # 2's turn 
        c4_problem = HeuristicConnectFourProblem(board=board, player_to_move=1)
        result, _ = minimax(c4_problem, cutoff_depth=4)
        opt_action = 5
        self.assertEqual(result, opt_action, "Minimax Connect 4 - not optimal")

    def test_alpha_beta_connect_four_optimal_action(self):
        """
        Test alpha-beta pruning for optimal action for Player 2 on Connect 4.
        """
        board = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [2, 2, 1, 2, 2, 0, 0],
        ])
        # 2's turn
        c4_problem = HeuristicConnectFourProblem(board=board, player_to_move=1)
        result, _ = alpha_beta(c4_problem, cutoff_depth=4)
        opt_action = 5
        self.assertEqual(result, opt_action, "Alpha-beta Connect 4 - not optimal")
    
    def test_TTT_4x4_heur_and_minimax_optimal(self):
        """
        Test minimax for optimal action on a 4x4 Tic-Tac-Toe game, for Player 2,
         ensuring the TTT heuristic works for larger TTT board dims.
        """
        board = [
            [X, O, X, O],
            [O, X, O, X],
            [X, SPACE, O, X],  
            [O, X, SPACE, SPACE],
        ]
        h_ttt_problem = HeuristicTTTProblem(dim=4, board=board, player_to_move=1)
        result, _ = minimax(h_ttt_problem)
        winning_action = (2, 1) 
        self.assertEqual(result, winning_action, "Minimax TTT 4x4 - not optimal")
    
    def test_TTT_4x4_heur_and_alpha_beta_optimal(self):
        """
        Test alpha-beta pruning for optimal action on a 4x4 Tic-Tac-Toe board, for Player 2,
         ensuring that the TTT heuristic works for larger TTT board dims.
        """
        board = [
            [X, O, X, O],
            [O, X, O, X],
            [X, SPACE, O, X],  
            [O, X, SPACE, SPACE],
        ]
        h_ttt_problem = HeuristicTTTProblem(dim=4, board=board, player_to_move=1)
        result, _ = alpha_beta(h_ttt_problem)
        winning_action = (2, 1)  
        self.assertEqual(result, winning_action, "Alpha-beta TTT 4x4 - not optimal")
    
    def test_TTT_5x5_heur_and_minimax_optimal(self):
        """
        Test minimax for optimal action on a 5x5 Tic-Tac-Toe board, for Player 1,
         ensuring the heuristic works for larger board dims.
        """
        board = [
            [X, O, X, O, X],
            [O, X, O, X, O],
            [X, O, X, O, X],
            [O, X, O, X, O],
            [SPACE, SPACE, SPACE, SPACE, SPACE],
        ]
        h_ttt_problem = HeuristicTTTProblem(dim=5, board=board, player_to_move=0)
        result, _ = minimax(h_ttt_problem)
        winning_action = (4, 4)  
        self.assertEqual(result, winning_action, "Minimax TTT 5x5 - not optimal")

    def test_TTT_5x5_heur_and_alpha_beta_optimal(self):
        """
        Test alpha-beta pruning for optimal action on a 5x5 Tic-Tac-Toe board, for Player 1,
         ensuring that the TTT heuristic works on different board sizes.
        """
        board = [
            [X, O, X, O, X],
            [O, X, O, X, O],
            [X, O, X, O, X],
            [O, X, O, X, O],
            [SPACE, SPACE, SPACE, SPACE, SPACE],
        ]
        h_ttt_problem = HeuristicTTTProblem(dim=5, board=board, player_to_move=0)
        result, _ = alpha_beta(h_ttt_problem)
        winning_action = (4, 4)  
        self.assertEqual(result, winning_action, "Alpha-beta TTT 5x5 - not optimal")


if __name__ == "__main__":
    unittest.main()