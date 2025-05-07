import random
from typing import Dict, Tuple

from adversarial_search_problem import (
    Action,
    State as GameState,
)
from heuristic_adversarial_search_problem import HeuristicAdversarialSearchProblem


def minimax(asp: HeuristicAdversarialSearchProblem[GameState, Action], cutoff_depth=float('inf')) -> Tuple[Action, Dict[str, int]]:
    """
    Implement the minimax algorithm on ASPs, assuming that the given game is
    both 2-player and zero-sum.

    Input:
        asp - a HeuristicAdversarialSearchProblem
        cutoff_depth - the maximum search depth, where 0 is the start state. 
                    Depth 1 is all the states reached after a single action from the start state (1 ply).
                    cutoff_depth will always be greater than 0.
    Output:
        an action (an element of asp.get_available_actions(asp.get_start_state()))
        a dictionary of statistics for visualization
            states_expanded: stores the number of states expanded during current search
                            A state is expanded when get_available_actions(state) is called.

    """
    # TODO: Implement the minimax algorithm. Feel free to write helper functions.
    best_action = None
    stats = {
        'states_expanded': 0
    }

    def max_value(state, depth):
        """
        Recursively compute the maximum value a maximizer player can achieve from the 
        current state using the minimax algorithm. Max_value will evaluate the value 
        of next states and bubble up the best (largest) possible value, simulating the 
        maximizer's optimal play.

        Input:
            state - the current game state
            depth - the current depth level of the state 
        
        Output:
            The maximum value the maximizer player can achieve from this state,
            after propagating results from next states. Value is calculated from
            asp's heuristic function.
        """
        
        '''STEP 1: Check if we are at a "special" cases and return if so. 
        Are you at a TERMINAL state or at the CUTOFF DEPTH?'''
        # check if we are currently at a WIN/LOSS/TIED state
        if asp.is_terminal_state(state):
            # asp's get_reward will take in a TERMINAL state 
            # and return who won/lost/tied as a tuple of (p1 value, p2 value)
            return asp.get_reward(state) 
        # if not, check if we are at bottom depth & grab estimate_values
        elif depth == cutoff_depth:
            # asp's heuristic funciton calculates the board value
            # similarly to CS17's estimate_value in project Game!
            return asp.heuristic(state)

        '''STEP 2: If not a special case, calculate the value at this state 
        by propogating up lower state's values'''
        # get a list of all possible next moves
        possible_moves = asp.get_available_actions(state)
        stats['states_expanded'] += 1  
        value = float('-inf') # initialize as low as possible
        # for each possible next move, calculate the board value by bubbling up
        for next_move in possible_moves:
            next_state = asp.transition(state, next_move)
            next_depth = depth + 1
            # update the value to always be the largest value among the possible moves
            value = max(value, min_value(next_state, next_depth))
        return value

  
    def min_value(state, depth):
        """
        Recursively compute the minimum value a minimizer player can achieve from the 
        current state using the minimax algorithm. Min_value evaluates the value 
        of next states and bubbles up the lowest possible value, simulating 
        the minimizer's optimal play.

        Input:
            state - the current state
            depth - the current depth level of the state 
        
        Output:
            The minimum value the minimizer player can achieve from this state,
            after propagating results from next states. Value is calculated from
            asp's heuristic function.
        """

        '''STEP 1: Check if we are at a "special" case and return if so. 
        Are you at a TERMINAL state or at the CUTOFF DEPTH?'''
        # check if we are currently at a WIN/LOSS/TIED state
        if asp.is_terminal_state(state):
            return asp.get_reward(state)
        # if not, check if we are at bottom depth & grab estimate_values
        elif depth == cutoff_depth:
            # asp's heuristic funciton calculates the board value
            # similarly to CS17's estimate_value in project Game!
            return asp.heuristic(state)
        
        '''STEP 2: If not a special case, calculate the value at this state 
        by propograting up lower state's values'''
        possible_moves = asp.get_available_actions(state)
        stats['states_expanded'] += 1  
        value = float('inf') # initialize to be as high as possible
        # for each possible next move, calculate the board value by pruning up
        for next_move in possible_moves:
            next_state = asp.transition(state, next_move)
            # update the value to always be the smallest value among the possible moves
            next_depth = depth + 1
            value = min(value, max_value(next_state, next_depth))
        return value

    '''START the min-max algorithm, which will switch off between min_value and max_value calls.'''
    # get start state & start player
    start_state = asp.get_start_state()
    player = start_state.player_to_move()
    if player == 0: 
        curr_best_value = float('-inf')
    else:
        curr_best_value = float('inf')

    # get a list of all possible next moves
    possible_moves = asp.get_available_actions(start_state)
    stats['states_expanded'] += 1  
    
    # (for testing purpose)
    if not possible_moves:
        print("MINIMAX: INVALID BOARD, No available actions in this test state.")
        return None, stats
    
    # for each possible next move, calculate the board value by propogating up next states' values
    for next_move in possible_moves:
        next_state = asp.transition(start_state, next_move)
        if player == 0:  
            value = min_value(next_state, 1)
            # update best_action to return if its value is better than current value
            if value > curr_best_value:
                curr_best_value = value
                best_action = next_move
        else:  
            value = max_value(next_state, 1)
            # update best_action to return if its value is better than current value
            if value < curr_best_value:
                curr_best_value = value
                best_action = next_move
    return best_action, stats


def alpha_beta(asp: HeuristicAdversarialSearchProblem[GameState, Action], cutoff_depth=float('inf')) -> Tuple[Action, Dict[str, int]]:
    """
    Implement the alpha-beta pruning algorithm on ASPs,
    assuming that the given game is both 2-player and constant-sum.

    Input:
        asp - an AdversarialSearchProblem
        cutoff_depth - the maximum search depth, where 0 is the start state,
                    Depth 1 is all the states reached after a single action from the start state (1 ply).
                    cutoff_depth will always be greater than 0.
    Output:
        an action (an element of asp.get_available_actions(asp.get_start_state()))
         a dictionary of statistics for visualization
            states_expanded: stores the number of states expanded during current search
                            A state is expanded when get_available_actions(state) is called.
    """
    best_action = None
    stats = {
        'states_expanded': 0  # Increase by 1 for every state transition
    }

    # TODO: Implement the alpha-beta pruning algorithm. Feel free to use helper functions.

   
    def max_value(state, alpha, beta, depth):
        """
        Recursively compute the maximum value a maximizer player can achieve from the 
        current state in alpha-beta pruning. Max_value evaluates value of lower states and 
        bubbling up the best possible value while pruning (getting rid of) subtrees 
        where the minimizer would not explore (alpha >= beta).

        Input:
            state - the current state
            alpha - the maximum board value (the best value the maximizer can obtain thus far)
            beta - the minimum board value (the best value the minimizer can obtain thus far)
            depth - the current depth level of the state
        Output:
            The maximum board value the maximizer can achieve from this state,
            after propagating results from next states and applying pruning 
            to efficiently not look down branches we know won't improve the current 
            best option for the maximizer (i.e., when alpha >= beta)
        """

        '''STEP 1: Check if we are at a 'special' cases and return if so.
        Are we at a TERMINAL state or at a CUTOFF DEPTH?'''
        if asp.is_terminal_state(state):
            return asp.get_reward(state)
        elif depth == cutoff_depth:
            return asp.heuristic(state)
        

        '''STEP 2: Otherwise, calculate the value at this state by propogating up and pruning.'''
        value = float('-inf')  # initialize value as lowest possible value
        possible_moves = asp.get_available_actions(state)
        stats['states_expanded'] += 1
        # for each possible next move, calculate its value
        for next_move in possible_moves:
            next_state = asp.transition(state, next_move)
            next_depth = depth + 1
            # update the value to always the largest among the possible next moves
            value = max(value, min_value(next_state, alpha, beta, next_depth))

            '''STEP 3: The Alpha-Beta Pruning part that makes this function
            more efficient than minimax. If the calculated value is greater than beta,
            we can stop looking at the 'adjacent branches' which are the other next_moves 
            in the list of possible next moves for this state, as the minimizer would
            never allow the maximer to reach this state because there is a better 
            maximum available, as informed by the higher alpha value.
            If not, keep checking the branches left to right.'''
            if value >= beta:
                return value  

            # update the alpha (holds the maximum value)
            alpha = max(alpha, value)
        return value

   
    def min_value(state, alpha, beta, depth):
        """
        Recursively compute the minimum value a minimizer player can achieve from the
        current state in alpha-beta pruning. Min_value evaluates child states and 
        'bubbles up' the lowest possible value while pruning (eliminating) 
        subtrees where the maximizer would not allow exploration (beta <= alpha).

        Input:
            state - the current state
            alpha - the maximum board value (the best value the maximizer can obtain thus far)
            beta - the minimum board value (the best value the minimizer can obtain thus far)
            depth - the current depth level of the state
        Output:
            The minimum utility value the minimizer can achieve from this state,
            after propagating results from child states and applying pruning 
            where applicable.
        """
    
        '''STEP 1: Check if we are at a 'special' cases and return if so.
        Are we at a TERMINAL state or at a CUTOFF DEPTH?'''
        if asp.is_terminal_state(state):
            return asp.get_reward(state)
        elif depth == cutoff_depth:
            return asp.heuristic(state)
        
        '''STEP 2: Otherwise, compute the value at this state by propogating up and pruning.'''
        value = float('inf') # initialize value as highest possible value
        possible_moves = asp.get_available_actions(state)
        stats['states_expanded'] += 1
        # for each possible next move, calculate its value
        for next_move in possible_moves:
            next_state = asp.transition(state, next_move)
            next_depth = depth + 1
            # update the value to always be the smallest among the possible next moves
            value = min(value, max_value(next_state, alpha, beta, next_depth))

            '''STEP 3: The Alpha-Beta Pruning part that makes this function
            more efficient than minimax. If the calculated value is less than alpha,
            we can stop looking at the 'adjacent branches' which are the other next_moves 
            in the list of possible next moves for this state, as the minimizer would
            never allow the maximer to reach this state because there is a better 
            maximum available, as informed by the higher alpha value.
            If not, keep checking the branches left to right.'''
            if value <= alpha:
                return value  
            # update the beta (holds the lowest possible value)
            beta = min(beta, value)
        return value

    '''START the alpha-beta pruning algorithm, which will switch off between max_value
    and min_value calls for each depth level.'''
    # get the start state & start player
    start_state = asp.get_start_state()
    player = start_state.player_to_move()
    alpha = float('-inf')
    beta = float('inf')
    if player == 0: 
        curr_best_value = float('-inf')
    else:
        curr_best_value = float('inf')

    # get the list of all possible next moves from the start state
    possible_moves = asp.get_available_actions(start_state)
    stats['states_expanded'] += 1 
    # (for testing purpose)
    if not possible_moves:
        print("ALPHA-BETA: INVALID BOARD, No available actions in this test state.")
        return None, stats

    # for each possible next move, calculate their values by propogating and pruning
    for next_move in possible_moves:
        next_state = asp.transition(start_state, next_move)
        if player == 0: 
            # update the value to always be the largest among the possible next moves 
            value = min_value(next_state, alpha, beta, 1)
            # update the best_action to return if this next_move's value is better than current
            if value > curr_best_value:
                curr_best_value = value
                best_action = next_move
            # update the alpha (holds maximum-est value)
            alpha = max(alpha, curr_best_value)
        else: 
            # update the value to always be the largest among the possible next moves 
            value = max_value(next_state, alpha, beta, 1)
            # update the best_action to return if this next_move's value is better than current
            if value < curr_best_value:
                curr_best_value = value
                best_action = next_move
            # update the beta (holds minimum-est value)
            beta = min(beta, curr_best_value)

    return best_action, stats