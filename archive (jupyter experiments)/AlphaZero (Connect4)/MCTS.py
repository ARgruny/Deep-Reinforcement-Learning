### MCTS.py Taken from the Udacity Deep Reinforcement Learning Nanodegree. Slightly modified to fit the project. 
### I also added more documentation.

import torch
from copy import copy
from math import *
import random

c=1.0

# transformations
t0= lambda x: x
t1= lambda x: x[:,::-1].copy()
t2= lambda x: x[::-1,:].copy()
t3= lambda x: x[::-1,::-1].copy()
t4= lambda x: x.T
t5= lambda x: x[:,::-1].T.copy()
t6= lambda x: x[::-1,:].T.copy()
t7= lambda x: x[::-1,::-1].T.copy()

tlist=[t0, t1,t2,t3,t4,t5,t6,t7]
tlist_half=[t0,t1,t2,t3]


def flip(x, dim):
    """
    Flip a tensor along a specified dimension.

    Args:
        x (torch.Tensor): The input tensor to flip.
        dim (int): The dimension along which to flip the tensor.

    Returns:
        torch.Tensor: The flipped tensor.

    Example:
        >>> x = torch.arange(8).view(2, 4)
        >>> flip(x, dim=1)
        tensor([[3, 2, 1, 0],
                [7, 6, 5, 4]])
    """
    # Generate a list of slice(None) objects for each dimension of the input tensor.
    # This will serve as a placeholder for indices in all dimensions except the one to be flipped.
    indices = [slice(None)] * x.dim()

    # Replace the placeholder in the target dimension with a torch.arange sequence that
    # generates indices in reverse order. This effectively selects elements in reverse
    # along the specified dimension.
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, device=x.device, dtype=torch.long)
    
    # Use the tuple of indices to index into the original tensor, thereby flipping it
    # along the specified dimension.
    return x[tuple(indices)]


t0inv= lambda x: x
t1inv= lambda x: flip(x,1)
t2inv= lambda x: flip(x,0)
t3inv= lambda x: flip(flip(x,0),1)
t4inv= lambda x: x.t()
t5inv= lambda x: flip(x,0).t()
t6inv= lambda x: flip(x,1).t()
t7inv= lambda x: flip(flip(x,0),1).t()

tinvlist = [t0inv, t1inv, t2inv, t3inv, t4inv, t5inv, t6inv, t7inv]
tinvlist_half=[t0inv, t1inv, t2inv, t3inv]

transformation_list = list(zip(tlist, tinvlist))
transformation_list_half = list(zip(tlist_half, tinvlist_half))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
device ='cpu'


def process_policy(
    policy, 
    game, 
    transformation_list=transformation_list, 
    transformation_list_half=transformation_list_half
    ):
    """
    Process the game state through the policy network after applying a random transformation.
    The output probabilities are adjusted for available moves only.

    Args:
        policy (callable): A function that takes the game state as input and returns move probabilities and a value.
        game (Game): An instance of a game with a defined state, player, size, and available moves.

    Returns:
        tuple: A tuple containing the available moves, adjusted probabilities for those moves, and the game value.
    """

    # Select the appropriate transformation list based on the board size.
    transformation_list = transformation_list_half if game.size[0] != game.size[1] else transformation_list
    
    # Choose a random transformation and its inverse.
    t, tinv = random.choice(transformation_list)
    
    # Apply the transformation to the game state and prepare it as input to the policy.
    frame = torch.tensor(t(game.state * game.player), dtype=torch.float, device=device)
    frame = frame.unsqueeze(0).unsqueeze(0)
    
    #print("Frame:", frame.shape)
    # Obtain probabilities and value from the policy for the transformed game state.
    prob, v = policy(frame)
    
    # Apply the mask to filter probabilities for available moves.
    mask = torch.tensor(game.available_mask(), dtype=torch.bool, device=device)
    
    # Transform the probabilities back and filter by the available moves mask.
    adjusted_prob = tinv(prob)[mask].view(-1)
    
    # Return the available moves, adjusted probabilities, and squeezed game value.
    return game.available_moves(), adjusted_prob, v.squeeze().squeeze()


class Node:
    def __init__(self, game, mother=None, prob=torch.tensor(0., dtype=torch.float)):
        """Node Class for the Monte Carlo Tree Search

        Args:
            game (_type_): _description_
            mother (_type_, optional): _description_. Defaults to None.
            prob (_type_, optional): _description_. Defaults to torch.tensor(0., dtype=torch.float).
        """
        self.game = game
          
        # child nodes
        self.child = {}
        # numbers for determining which actions to take next
        self.U = 0

        # V from neural net output
        # it's a torch.tensor object
        # has require_grad enabled
        self.prob = prob
        # the predicted expectation from neural net
        self.nn_v = torch.tensor(0., dtype=torch.float)
        
        # visit count
        self.N = 0

        # expected V from MCTS
        self.V = 0

        # keeps track of the guaranteed outcome
        # initialized to None
        # this is for speeding the tree-search up
        # but stopping exploration when the outcome is certain
        # and there is a known perfect play
        self.outcome = self.game.score


        # if game is won/loss/draw
        if self.game.score is not None:
            self.V = self.game.score * self.game.player 
            if self.game.score == 0:
                self.U = 0
            else: 
                self.U = self.V * float('inf')

        # link to previous node
        self.mother = mother
        
    
    def create_child(self, actions, probs):
        """
        Create child nodes for the current node based on possible actions and their probabilities.

        Args:
            actions (list): A list of possible actions from the current game state.
            probs (list): A list of probabilities corresponding to each action.
        """
        # Directly create the child dictionary with game states updated for each action
        games = [copy(self.game) for _ in actions]
        
        for action, game in zip(actions, games):
            game.move(action)
        
        child = { tuple(a): Node(g, self, p) for a, g, p in zip(actions, games, probs) }
        
        self.child = child
        
        
    def explore(self, policy):
        """
        Explore the game tree to expand nodes, evaluate them using a policy, and backpropagate the results.

        This function performs a series of actions:
        - Validates the game state is not at a terminal point.
        - Iteratively selects child nodes based on the highest utility (U) value until a leaf node is reached.
        - If the leaf node is unexpanded, it uses the provided policy to evaluate the game state and create child nodes.
        - Updates visit counts (N) and value estimates (V) of nodes up the tree from the leaf to the root.

        Args:
            policy (callable): A function that accepts a game state and returns a tuple of next actions, their probabilities, and an evaluation value (v).
        """
        # Check if the game is already over and raise an error if so.
        if self.game.score is not None:
            raise ValueError(f"Game has ended with score {self.game.score}")

        current = self

        # Explore children of the node based on U values to find the best action.
        while current.child and current.outcome is None:
            # Find the child with the maximum U value.
            max_U = max(c.U for c in current.child.values())
            # Select all actions leading to children with this max U value.
            actions = [action for action, c in current.child.items() if c.U == max_U]
            
            # Handle case where no actions are found (should not happen in correctly implemented MCTS).
            if not actions:
                print(f"Error: zero length actions with max_U = {max_U}")
                print(current.game.state)
                break

            # Randomly choose among the best actions.
            action = random.choice(actions)
            
            # Handle cases where max U indicates a terminal state.
            if max_U == float("inf"):
                current.U = -float("inf")
                current.V = -1.0
                break
            
            if max_U == -float("inf"):
                current.U = float("inf")
                current.V = 1.0
                break

            # Move to the selected child node to continue exploration.
            current = current.child[action]

        # If the current node is a leaf and not a terminal game state, expand it using the policy.
        if not current.child and current.outcome is None:
            next_actions, probs, v = process_policy(policy, current.game)
            current.nn_v = -v  # Store the negated policy value (assuming opponent's perspective).
            current.create_child(next_actions, probs)  # Create children based on policy output.
            current.V = -float(v)  # Set the current node's value to the negated policy value.
        
        current.N += 1  # Initialize the visit count for the newly expanded node.

        # Back-propagate the evaluation from the expanded node up to the root.
        while current.mother:
            mother = current.mother
            mother.N += 1  # Increment visit count.
            # Update the value estimate based on the newly evaluated child.
            mother.V += (-current.V - mother.V) / mother.N

            # Update U values for all siblings based on the new visit count and value estimates.
            for sibling in mother.child.values():
                if sibling.U not in [float("inf"), -float("inf")]:  # Skip update for terminal nodes.
                    sibling.U = sibling.V + c * float(sibling.prob) * sqrt(mother.N) / (1 + sibling.N)

            current = mother  # Move up the tree to continue back-propagation.
    
    
    def next(self, temperature=1.0):
        """
        Selects the next state from child nodes based on their utilities and visit counts.

        Args:
            temperature (float): A parameter controlling exploration; higher values increase randomness.

        Returns:
            tuple: The selected next state and a tuple of various metrics (-V, -nn_v, prob, nn_prob).
        """
        # Check if the game has already ended.
        if self.game.score is not None:
            raise ValueError(f'Game has ended with score {self.game.score}')

        # Ensure there are child nodes to select from.
        if not self.child:
            print(self.game.state)
            raise ValueError('No children found and game hasn\'t ended')
        
        child = self.child
        
        # Extract utilities and visit counts.
        children = child.values()
        U_values = [c.U for c in children]
        N_values = [c.N for c in children]

        # Handle winning moves directly.
        if max(U_values) == float("inf"):
            prob = torch.tensor([1.0 if c.U == float("inf") else 0 for c in children], device=device)
        else:
            # Adjust visit counts for numerical stability and apply temperature.
            maxN = max(N_values) + 1
            prob = torch.tensor([(node.N / maxN) ** (1 / temperature) for node in children], device=device)

        # Normalize probabilities, using random distribution if all zeros.
        prob = prob / prob.sum() if prob.sum() > 0 else torch.full((len(children),), 1.0 / len(children), device=device)

        # Extract and stack neural network probabilities for child nodes.
        nn_prob = torch.stack([node.prob for node in children]).to(device)

        # Select the next state based on the calculated probabilities.
        next_state = random.choices(list(children), weights=prob)[0]
        
        # Return the next state along with various metrics, adjusting V and nn_v for the current player.
        return next_state, (-self.V, -self.nn_v, prob, nn_prob)


    def detach_mother(self):
        """
        Detaches the current node from its mother node by removing the reference.
        """
        del self.mother
        self.mother = None

