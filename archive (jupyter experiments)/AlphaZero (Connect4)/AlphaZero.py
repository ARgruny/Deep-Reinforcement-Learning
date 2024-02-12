from sklearn import tree
import MCTS
import torch
from Model import PolicyNetwork
from copy import copy
import random





class AlphaZero:
    """
    AlphaZero Agent Class
    """
    def __init__(
        self,
        temperature: float = 0.1,
        learning_rate: float = 0.01,
        weight_decay: float = 1e-5
    ) -> None:
        """This is the constructor for the AlphaZero class

        Args:
            game (object): The game object
        """
        self.temperature = temperature
        self.learning_rate = learning_rate
        
        self.policy = PolicyNetwork()
        
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )     
        
        
    def policy_player_mcts(self, game: object, n_iter: int = 1000) -> None:
        """This function is used to play a move using the MCTS algorithm

        Args:
            n_iter (int, optional): The number of iterations. Defaults to 1000.

        Returns:
            tree_next.game.last_move: The last move
        """
        self.tree = MCTS.Node(copy(game))
        
        for i in range(n_iter):
            self.tree.explore(self.policy)
            
        tree_next, (_, _, _, _) = self.tree.next(temperature=self.temperature)
        
        return tree_next.game.last_move
    
    
    def explore_and_learn(self, game: object, n_iter: int = 1000) -> None:
        """This function is used to explore the game using the MCTS algorithm

        Args:
            game (object): The game object
            n_iter (int, optional): The number of iterations. Defaults to 1000.
            
        Returns:
            loss.item(): The loss
            outcome: The outcome
        """
        self.tree = MCTS.Node(game)
        
        logterm = []
        vterm = []
        
        while self.tree.outcome == None:
            for _ in range(n_iter):
                self.tree.explore(self.policy)
                if self.tree.N >= n_iter:
                    break
                
            current_player = self.tree.game.player
            tree_next, (_, nn_v, p, nn_p) = self.tree.next()
            self.tree = tree_next
            self.tree.detach_mother()
            
            loglist = torch.log(nn_p) * p
            constant = torch.where(p > 0, p * torch.log(p), torch.tensor(0.))
            logterm.append(-torch.sum(loglist - constant))
            
            vterm.append(nn_v * current_player)
            
        # compute the loss
        outcome = self.tree.outcome
        
        loss = torch.sum((torch.stack(vterm) - outcome) ** 2 + torch.stack(logterm))
        
        # backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), outcome
            
    
    
    def random_player(self, game: object) -> None:
        """This function is used to play a move using the random algorithm

        Returns:
            move: The move
        """
        move = random.choice(game.available_moves())
        
        return move