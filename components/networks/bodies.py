import torch.nn as nn
import torch.nn.functional as F

from base import BaseNetwork

class VanillaMLP(BaseNetwork):
    """Vanilla MLP network."""

    def __init__(self, config):
        """Initialize parameters and build model.

        Params
        ======
            config (dict): dictionary of configuration parameters.
        """
        super().__init__(config)

        # define input and output sizes
        self.input_size = config.input_size
        self.output_size = config.output_size

        # define hidden layers
        self.hidden_layers = nn.ModuleList()
        for hidden_size in config.hidden_sizes:
            self.hidden_layers.append(nn.Linear(self.input_size, hidden_size))
            self.input_size = hidden_size

        # define output layer
        self.output = nn.Linear(self.input_size, self.output_size)

        # define activation function
        self.activation = config.activation

    def forward(self, state):
        """Build a network that maps state -> action values."""
        # flatten the state
        x = state.view(state.size(0), -1)

        # pass through hidden layers
        for layer in self.hidden_layers:
            x = self.activation(layer(x))

        # pass through output layer
        return self.output(x)