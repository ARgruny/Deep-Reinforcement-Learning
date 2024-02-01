import torch

class NetowkConfig:
    """Network configuration."""
    def __init__(
        self,
        input_size,
        output_size,
        hidden_sizes,
        activation,
        initialization,
        device,
        seed
    ):
        """Initialize parameters for network configuration.
        
        Params
        ======
            input_size (int): input size of the network.
            output_size (int): output size of the network.
            hidden_sizes (list): list of hidden layer sizes.
            activation (str): activation function.
            initialization (str): type of initialization.
            device (str): device to use for training.
            seed (int): random seed.
        """
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.initialization = initialization
        self.device = device
        self.seed = seed
    


class BaseNetwork:
    """Base class for all networks.
    
    This class defines the basic methods that all networks should have.
    like saving and loading the network.
    print the network architecture.
    count the number of parameters in the network.
    """

    def __init__(self, config):
        """Initialize parameters and build network.
        
        Params
        ======
            config (dict): dictionary of configuration parameters.
        """
        self.config = config


    def save(self, filename):
        """Save the network to a file.
        
        Params
        ======
            filename (str): name of the file to save the network.
        """
        torch.save(self.state_dict(), filename)


    def load(self, filename):
        """Load the network from a file.
        
        Params
        ======
            filename (str): name of the file to load the network.
        """
        self.load_state_dict(torch.load(filename))


    def print(self):
        """Print the network architecture."""
        print(self)


    def count_parameters(self):
        """Count the number of parameters in the network."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    

    def __str__(self):
        """Return the network architecture as a string."""
        return super().__str__()
    

    def __repr__(self):
        """Return the network architecture as a string."""
        return super().__repr__()
    

    def _init_weights_xavier(self, m):
        """Initialize the weights of the network using Xavier initialization.
        
        Params
        ======
            m (torch.nn.Module): layer of the network.
        """
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


    def _init_weights_uniform(self, m):
        """Initialize the weights of the network using uniform initialization.
        
        Params
        ======
            m (torch.nn.Module): layer of the network.
        """
        if type(m) == torch.nn.Linear:
            torch.nn.init.uniform_(m.weight, -3e-3, 3e-3)
            m.bias.data.fill_(0.01)


    def _init_weights_normal(self, m):
        """Initialize the weights of the network using normal initialization.
        
        Params
        ======
            m (torch.nn.Module): layer of the network.
        """
        if type(m) == torch.nn.Linear:
            torch.nn.init.normal_(m.weight, 0, 1e-3)
            m.bias.data.fill_(0.01)
    
    
    def initialize_weights(self, type):
        """Initialize the weights of the network.

        Params
        ======
            type (str): type of initialization.
        """
        if type == 'xavier':
            self.apply(self._init_weights_xavier)
        elif type == 'uniform':
            self.apply(self._init_weights_uniform)
        elif type == 'normal':
            self.apply(self._init_weights_normal)
        else:
            raise ValueError('Unknown initialization: {}'.format(type))