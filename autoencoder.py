import torch.nn as nn
from collections import OrderedDict


class AutoEncoder(nn.Module):
    
    def __init__(self, args):
        super(AutoEncoder, self).__init__()
        self.args = args
        self.input_dim = args.input_dim
        self.output_dim = self.input_dim
        self.hidden_dims = (args.hidden_dims + 
                            args.hidden_dims[:-1][::-1])  # mirrored structure
        self.n_layers = len(self.hidden_dims)
        self.n_classes = args.n_classes
        
        # Validation check
        assert self.n_layers % 2 > 0
        assert self.hidden_dims[self.n_layers // 2] == self.n_classes
        
        # Initialize layers
        layers = OrderedDict()
        for idx, hidden_dim in enumerate(self.hidden_dims):
            if idx == 0:
                layers.update(
                    {'linear0': nn.Linear(self.input_dim, self.hidden_dims[0]),
                     'activation0': nn.ReLU()
                    })
            elif idx == self.n_layers - 1:
                layers.update(
                    {'linear{}'.format(idx): nn.Linear(
                        self.hidden_dims[-1], self.output_dim)
                    })
            else:
                layers.update(
                    {'linear{}'.format(idx): nn.Linear(
                        self.hidden_dims[idx-1], self.hidden_dims[idx]),
                     'activation{}'.format(idx): nn.ReLU(),
                     # 'bn{}'.format(idx): nn.BatchNorm1d(self.hidden_dims[idx])
                    })
        
        self.model = nn.Sequential(layers)
    
    def __repr__(self):
        repr_str = '[Structure]: {}-'.format(self.input_dim)
        for idx, dim in enumerate(self.hidden_dims):
                repr_str += '{}-'.format(dim)
        repr_str += str(self.output_dim) + '\n'
        repr_str += '[n_layers]: {}'.format(self.n_layers) + '\n'
        repr_str += '[n_classes]: {}'.format(self.n_classes) + '\n'
        repr_str += '[input_dims]: {}'.format(self.input_dim)
        return repr_str
    
    def __str__(self):
        return self.__repr__()
    
    def forward(self, X):
        return self.model(X)

    """ Hook on collecting the latent output """
    def latent_forward(self, X):
        output = X
        for module in self.model:
            output = module(output)
            if output.size()[1] == self.args.n_classes:
                return output
