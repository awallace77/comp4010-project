import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """
        Defines a 3-layer Q Network with ReLU activation
    """
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        """
        Args:
            input_dim: size of the observation vector
            output_dim: number of discrete actions
            hidden_dim: number of neurons in hidden layers
        """
        super(QNetwork, self).__init__()
        
        # Two hidden layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output layer: one Q-value per action
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values
