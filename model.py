import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hsize1, hsize2, seed=0):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hsize1 (int): Size of 1st hidden layer
            hsize2 (int): Size of 2nd hidden layer
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = seed
        if seed is not None: torch.manual_seed(self.seed)

        self.state_size = state_size
        self.action_size = action_size
        self.hsize1 = hsize1
        self.hsize2 = hsize2

        self.fc1 = nn.Linear(state_size, hsize1)
        self.fc2 = nn.Linear(hsize1, hsize2)
        self.fc3 = nn.Linear(hsize2, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values.
        
        Params
        ======
            state (torch.Tensor): batch of input states
        Returns
        ======
            x (torch.Tensor): batch of output action Q-values of torch.Size([batch_size, action_size])
        """
        x = F.relu(self.fc1(state.view(-1, self.state_size)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def initialize_weights(self):
        """Initialize model weights using normal distribution."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                std = 1.0 / torch.sqrt(torch.tensor(m.in_features).float())
                m.weight.data.normal_(0.0, std)
                m.bias.data.fill_(0.0)

    def update_weights(self, qnet_local, tau=1.0):
        """Soft update model parameters: θ_target = τ*θ_local + (1 - τ)*θ_target.
        
        Params
        ======
            qnet_local (PyTorch model): weights will be copied from
            tau (float): interpolation parameter (complete replacement if tau = 1)
        """
        for target_param, local_param in zip(self.parameters(), qnet_local.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
