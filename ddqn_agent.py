import numpy as np
import random
from collections import deque, namedtuple

from model import QNetwork

import torch
import torch.optim as optim

DEBUG_ON = True         # control whether assert statements for debugging are executed

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LEARN_EVERY = 4         # how often to learn parameters of the network
UPDATE_EVERY = 4        # how often to update the target network parameters
INIT_PRIO = 1.0         # initial priority for prioritized experience replay
PRIO_ALPHA = 0.7        # exponent applied to priority to obtain probability

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDQNPERAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, lrn_rate, hsize1, hsize2, seed=0):
        """Initialize a DDQN Agent object with PER (Prioritized Experience Replay) support.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            lrn_rate (float): learning rate for Q-Network training
            hsize1 (int): size of the first hidden layer of the Q-Network
            hsize2 (int): size of the second hidden layer of the Q-Network 
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.lrn_rate = lrn_rate
        
        self.hsize1 = hsize1
        self.hsize2 = hsize2
        
        self.seed = seed
        if seed is not None: random.seed(seed)
                    
        # Set up Q-Networks.
        self.qnetwork_local = QNetwork(state_size, action_size, hsize1, hsize2, seed).to(device)
        self.qnetwork_local.initialize_weights() # initialize network with random weights
        self.qnetwork_target = QNetwork(state_size, action_size, hsize1, hsize2, seed=None).to(device)
        self.qnetwork_target.update_weights(self.qnetwork_local) # copy network weights to target network
        self.qnetwork_target.eval()
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lrn_rate)
        
        # Store trained Q-model when the environment is solved.
        self.qnetwork_solved = None

        # Set up experience replay memory.
        self.ebuffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)

        # Initialize interval steps.
        self.l_step = 0 # for learning every LEARN_EVERY time steps
        self.t_step = 0 # for updating target network every UPDATE_EVERY learnings
    
    def step(self, state, action, reward, next_state, done):
        """Update replay memory and parameters of Q-Network by training.
        
        Params
        ======
            state (array_like): starting state of the step
            action (int): action performed in the step
            reward (float): reward from the action
            next_state (array_like): resulting state of the action in the step
            done (bool): indicator for whether next_state is terminal (i.e., end of episode) or not
        """
        # Save experience in replay memory.
        self.ebuffer.add(state, action, reward, next_state, done)
        
        # Learn every LEARN_EVERY steps after memory reaches batch_size.
        if len(self.ebuffer.memory) >= self.ebuffer.batch_size:
            self.l_step += 1
            self.l_step %= LEARN_EVERY
            if self.l_step == 0:
                experiences, weights = self.ebuffer.sample()
                self.learn(experiences, weights, GAMMA)

    def act(self, state, eps=0.0):
        """Select action for given state as per epsilon-greedy current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for adjusting epsilon-greedy action selection
        Returns
        ======
            action (int): the chosen action
        """
        # Randomly select action.
        action = random.choice(np.arange(self.action_size))
        
        # Epsilon-greedy action selection.
        if random.random() >= eps:
            state = torch.from_numpy(state).float().to(device)
            self.qnetwork_local.eval()
            with torch.no_grad(): action = self.qnetwork_local(state).squeeze().argmax().cpu().item()
            
        return action
        
    def learn(self, experiences, is_weights, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple (s, a, r, s', done) of batched experience data
            is_weights (torch.Tensor): importance sampling weights for the batched experiences
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Double DQN method for obtaining target Q-values.
        self.qnetwork_local.eval()
        with torch.no_grad():
            maxq_actions = self.qnetwork_local(next_states).max(1)[1].unsqueeze(1)
            qouts_next_states = self.qnetwork_target(next_states).gather(1, maxq_actions).squeeze()
        qouts_target = rewards + gamma * qouts_next_states * (1 - dones)

        # Obtain current Q-values and its difference from the target Q-values.
        self.qnetwork_local.train()
        qouts_states = self.qnetwork_local(states).gather(1, actions).squeeze()
        delta_qouts = qouts_states - qouts_target
        
        # Calculated weighted sum of squared losses.
        wsqr_loss = is_weights * delta_qouts**2 # weighted squared loss
        loss_sum = wsqr_loss.sum()
                
        # Update model parameters by minimizing the loss sum.
        self.optimizer.zero_grad()
        loss_sum.backward()
        self.optimizer.step()
        
        # Update priorities of the replay memory.
        neg_prios = -torch.abs(delta_qouts.detach())
        self.ebuffer.update_priorities(neg_prios.cpu().numpy())        

        # Update target network.
        self.t_step += 1
        self.t_step %= UPDATE_EVERY
        if self.t_step == 0: self.qnetwork_target.update_weights(self.qnetwork_local, TAU)

    def update_beta(self, beta):
        """Update importance sampling weights for memory buffer with new Beta.

        Params
        ======
            beta (float): new Beta value
        """
        if beta != self.ebuffer.beta:
            self.ebuffer.beta = beta
            if len(self.ebuffer.memory) >= self.ebuffer.batch_size: self.ebuffer.update_is_weights()
            
    def copy_solved_qnet(self):
        """Copy current local Q-Network to solved Q-Network while local Q-Network will continue the training."""             
        if self.qnetwork_solved is None:
            self.qnetwork_solved = QNetwork(self.state_size, self.action_size, self.hsize1, self.hsize2, seed=None).to(device)
        self.qnetwork_solved.update_weights(self.qnetwork_local) # copy local network weights to solved network
            
    def save_qnet(self, model_name):
        """Save Q-Network parameters into file.

        Params
        ======
            model_name (str): name of the Q-Network
        """
        # Save CPU version since it can be used with or without GPU.
        if self.qnetwork_solved is not None:
            torch.save(self.qnetwork_solved.cpu().state_dict(), model_name+'.pth')
            self.qnetwork_solved = self.qnetwork_solved.to(device)
        else:
            torch.save(self.qnetwork_local.cpu().state_dict(), model_name+'.pth')
            self.qnetwork_local = self.qnetwork_local.to(device)            
    
    def load_qnet(self, model_name):
        """Load Q-Network parameters from file.

        Params
        ======
            model_name (str): name of the Q-Network
        """
        # Saved QNetwork is alway the CPU version.
        qnetwork_loaded = QNetwork(self.state_size, self.action_size, self.hsize1, self.hsize2, seed=None)
        qnetwork_loaded.load_state_dict(torch.load(model_name+'.pth'))
        self.qnetwork_local.update_weights(qnetwork_loaded.to(device)) # copy loaded network weights to local network


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples with support for PER (Prioritized Experience Replay) sampling."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.beta = 1.0 # full importance sampling by default

        self.memory = []
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

        self.seed = seed
        if seed is not None: np.random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory.

        Params
        ======
            state (array_like): starting state of the step
            action (int): action performed in the step
            reward (float): reward from the action
            next_state (array_like): resulting state of the action in the step
            done (bool): indicator for whether next_state is terminal (i.e., end of episode) or not
        """
        e = self.experience(state, action, reward, next_state, done)    
        if not self.memory: # memory is empty
            # Start the value and index queues for the negative priorities of the experiences.
            self.memory.append(e)
            self.nprio_queue = np.array([-INIT_PRIO])
            self.queue_idxes = np.array([0], dtype=np.int64)
            self.rnk_probs = [1.0]
        elif len(self.memory) < self.buffer_size: # memory is not yet full
            # Assgin highest priority to new experience and put it to the front of the experience queue.
            new_idx = len(self.memory)
            self.memory.append(e)
            self.nprio_queue = np.concatenate(([self.nprio_queue[0]], self.nprio_queue))
            self.queue_idxes = np.concatenate(([new_idx], self.queue_idxes))
            self.rnk_probs.append(np.power(new_idx+1, -PRIO_ALPHA))

            if len(self.memory) >= self.batch_size:
                # Partition distribution into batch_size number of bins of approximately equal probability.
                rank_dist = np.array(self.rnk_probs)
                for i in range(1, len(self.memory)): rank_dist[i] += rank_dist[i-1]
                rank_dist /= rank_dist[-1]             
                unif_dist = np.array([k/self.batch_size for k in range(1, self.batch_size)])
                self.bin_bnds = np.concatenate(([0], np.searchsorted(rank_dist, unif_dist), [len(self.memory)]))
                        
                # Adjust distribution partitions to fill in empty bins.
                for k in range(1, self.batch_size):
                    if self.bin_bnds[k] <= self.bin_bnds[k-1]: self.bin_bnds[k] = self.bin_bnds[k-1] + 1
                        
                if DEBUG_ON:
                    assert(np.all(self.bin_bnds[1:] > self.bin_bnds[:-1])), \
                        "Some partition bins are empty at memory length {}!".format(len(self.memory))
                        
                # Obtain sampling probabilities for the partitioned bins. The actaul probabilities are also adjusted based
                # on the partition. When the memory buffer is starting to fill, the adjustment amount will be large and the
                # amount will decrease to essentially zero for a full buffer if the buffer size is large. The probabilities
                # of each bin is used as input to numpy.random.choice for sampling a batch of experiences for training the
                # Q-Network (This is different from that was presented in the Prioritezed Experience Replay paper, which 
                # uniformly samples the experiences within each partitioned bin). So they sum to one; but each probability
                # is amplified by a factor of batch_size and should be accounted for in calculating the importance
                # sampling weights.
                self.smp_probs = np.array(self.rnk_probs)
                for k in range(self.batch_size):
                    self.smp_probs[self.bin_bnds[k]:self.bin_bnds[k+1]] /= np.sum(self.smp_probs[self.bin_bnds[k]:self.bin_bnds[k+1]])
                self.update_is_weights()
        else: # memory buffer is full
            # Replace old experience with lowest priority with new one.
            rep_idx = self.queue_idxes[-1]
            self.memory[rep_idx] = e
            
            # Assign highest priority to new experience and put it to the front of the experience queue
            self.nprio_queue = np.concatenate(([self.nprio_queue[0]], self.nprio_queue[:-1]))
            self.queue_idxes = np.concatenate(([rep_idx], self.queue_idxes[:-1]))
                                
    def update_is_weights(self):
        """Update importance sampling weights."""
        # Account for the batch_size factor in sampling probabilities (smp_probs).
        self.is_weights = np.power((len(self.memory)/self.batch_size)*self.smp_probs, -self.beta)
        self.is_weights /= self.is_weights[-1]

    def sample(self):
        """Sample a batch of experiences from memory using prioritized experience sampling.
        
        Returns
        ======
            experiences (Tuple[torch.Tensor]): tuple of batched experience data of (s, a, r, s', done)
            is_weights (torch.Tensor): importance sampling weights of torch.Size([batch_size])
        """
        # Obtain selected indices from the negative priority queue and the corresponding indices of the replay memory.
        self.sel_idxes = np.array([ np.random.choice(np.arange(self.bin_bnds[k], self.bin_bnds[k+1]),
                                                     p=self.smp_probs[self.bin_bnds[k]:self.bin_bnds[k+1]])
                                    for k in range(self.batch_size) ])
        self.smp_idxes = self.queue_idxes[self.sel_idxes]

        # Obtain the sampled experiences and their corresponding importance sampling weights.
        experiences = [self.memory[idx] for idx in self.smp_idxes]
        is_weights = torch.from_numpy(self.is_weights[self.sel_idxes]).float().to(device)
                                  
        if DEBUG_ON:
            assert(len(experiences) == self.batch_size), "Sampled size is not the same as batch size!"

        # Break experience data into tuples of Torch Tensor data.
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(device)
        rewards = torch.from_numpy(np.array([e.reward for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.array([e.done for e in experiences]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones), is_weights

    def update_priorities(self, neg_prios):
        """Update priorities of the sampled experiences in memory.

        Params
        ======
            neg_prios (array_like): new negative priority values used for updating
        """
        # Remove old negative priority values and their corresponding indices from queues.
        nprio_queue = np.delete(self.nprio_queue, self.sel_idxes)
        queue_idxes = np.delete(self.queue_idxes, self.sel_idxes)

        # Obtain insertion points from the sorted new negative priority values.
        srt_idxes = np.argsort(neg_prios)
        neg_prios = neg_prios[srt_idxes]
        smp_idxes = self.smp_idxes[srt_idxes]
        ins_idxes = np.concatenate((np.searchsorted(nprio_queue, neg_prios), [len(self.memory)]))

        # Build new value and index queues for the negative priorities of the experiences.                         
        self.nprio_queue = nprio_queue[:ins_idxes[0]]
        self.queue_idxes = queue_idxes[:ins_idxes[0]]
        for k in range(self.batch_size):
            self.nprio_queue = np.concatenate((self.nprio_queue, [neg_prios[k]], nprio_queue[ins_idxes[k]:ins_idxes[k+1]]))
            self.queue_idxes = np.concatenate((self.queue_idxes, [smp_idxes[k]], queue_idxes[ins_idxes[k]:ins_idxes[k+1]]))
                                  
        if DEBUG_ON:
            assert(len(self.nprio_queue) == len(self.memory)), \
                "Length of negative priority queue is incorrect after updating!"
            assert(np.all(self.nprio_queue[1:] >= self.nprio_queue[:-1])), \
                "Negative priority queue is no longer sorted after updating!"
                   