import numpy as np
import random
import heapq
from collections import deque, namedtuple

from model import QNetwork

import torch
import torch.optim as optim

DEBUG_ON = False        # control whether assert statements for debugging are executed

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LEARN_EVERY = 4         # how often to learn parameters of the network
UPDATE_EVERY = 4        # how often to update the target network parameters
INIT_PRIO = 1.0         # initial priority for prioritized experience replay
PRIO_ALPHA = 0.7        # exponent applied to priority to obtain probability

# Encode an action value into a two-element vector. First index encodes moves with zero being non-move,
# and the seond encodes turns with zero being non-turn.
ACT_INVALID = -1
ACT_FORWARD = 0
ACT_BACKWARD = 1
ACT_LEFT = 2
ACT_RIGHT = 3
ACT_CODES = [np.array([ 1,  0], dtype=int), # work forward
             np.array([-1,  0], dtype=int), # work backward
             np.array([ 0,  1], dtype=int), # turn left
             np.array([ 0, -1], dtype=int)] # turn right

# Orientation vector table for obtaining the x-y orientation vector for calculating the new simulated position
# resulting from from an agent action.
NUM_ORIS = 12 # number of uniformly spaced orientations
ORIVEC_TABLE = [np.array([np.cos(2*ori*np.pi/NUM_ORIS), np.sin(2*ori*np.pi/NUM_ORIS)]) for ori in range(NUM_ORIS)]

# Lookup table for converting move code inferred from two consecutive simulated states to the next move action
# to avoid if the observed states are essentially the same indicating a stucked agent.
# mcode = -1 (moved backward) -- mcode+1 --> 0 -- lookup --> ACT_BACKWARD (avoid moving in same direction)
# mcode = 0  (no movement)    -- mcode+1 --> 1 -- lookup --> ACT_INVALID (no move action to avoid)
# mcode = 1  (moved forward)  -- mcode+1 --> 2 -- lookup --> ACT_FORWARD (avoid moving in same direction)
# Table Index:      0,            1,           2
AVOID_MOVE_TABLE = [ACT_BACKWARD, ACT_INVALID, ACT_FORWARD]

# Lookup table for converting turn code inferred from two consecutive simulated states to the next turn action
# avoid if the observed states are essentially the same indicating an agent in blind spot.
# tcode = -1, NUM_ORIS-1   (turned right) --> (tcode+1)%NUM_ORIS --> 0 -- lookup --> ACT_LEFT (avoid turning back)
# tcode = 0                (no turn)      --> (tcode+1)%NUM_ORIS --> 1 -- lookup --> ACT_INVALID (no turn action to avoid)
# tcode = 1, -(NUM_ORIS-1) (turned left)  --> (tcode+1)%NUM_ORIS --> 2 -- lookup --> ACT_RIGHT (avoid turning back)
# Table Index:      0,        1,           2
AVOID_TURN_TABLE = [ACT_LEFT, ACT_INVALID, ACT_RIGHT]
            
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDQNPERAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, tor_dstate, srpt_pens, lrn_rate, hsize1, hsize2, seed=0):
        """Initialize a DDQN Agent object with PER (Prioritized Experience Replay) support.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            tor_dstate (float): tolerance for deciding whether two states are the same
            srpt_pens (array_like): penalty (negative reward) values for undesirable actions
            lrn_rate (float): learning rate for Q-Network training
            hsize1 (int): size of the first hidden layer of the Q-Network
            hsize2 (int): size of the second hidden layer of the Q-Network 
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.tor_dstate = tor_dstate
        self.srpt_pens = srpt_pens
        self.lrn_rate = lrn_rate
        
        self.hsize1 = hsize1
        self.hsize2 = hsize2
        
        self.seed = seed
        if seed is not None: random.seed(seed)
        
        # Each penalty value adds a vector of action_size to signal which action causes the penalty.
        self.aug_state_size = state_size + len(srpt_pens) * action_size
                    
        # Set up Q-Networks.
        self.qnetwork_local = QNetwork(self.aug_state_size, action_size, hsize1, hsize2, seed).to(device)
        self.qnetwork_local.initialize_weights() # initialize network with random weights
        self.qnetwork_target = QNetwork(self.aug_state_size, action_size, hsize1, hsize2, seed=None).to(device)
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
    
    def reset_epsisode(self, state, srpt_det=0):
        """Re-initialize buffers after environment reset for a new episode.
        
        Params
        ======
            state (array_like): initial state after environment reset
            srpt_det (int): number of repeated state types to be checked for post-processing
        """
        self.srpt_det = 0
        if len(self.srpt_pens) == 0:
            # State repeat detection for post-processing is active only when state repeat penalty option is off.
            self.srpt_det = srpt_det
        else:
            # This is used to signal self.step() hasn't been run yet.
            self.next_aug_state = None
                
        if len(self.srpt_pens) > 0 or self.srpt_det > 0:
            self.state_buffer = deque(maxlen=2)
            buffer_size = 2 * (max(len(self.srpt_pens), self.srpt_det) - 1)
            self.smsta_buffer = deque(maxlen=max(2, buffer_size))
        
            # The initial state will be pushed to the buffer again and be compared to this state in the process of
            # selecting the first action. So add 1 to the initial state here to ensure the states are different
            # enough for the first comparison.
            self.state_buffer.append(np.array(state) + 1)
        
            # Any position and orientation can be the initial simulated state here. It is like putting in a
            # coordinate system (origin and x-direction) for a 2-D plane and all the other simulated states
            # in the episode will be specified based on this reference coordinate system.
            self.smsta_buffer.append((np.array([0, 0]), 0))
               
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
        if len(self.srpt_pens) > 0:
            # Augment state vector and modify reward using state repeat penalty values.
            self.state_buffer.append(np.array(next_state))
            self.next_aug_state = self.augment_state(next_state)
            state = self.aug_state
            next_state = self.next_aug_state
            reward = self.modify_reward(reward, state, action)
        
        # Save experience in replay memory.
        self.ebuffer.add(state, action, reward, next_state, done)
        
        # Learn every LEARN_EVERY steps after memory reaches batch_size.
        if len(self.ebuffer.memory) >= self.ebuffer.batch_size:
            self.l_step += 1
            self.l_step %= LEARN_EVERY
            if self.l_step == 0:
                experiences, weights = self.ebuffer.sample()
                self.learn(experiences, weights, GAMMA)
        
    def augment_state(self, state):
        """Augment state vector to penalize undesirable actions.
        
        Params
        ======
            state (array_like): original state vector to be augmented
        Returns
        ======
            aug_state (numpy.ndarray): augmented state vector
        """
        # Each penalty value adds a vector of action_size to signal which action causes the penalty.       
        aug_state = np.concatenate((state, np.zeros((len(self.srpt_pens)*self.action_size,))))
        
        # Detect situation where the two preceeding observed states (not augmented) are essentially the
        # same, which indicates the agent is either stucked at a wall or in some kind of undesirable
        # blind spot. The next action to avoid (i.e., to be penalized) is the one that will keep the
        # agent stuck or in blind spot.
        avoid_action = self.get_avoid_action()
        if avoid_action != ACT_INVALID: aug_state[self.state_size + avoid_action] = 1
        if avoid_action != ACT_INVALID or len(self.srpt_pens) == 1: return aug_state
        
        # If agent is not stuck or in blind spot and there are more penalty values, continue to check
        # state repeats separated by more than two actions. Assuming NUM_ORIS is even, states separated
        # by odd number of actions won't repeat. So only even number of actions needs to be checked.
        for action in range(self.action_size):
            nxt_sta = self.sim_step(action)
            for act_cnt in range(2, 2*len(self.srpt_pens), 2):
                if self.is_state_repeated(act_cnt, nxt_sta):
                    aug_state[self.state_size + (act_cnt // 2) * self.action_size + action] = 1 # signal undesirable action
                    break
        
        return aug_state
    
    def modify_reward(self, reward, aug_state, action):
        """Modify reward to penalized undesirable action.
        
        Params
        ======
            reward (float): original reward
            aug_state (numpy.ndarray): augmented state vector
            action (int): action performed
        Returns
        ======
            reward (float): modified reward
        """
        # Penalize undesirable action when it doesn't earn a reward or cause a penalty. If it earns a positive
        # reward or causes a more negative reward, leave the reward unchanged.
        if reward <= 0:
            for i, penalty in enumerate(self.srpt_pens):
                if aug_state[self.state_size + i * self.action_size + action] > 0: # action is undesirable
                    reward = min(reward, penalty)
                    break
        return reward
    
    def sim_step(self, action):
        """Advance simulated state (position and orientation) for one step by the action.
        
        Params
        ======
            action (int): action to advance the simulated state
        Returns
            pos, ori (numpy.ndarray, int): resulting simulated state
        """
        # An action can either be a move or turn (but not both) with the type of actions (including non-actions)
        # identified by the action code.
        pos, ori = self.smsta_buffer[-1]
        act_code = ACT_CODES[action]
        pos = pos + act_code[0] * ORIVEC_TABLE[ori]
        ori = (ori + act_code[1]) % NUM_ORIS
        return pos, ori
    
    def is_state_repeated(self, act_cnt, nxt_sta):
        """Check whether the next state repeats the past state separated by the specified number of actions.
        
        Params
        ======
            act_cnt (int): number of actions separating the past state to be checked and the next state
            nxt_sta (numpy.ndarray, int): next state resulting from an action
        Returns
        ======
            repeated (bool): indicator for repeated state
        """        
        repeated = False
        if act_cnt <= len(self.smsta_buffer):
            chk_sta = self.smsta_buffer[-act_cnt] # past state to be checked
            if chk_sta[1] == nxt_sta[1]:
                if np.linalg.norm(nxt_sta[0] - chk_sta[0]) <= self.tor_dstate: repeated = True
        return repeated
                
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
        # If the agent is in testing mode, self.step() won't be invoked and some of the operations done there
        # need to be done here.
        if (len(self.srpt_pens) > 0 and self.next_aug_state is None) or self.srpt_det > 0:
            # Push current state into state buffer for comparing with previous state if it is not alraedy pushed
            # by self.step() in the agent training process.
            self.state_buffer.append(np.array(state))
            
        if len(self.srpt_pens) > 0:
            if self.next_aug_state is None:
                self.aug_state = self.augment_state(state)
            else:
                self.aug_state = self.next_aug_state
            state = self.aug_state
        
        if self.srpt_det == 0: # no checking for repeated states (observed or simulated)
            # Randomly select action.
            action = random.choice(np.arange(self.action_size))
            
            # Epsilon-greedy action selection.
            if random.random() >= eps:
                state = torch.from_numpy(state).float().to(device)
                self.qnetwork_local.eval()
                with torch.no_grad(): action = self.qnetwork_local(state).squeeze().argmax().cpu().item()
                    
            if len(self.srpt_pens) > 0:
                # Update simulated state buffer with result of chosen action.
                nxt_sta = self.sim_step(action)
                self.smsta_buffer.append(nxt_sta)
                
            return action
        
        # This is the implementation of the post-processing of the Epsilon-greedy policy to avoid repeated states
        # within a short series of actions. This option is set in self.reset_episode() for each espisode and is
        # only active when the option of penalizing undesirable actions, which is set for the class object, is
        # disabled when len(self.srpt_pens) == 0. To accomondate the post-processing of the selected actions, the
        # random policy is modified to randomly assign rankings to all the available actions.
        
        # Randomly assign rankings to action candidates.
        ranked_actions = np.random.permutation(self.action_size)
                   
        # Epsilon-greedy action selection.
        if random.random() >= eps:
            state = torch.from_numpy(state).float().to(device)
            self.qnetwork_local.eval()
            with torch.no_grad(): neg_act_qvals = -self.qnetwork_local(state).squeeze()
            ranked_actions = neg_act_qvals.argsort().cpu().numpy().astype(int)
        
        # Post-process ranked action candidates to remove undesirable action.
        avoid_action = self.get_avoid_action()
        action = self.select_nosrpt_action(avoid_action, ranked_actions)
       
        return action
        
    def get_avoid_action(self):
        """Avoid action that will keep the agent stucked or in a blind spot. 
        
        Returns
            avoid_action (int): next action to avoid
        """
        avoid_action = ACT_INVALID # used to sigal agent is not stucked or in a blind spot       
        if np.linalg.norm(self.state_buffer[1] - self.state_buffer[0]) <= self.tor_dstate:
            sim_sta0 = self.smsta_buffer[-2]
            sim_sta1 = self.smsta_buffer[-1]
            if sim_sta0[1] == sim_sta1[1]: # action is not a turn, must be a move
                # Agent is stuck at a wall
                dpos = sim_sta1[0] - sim_sta0[0]
                mcode = np.around(np.dot(dpos, ORIVEC_TABLE[sim_sta0[1]])).astype(int) # dot(mcode*(cos, sin), (cos, sin)) = mcode
                avoid_action = AVOID_MOVE_TABLE[mcode+1]
                self.smsta_buffer.clear()          # it is reasonable to backtrack to get unstucked except the last state which
                self.smsta_buffer.append(sim_sta0) # the agent is stucked in (as the new reference, it can be any state)           
            else: # action is a turn
                # Agent is in a blind spot (turned, but observed same state).
                tcode = sim_sta1[1] - sim_sta0[1]
                avoid_action = AVOID_TURN_TABLE[(tcode+1)%NUM_ORIS]    
                self.smsta_buffer.clear()          # it is reasonable to backtrack to get out of blind
                self.smsta_buffer.append(sim_sta0) # spot except the last two states, which represent
                self.smsta_buffer.append(sim_sta1) # the blind spot                   
        return avoid_action

    def select_nosrpt_action(self, avoid_action, ranked_actions):
        """Select action that avoids repeated state (i.e., loops) by a short series of actions.
        
        Params
        ======
            avoid_action (int): action to avoid if agent is stuck or in blind spot
            ranked_actions (array like): action candidates ranked by decreasing Q-values
        Returns
        ======
            action (int): the selected action
        """
        action = ranked_actions[0]
        if action == avoid_action: action = ranked_actions[1]
        nxt_sta = self.sim_step(action)
                
        # If repeated observed state by an action is detected (signaled by avoid_action != ACT_INVALID), the selected
        # action for avoiding the repeated state will be used since it is more important to free a agent that is
        # stucked or in a blind spot than to go back further to check for repeated simulated states. So the checking
        # for repeated simulated states by 2 or more actions will only occur when avoid_action == ACT_INVALID.
        if avoid_action == ACT_INVALID and self.srpt_det > 1:
            act_heapq = []
            for action in ranked_actions:
                nxt_sta = self.sim_step(action)
                for act_cnt in range(2, 2*self.srpt_det, 2): # assuming NUM_ORIS is even, only check even number of actions
                    if self.is_state_repeated(act_cnt, nxt_sta):
                        # Simulated state repeated, go checking next action.
                        heapq.heappush(act_heapq, [-act_cnt, action, nxt_sta])
                        break
                else:
                    # No repeated state detected, action is found.
                    break
            else:
                # No action can satisfy all the no repeated state conditions, select the action that repeats the
                # state separated by most actions (i.e., long loop is more acceptable than short loop).
                action, nxt_sta = heapq.heappop(act_heapq)[1:]
        
        self.smsta_buffer.append(nxt_sta) # update simulated state buffer with result of chosen action.
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
            self.qnetwork_solved = QNetwork(self.aug_state_size, self.action_size, self.hsize1, self.hsize2, seed=None).to(device)
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
        qnetwork_loaded = QNetwork(self.aug_state_size, self.action_size, self.hsize1, self.hsize2, seed=None)
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
                   