import numpy as np
from collections import deque
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def train_ddqn_per(env, agent, n_episodes, print_every, beta_start, qnet_name, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning training.
    
    Params
    ======
        env (UnityEnvironment): environment to be solved
        agent (DDQNPERAgent): DDQN Agent with PER (Prioritized Experience Replay) support
        n_episodes (int): maximum number of training episodes
        print_every (int): print average training score print_every number of episodes
        beta_start (float): starting Beta value for importance sampling in Prioritized Experience Replay
        qnet_name (str): name of the QNetwork to be saved
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    Returns
    ======
        scores (list): scores of all the training episodes for plotting
    """
    
    scores = []                        # list containing scores from each episode
    best_score = -np.inf               # best average score achieved
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        # Anneal Beta value to 1 for the last training episode.
        beta = (i_episode - 1) * (1 - beta_start) / (n_episodes - 1) + beta_start
        agent.update_beta(beta)
        
        env_info = env.reset(train_mode=True)[env.brain_names[0]]
        state = env_info.vector_observations[0]
        score = 0
        
        while True:
            action = agent.act(state, eps)
            env_info = env.step(action)[env.brain_names[0]]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                eps = max(eps_end, eps_decay*eps) # decrease epsilon
                break
                
        # Save most recent score.
        scores_window.append(score)
        scores.append(score)
        
        # Store QNetwork that solved the environment and with the best average score.
        mean_score = np.mean(scores_window)
        if i_episode >= 100 and mean_score >= 13 and mean_score > best_score:
            agent.copy_solved_qnet()
            if best_score == -np.inf:
                solve_episode = i_episode - 100
                solve_score = mean_score
            best_episode = i_episode - 100
            best_score = mean_score
            
        if best_score == -np.inf:
            print('\rEpisode {:4d}\tAverage Score: {:6.2f}'.format(i_episode, mean_score), end="")
        else:
            print('\rEpisode {:4d}\tAverage Score: {:6.2f} (Best Solved Average Score: {:6.2f} at Episode {:4d}!)'.format(
                i_episode, mean_score, best_score, best_episode), end="")
        if i_episode % print_every == 0 or i_episode == n_episodes: print('\r')
    
    if best_score != -np.inf:
        print("First Solved Average Score: {:6.2f} at Episode {:4d}!".format(solve_score, solve_episode))
    else:
        print("Environment not solved in {} episodes!".format(n_episodes - 100))
        
    agent.save_qnet(qnet_name)
    return scores

def test_ddqn_per(env, agent, n_episodes):
    """Deep Q-Learning testing.
    
    Params
    ======
        env (UnityEnvironment): environment to be solved
        agent (DDQNPERAgent): DDQN Agent with PER (Prioritized Experience Replay) support
        n_episodes (int): maximum number of training episodes
    Returns
    ======
        scores (list): scores of all the testing episodes for plotting
    """
    scores = []
    for i in range(n_episodes):
        env_info = env.reset(train_mode=False)[env.brain_names[0]]
        state = env_info.vector_observations[0]             
        
        score = 0                                           
        while True:
            action = agent.act(state)                       
            env_info = env.step(action)[env.brain_names[0]] 
            next_state = env_info.vector_observations[0]    
            reward = env_info.rewards[0]                    
            done = env_info.local_done[0]                   
            score += reward                            
            state = next_state 
            if done: break
            
            if i == 0:
                print("\rTest Run {:3d}\tScore: {:3d}".format(i+1, int(score)), end="")
            else:
                print("\rTest Run {:3d}\tScore: {:3d} (Previous Run Final Score = {:3d})".format(
                    i+1, int(score), int(final_score)), end="")                    
        
        final_score = score
        scores.append(score)
        
    print("\rTest Results: MeanScore = {:6.2f}, MaxScore = {:3d}, MinScore = {:3d}".format(
        np.mean(scores), int(np.amax(scores)), int(np.amin(scores))))
    return scores
        
def plot_train_scores(scores, rolling_window=100):
    """Plot training scores and rolling mean using specified window.
    
    Params
    ======
        scores (list): scores to be plotted
        rolling_window (int): size of rolling window
    """
    plt.plot(scores, label='Raw Data')
    plt.ylabel('Score');
    plt.xlabel('Episode #');
    rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
    plt.plot(rolling_mean, label='Rolling Mean')
    plt.legend(loc='upper left')
        
def plot_test_scores(scores, rolling_window=10):
    """Plot testing scores with rolling mean and score histogram.
    
    Params
    ======
        scores (list): scores to be plotted
        rolling_window (int): size of rolling window
    """
    plt.figure(figsize=(20, 10))
    rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
    
    ax = plt.subplot(1, 2, 1)
    ax.plot(scores, label='Score')
    ax.plot(rolling_mean, label='Rolling Mean')
    ax.legend(loc='upper left')
    ax.set_ylabel('Score')
    ax.set_xlabel('Episode #')
       
    ax = plt.subplot(1, 2, 2)
    ax.hist(scores, bins=25, range=(0, 25), density=True)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.set_xlabel('Score')
    
    plt.show()
