from IPython import display
import gymnasium as gym
import numpy as np
import random

def epsilon_decay(episode, N, epsilon_start=1.0, epsilon_end=1e-6, decay_type='linear'):
    """ Decay epsilon value over time according to a specified decay model.
    Parameters:
        episode_num (int): current episode number
        N (int): total number of episodes
        epsilon_start (float): starting epsilon value
        epsilon_end (float): ending epsilon value
        decay_type (str): the type of model being used to decay epsilon ('linear' or 'exp')
    Returns:
        epsilon (float): decayed epsilon value
    """

    # Use a linear decay
    if decay_type.lower() == 'linear':
        return epsilon_start - ((epsilon_start - epsilon_end) / N) * (episode - 1)

    # Use an exponential decay
    elif decay_type.lower() == "exp":
        return epsilon_start * (epsilon_end / epsilon_start) ** (episode / N)

    else:
        raise ValueError("The decay types are either 'linear' or 'exp' for exponential.")

def run_qlearn(env, alpha=0.1, gamma=0.6, epsilon=0.1, N=70_000, decay=False, decay_type='linear'):
    """ Use the Q-learning algorithm to find qvalues.
    Parameters:
        env (str): environment name (gym environment)
        alpha (float): learning rate
        gamma (float): discount factor
        epsilon (float): epsilon value for epsilon-greedy algo
        N (int): number of episodes to train for
        decay (bool): whether to decay epsilon according to epsilon_decay
        decay_type (str): the type of model being used to decay epsilon ('linear' or 'exp')
    Returns:
        q_table (ndarray nxm)
    """

    print("Starting Q-learning...")
    # Make environment
    env = gym.make(env, desc=None, map_name='8x8', is_slippery=True)
    # Make Q-table
    q_table = np.zeros((env.observation_space.n, env.action_space.n))

    # Train for N episodes
    print("Training model...")
    for i in range(1, N + 1):

        # Reset episode and get initial state; Initialize penalties, reward, and done
        curr_state, info = env.reset()
        penalties, reward, = 0, 0
        done = False

        # Get epsilon value
        if not decay:
            epsilon = epsilon
        else:
            epsilon = epsilon_decay(episode=i, N=N, decay_type=decay_type)

        # Keep going until the terminal state is reached
        while not done:

            # Employ the epsilon-greedy algorithm
            if random.uniform(0, 1) < epsilon:  # Explore
                curr_action = env.action_space.sample()
            else:  # Exploit
                curr_action = (q_table[curr_state]).argmax()

            # Take action and get new state and reward
            next_state, reward, done, truncated, info = env.step(curr_action)

            # Calculate new qvalue
            old_value = q_table[curr_state, curr_action]
            next_max = (q_table[next_state]).max()
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[curr_state, curr_action] = new_value

            # Check if penalty is made
            if reward == -100:
                penalties += 1

            # Get next observation
            curr_state = next_state

    env.close()
    print("Training finished.")
    print("Penalties obtained:", penalties)
    print("Saving Q-table...")
    np.save("q_table.npy", arr=q_table)
    print("Q-table saved.")
    print("All done.\n")

def run_simulation_table(env, beta=1.0):
    """ Evaluates policy by using it to run a simulation and calculate the reward.

    Parameters:
        env (gym environment): The gym environment.
        beta (float): The discount factor.

    Returns:
        total reward (float): Value of the total reward received under policy.
    """

    #Set up
    table = np.load('q_table.npy')
    observation, info = env.reset()
    done = False
    tot_reward = 0
    t = 0

    #Move the elf until he dies or gets the gift
    while not done:
        observation, reward, done, _, __ = env.step(table[observation].argmax())
        tot_reward += beta**t*reward
        t += 1

    return tot_reward