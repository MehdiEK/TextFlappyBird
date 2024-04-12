import os, sys
import gymnasium as gym
import time
import numpy as np

import text_flappy_bird_gym

from agent import SarsaAgent, MCAgent, SarsaLambdaAgent


def train_sarsa_function(env, agent, num_episode=1000, verbose=False):
    """
    """
    cum_rewards = [0] * num_episode

    # generate episode to train agent
    for episode in range(num_episode):
        # initiliaze nb of steps lists containing data
        step = 0
        done = False
        actions = []
        rewards = []
        observations = []

        # first observation 
        obs, _ = env.reset()
        observations.append(obs)

        # start taking actions ! 
        while step < 1000 and not done:
            if step == 0:  # first step
                action = agent.agent_start(obs)
            else:
                action = agent.agent_step(reward, obs)
            actions.append(action)
            
            # get response of the env
            obs, reward, done, _, info = env.step(action)
            # print(obs, reward, done, _, info)
            # store info 
            rewards.append(reward)

            # update nb of steps
            step += 1
        
            # keep last obs 
            observations.append(obs)

        # last update of the agent
        agent.agent_end(reward)

        # store cum reward
        cum_rewards[episode] = np.sum(rewards)

        if verbose: 
            print("\nEpisode: ", episode)
            print("Sum of rewards: ", cum_rewards[episode])
    
    return agent, cum_rewards


def train_mc_agent(env, agent, num_episode=1000, verbose=False):
    """
    """
    cum_rewards = [0] * num_episode

    # generate episode to train agent
    for episode in range(num_episode):
        # initiliaze nb of steps lists containing data
        step = 0
        done = False
        actions = []
        rewards = []
        observations = []

        # first observation 
        obs, _ = env.reset()
        observations.append(obs)

        # start taking actions ! 
        while step < 1000 and not done:
            action = agent.policy(obs)
            actions.append(action)
            
            # get response of the env
            obs, reward, done, _, info = env.step(action)
            # print(obs, reward, done, _, info)
            # store info 
            rewards.append(reward)

            # update nb of steps
            step += 1
        
            # keep last obs 
            observations.append(obs)

        # last update of the agent
        agent.policy_evaluation(observations, actions, rewards)
        
        # store cumulative reward
        cum_rewards[episode] = np.sum(rewards)

        if verbose:
            print("\nEpisode: ", episode)
            print("Sum of rewards: ", cum_rewards[episode])
    
    return agent, cum_rewards


def inference(env, agent, epsilon, show=True):
    """
    No more exploration in this mode, the goal is to reach the best score
    possible. 

    :params env: text_flappy_bird_env_gym
        Environment
    :params agent: agent type
        Agent for interacting with env 
    :params epsilon: float 
        Set to 0 if you want greedy agent
    """
    # initialization of reward store
    rewards = []
    # initialize the env and get first obs
    obs, _ = env.reset()

    # let's play
    while True:
    
        action = agent.inference(obs, epsilon)

        # Appy action and return new observation of the environment
        obs, reward, done, _, _ = env.step(action)
        rewards.append(reward)

        if show: 
            # Render the game
            os.system("clear")
            sys.stdout.write(env.render())
            time.sleep(0.1) # FPS

        # If player is dead break
        if done:
            break
    
    if show: 
        env.close()

    return np.sum(rewards)


def main():
    """
    """
    # initilization of the env
    env = gym.make(
        'TextFlappyBird-v0', 
        height=15, 
        width=20, 
        pipe_gap=4
        )
    """
    # create the agent
    mc_agent = MCAgent(
        epsilon=0.1, 
        step_size=0.5, 
        discount=.9, 
        num_actions=2    
    )

    # train sarsa agent
    mc_agent, _ = train_mc_agent(
        env=env, 
        agent=mc_agent, 
        num_episode=5000
    )

    # play real game
    inference(env, mc_agent)
    """
    # create the agent
    sarsa_agent = SarsaAgent(
        epsilon=0.1, 
        step_size=0.5,  # set to 0.5 please 
        discount=1., 
        num_actions=2    
    )

    # train sarsa agent
    sarsa_agent, _ = train_sarsa_function(
        env=env, 
        agent=sarsa_agent, 
        num_episode=5000
    )

    new_env = gym.make(
        'TextFlappyBird-v0', 
        height=20, 
        width=15, 
        pipe_gap=3
        )

    # play real game
    inference(env, sarsa_agent, epsilon=0.)


if __name__ == '__main__':
    main()
    
    