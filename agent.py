"""
Agent for playing flappy bird game. 
"""
import numpy as np 


class SarsaAgent(object):
    def __init__(self, epsilon, step_size, discount, num_actions):
        """
        Initialization of the SARSA agent. 

        :params epsilon: float
            Parameter for epsilon greedy agent setup. 
        :params step_size: float
            Leanring rate of the sarsa agent. 
        :params discount: float
            Discount for future rewards. 
        :params num_actions: int 
            Nb of possible actions. 
        """
        # Define intra class parameters 
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.step_size = step_size
        self.discount = discount
        
        # create q values table
        self.q = {}  # initialized to 0 (see get method)
        
    def agent_start(self, obs):
        """
        First action that the agent can take. 

        :params obs: tuple
            Observation of initial state by the agent. 
        
        :return action (int)
            First action taken by the agent. 
        """
        # Take action following epsilon-greedy agent policy
        current_q = [self.q.get((obs, a), 0) 
                     for a in range(self.num_actions)]
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.num_actions)
        else:
            action = self.argmax(current_q)

        # update states and action 
        self.prev_state = obs
        self.prev_action = action

        return action
    
    def agent_step(self, reward, obs):
        """
        Given an obs and a reward, update the q-values table and 
        return a nexw action in its env.

        :params reward: float 
            Reward collected by the agent. 
        :params obs: tuple
            Observation of last state

        :return action (int)
            Next action taken by the agent following the espilon-greedy 
            agent policy. 
        """
        # Choose action using epsilon greedy.
        current_q = [self.q.get((obs, a), 0) 
                     for a in range(self.num_actions)]
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.num_actions)
        else:
            action = self.argmax(current_q)
        
        ### Update step
    
        # get proba of each action 
        policy = np.zeros(self.num_actions) + self.epsilon / self.num_actions
        policy[self.argmax(current_q)] += (1-self.epsilon)

        # compute last and next q_value + update term 
        last_q = self.q.get((self.prev_state, self.prev_action), 0)
        next_q = self.q.get((obs, action), 0)
        update = self.step_size * (reward + self.discount * next_q - last_q)

        # perform the update
        self.q[(self.prev_state, self.prev_action)] = last_q + update
                
        # update action and state
        self.prev_state = obs
        self.prev_action = action
        return action
    
    def agent_end(self, reward):
        """
        Perform last update of the agent when it encouters a terminal 
        state. 

        :params reward: float
            Last reward obtained for reaching terminal state.
        """
        last_q = self.q.get((self.prev_state, self.prev_action), 0)
        update = self.step_size * (reward - last_q )
        self.q[self.prev_state, self.prev_action] = last_q + update   
        
    def argmax(self, q_values):
        """
        Argmax function to choose next action. In case of equal top q-values,
        sampel one random 

        :params q_values: list
            List of q_values for each action. 
        """
        top = float("-inf")
        ties = []

        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []

            if q_values[i] == top:
                ties.append(i)

        return np.random.choice(ties)
    
    def inference(self, obs, epsilon=0.):
        """
        Function for inference. No more exploration if you choose epsilon=0, 
        let's explose your score (if the agent is well trained...)

        :params obs: tuple
            Last observation of the agent. 
        :params epislon: float, default=0.
            If set to 0, is greedy.
        
        :return action (int)
            Next action to take. 
        """
        # Choose action using epsilon greedy.
        current_q = [self.q.get((obs, a), 0) 
                     for a in range(self.num_actions)]
        if np.random.random() < epsilon:
            action = np.random.choice(self.num_actions)
        else:
            action = self.argmax(current_q)
        
        return action


class MCAgent(object):
    def __init__(self, epsilon, step_size, discount, num_actions):
        """
        Initialization of the SARSA agent. 

        :params epsilon: float
            Parameter for epsilon greedy agent setup. 
        :params step_size: float
            Leanring rate of the sarsa agent. 
        :params discount: float
            Discount for future rewards. 
        :params num_actions: int 
            Nb of possible actions. 
        """
        # Define intra class parameters 
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.step_size = step_size
        self.discount = discount
        
        # create q values table and count table 
        self.q = {}  # initialized to 0 (see get method)
        self.count = {}

    def policy(self, obs):
        """
        Policy of the agent for exploring its environment is epsilno greedy.  

        :params obs: 

        :return action (int)
        """
        current_q = [self.q.get((obs, a), 0) 
                     for a in range(self.num_actions)]
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.num_actions)
        else:
            action = self.argmax(current_q)

        return action 

    def policy_evaluation(self, obs, actions, rewards):
        """
        """
        episode_length = len(rewards)
        disc = [self.discount ** i for i in range(episode_length)]
        for i in range(episode_length):

            # get data 
            prev_obs = obs[i]
            action = actions[i]
            cum_reward = np.dot(rewards[i:], disc[:episode_length-i])

            # get former information on q value
            count = self.count.get((prev_obs, action), 0)
            q_val = self.q.get((prev_obs, action), 0)

            # update q_value
            update = (q_val*count + cum_reward) / (count+1)
            self.q[(prev_obs, action)] = update

            # update count
            self.count[(prev_obs, action)] = count + 1
        
    def argmax(self, q_values):
        """
        Argmaxx function to choose next action. In case of equal top q-values,
        sampel one random 

        :params q_values: list
            List of q_values for each action. 
        """
        top = float("-inf")
        ties = []

        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []

            if q_values[i] == top:
                ties.append(i)

        return np.random.choice(ties)
    
    def inference(self, obs, epsilon=0.):
        """
        Function for inference. No more exploration if you choose epsilon=0, 
        let's explose your score (if the agent is well trained...)

        :params obs: tuple
            Last observation of the agent. 
        :params epislon: float, default=0.
            If set to 0, is greedy.
        
        :return action (int)
            Next action to take. 
        """
        # Choose action using epsilon greedy.
        current_q = [self.q.get((obs, a), 0) 
                     for a in range(self.num_actions)]
        if np.random.random() < epsilon:
            action = np.random.choice(self.num_actions)
        else:
            action = self.argmax(current_q)
        
        return action


class SarsaLambdaAgent(object):
    def __init__(self, epsilon, step_size, discount, num_actions, lam=0.9):
        """
        Initialization of the SARSA agent. 

        :params epsilon: float
            Parameter for epsilon greedy agent setup. 
        :params step_size: float
            Leanring rate of the sarsa agent. 
        :params discount: float
            Discount for future rewards. 
        :params num_actions: int 
            Nb of possible actions. 
        """
        # Define intra class parameters 
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.step_size = step_size
        self.discount = discount
        self.lamb = lam
        
        # create q values table and elligibility table
        self.q = {}  # initialized to 0 (see get method)
        self.e = {}  # initialized to 0 (see get method)
        
    def agent_start(self, obs):
        """
        First action that the agent can take. 

        :params obs: tuple
            Observation of initial state by the agent. 
        
        :return action (int)
            First action taken by the agent. 
        """
        # Take action following epsilon-greedy agent policy
        current_q = [self.q.get((obs, a), 0) 
                     for a in range(self.num_actions)]
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.num_actions)
        else:
            action = self.argmax(current_q)

        # update states and action 
        self.prev_state = obs
        self.prev_action = action

        return action
    
    def agent_step(self, reward, obs):
        """
        Given an obs and a reward, update the q-values table and 
        return a nexw action in its env.

        :params reward: float 
            Reward collected by the agent. 
        :params obs: tuple
            Observation of last state

        :return action (int)
            Next action taken by the agent following the espilon-greedy 
            agent policy. 
        """
        # Choose action using epsilon greedy.
        current_q = [self.q.get((obs, a), 0) 
                     for a in range(self.num_actions)]
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.num_actions)
        else:
            action = self.argmax(current_q)
        
        ### Update step
        q_old = self.q.get((self.prev_state, self.prev_action), 0)
        q_new = self.q.get((obs, action), 0)
        delta = reward + self.discount * q_new - q_old
        self.e[(self.prev_state, self.prev_action)] = self.e.get((self.prev_state, self.prev_action), 0) +1
        
        for key in self.e.keys():
            self.e[key] = self.discount * self.lamb * self.e[key]
            self.q[key] = self.q.get(key, 0) + self.step_size * delta * self.e[key]

        # update action and state
        self.prev_state = obs
        self.prev_action = action

        return action
    
    def agent_end(self, reward):
        """
        Perform last update of the agent when it encouters a terminal 
        state. 

        :params reward: float
            Last reward obtained for reaching terminal state.
        """
        q_old = self.q.get((self.prev_state, self.prev_action), 0)
        q_new = 0  # there's no next state
        delta = reward + self.discount * q_new - q_old
        self.e[(self.prev_state, self.prev_action)] = self.e.get((self.prev_state, self.prev_action), 0) +1
        
        for key in self.e.keys():
            self.q[key] = self.q.get(key, 0) + self.step_size * delta * self.e[key]
        
        self.e = {}
        
    def argmax(self, q_values):
        """
        Argmaxx function to choose next action. In case of equal top q-values,
        sampel one random 

        :params q_values: list
            List of q_values for each action. 
        """
        top = float("-inf")
        ties = []

        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []

            if q_values[i] == top:
                ties.append(i)

        return np.random.choice(ties)
    
    def inference(self, obs, epsilon=0.):
        """
        Function for inference. No more exploration if you choose epsilon=0, 
        let's explose your score (if the agent is well trained...)

        :params obs: tuple
            Last observation of the agent. 
        :params epislon: float, default=0.
            If set to 0, is greedy.
        
        :return action (int)
            Next action to take. 
        """
        # Choose action using epsilon greedy.
        current_q = [self.q.get((obs, a), 0) 
                     for a in range(self.num_actions)]
        if np.random.random() < epsilon:
            action = np.random.choice(self.num_actions)
        else:
            action = self.argmax(current_q)
        
        return action