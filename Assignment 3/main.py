import numpy as np
import matplotlib.pyplot as plt
import sys

###############################  TASK 1  ####################################################

class WindyGridWorld():
    def __init__(self, wind, stochastic = False, rows = 7, columns = 10, start = (3, 0), end = (3, 7)):
        self.rows = rows
        self.columns = columns
        self.stochastic = stochastic

        self.start = start
        self.end = end

        self.wind = wind

        self.award = 0
        self.penalty = -1

    def reward(self, state, action):
        if self.stochastic:
            wind_strength = self.wind[state[1]] + np.random.randint(-1, 2)
        else:
            wind_strength = self.wind[state[1]]

        # Action scheme:   4 0 5
        #                  3 . 1
        #                  7 2 6

        if action == 0:
            next_state = (min(max(state[0] - wind_strength - 1, 0), self.rows - 1), state[1])

        elif action == 1:
            next_state = (min(max(state[0] - wind_strength, 0), self.rows - 1), min(max(state[1] + 1, 0), self.columns - 1))
        
        elif action == 2:
            next_state = (min(max(state[0] - wind_strength + 1, 0), self.rows - 1), state[1])
        
        elif action == 3:
            next_state = (min(max(state[0] - wind_strength, 0), self.rows - 1), min(max(state[1] - 1, 0), self.columns - 1))
        
        elif action == 4:
            next_state = (min(max(state[0] - wind_strength - 1, 0), self.rows - 1), min(max(state[1] - 1, 0), self.columns - 1))

        elif action == 5:
            next_state = (min(max(state[0] - wind_strength - 1, 0), self.rows - 1), min(max(state[1] + 1, 0), self.columns - 1))

        elif action == 6:
            next_state = (min(max(state[0] - wind_strength + 1, 0), self.rows - 1), min(max(state[1] + 1, 0), self.columns - 1))

        else:
            next_state = (min(max(state[0] - wind_strength + 1, 0), self.rows - 1), min(max(state[1] - 1, 0), self.columns - 1))

        if (next_state == self.end):
            return next_state, self.award
        
        else:
            return next_state, self.penalty

class Agent():
    def __init__(self, epsilon = 0.1, alpha = 0.5, num_actions = 8, gamma = 1, rows = 7, columns = 10):
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_actions = num_actions
        self.gamma = gamma

        self.rows = rows
        self.columns = columns

        self.Q = np.zeros((rows, columns, num_actions), dtype = float)

    def epsilon_greedy(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.num_actions)
            
        else:
            action = np.amax(np.argmax(self.Q[state[0]][state[1]]))
        
        return action
    
    def SARSA(self, S, A, R, S_dash, A_dash):
        self.Q[S[0]][S[1]][A] += self.alpha*(R + self.gamma*self.Q[S_dash[0]][S_dash[1]][A_dash] - self.Q[S[0]][S[1]][A])

    def QLearning(self, S, A, R, S_dash):
        a_max = np.amax(np.argmax(self.Q[S_dash[0]][S_dash[1]]))
        self.Q[S[0]][S[1]][A] += self.alpha*(R + self.gamma*self.Q[S_dash[0]][S_dash[1]][a_max] - self.Q[S[0]][S[1]][A])
    
    def ExpectedSARSA(self, S, A, R, S_dash):
        a_max = np.amax(np.argmax(self.Q[S_dash[0]][S_dash[1]]))
        prob = np.ones(num_actions, dtype=float) * (self.epsilon/self.num_actions)
        prob[a_max] += (1 - self.epsilon)
        expectation = np.dot(prob, self.Q[S_dash[0]][S_dash[1]])
        self.Q[S[0]][S[1]][A] += self.alpha*(R + self.gamma*expectation - self.Q[S[0]][S[1]][A])

def runner(alpha, epsilon, episodes, num_trials, start, end, wind, num_actions, stochastic, agent_type = 'sarsa'):
    episode_steps = np.zeros(episodes, dtype=float)
    
    for seed in range(num_trials):
        
        np.random.seed(seed)
    
        environment = WindyGridWorld(wind, stochastic=stochastic)
        agent = Agent(num_actions=num_actions)

        for episode in range(episodes):
            S = start
            A = agent.epsilon_greedy(S)
            t = 0

            while(1):
                
                S_dash, R = environment.reward(S, A)
                A_dash = agent.epsilon_greedy(S_dash)

                if agent_type == 'e-sarsa':
                    agent.ExpectedSARSA(S, A, R, S_dash)

                elif agent_type == 'Qlearning':
                    agent.QLearning(S, A, R, S_dash)

                else:
                    agent.SARSA(S, A, R, S_dash, A_dash)
                
                S = S_dash
                A = A_dash
                
                t += 1

                if S == end:
                    break
            
            episode_steps[episode] += t

    episode_steps = np.cumsum(episode_steps)
    episode_steps /= num_trials
    return episode_steps

if __name__ == '__main__':
    alpha = 0.5
    epsilon = 0.1
    episodes = 170
    num_trials = 10

    start = (3, 0)
    end = (3, 7)

    wind = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 1, 0])

    ###############################  TASK 2  ####################################################

    num_actions = 4
    stochastic = False

    steps1 = runner(alpha, epsilon, episodes, num_trials, start, end, wind, num_actions, stochastic)

    x1 = np.zeros(len(steps1)+1)
    x1[1:] = steps1
    y1 = np.zeros(episodes+1)
    y1[1:] = [i for i in range(episodes)]
    
    plt.figure()
    plt.plot(x1, y1)
    plt.title("Task 2 (SARSA(0) agent), learning rate={}, epsilon={}".format(alpha, epsilon))
    plt.xlabel("Time Steps")
    plt.ylabel("Episodes")
    plt.show()
    
    ###############################  TASK 3  ####################################################

    num_actions = 8
    
    steps2 = runner(alpha, epsilon, episodes, num_trials, start, end, wind, num_actions, stochastic)
    
    x2 = np.zeros(len(steps2)+1)
    x2[1:] = steps2
    y2 = np.zeros(episodes+1)
    y2[1:] = [i for i in range(episodes)]
    
    plt.figure()
    plt.plot(x2, y2)
    plt.title("Task 3 (SARSA(0) agent + King's moves), lr={}, epsilon={}".format(alpha, epsilon))
    plt.xlabel("Time Steps")
    plt.ylabel("Episodes")
    plt.show()
    
    ###############################  TASK 4  ####################################################

    stochastic = True

    steps3 = runner(alpha, epsilon, episodes, num_trials, start, end, wind, num_actions, stochastic)

    x3 = np.zeros(len(steps3)+1)
    x3[1:] = steps3
    y3 = np.zeros(episodes+1)
    y3[1:] = [i for i in range(episodes)]

    plt.figure()
    plt.plot(x3, y3)
    plt.title("Task 4 (SARSA + King's moves + stochastic wind), lr={}, epsilon={}".format(alpha, epsilon))
    plt.xlabel("Time Steps")
    plt.ylabel("Episodes")
    plt.show()
    
    plt.figure()
    plt.plot(x1, y1, label="Task 1")
    plt.plot(x2, y2, label="Task 2")
    plt.plot(x3, y3, label="Task 3")
    plt.title("Combined plots for task 2-4 (lr={}, epsilon={})".format(alpha, epsilon))
    plt.legend(loc='lower right')
    plt.xlabel("Time Steps")
    plt.ylabel("Episodes")
    plt.show()

    ###############################  TASK 5  ####################################################
    
    num_actions = 4
    stochastic = False
    episodes = 700

    steps4 = runner(alpha, epsilon, episodes, num_trials, start, end, wind, num_actions, stochastic)
    steps5 = runner(alpha, epsilon, episodes, num_trials, start, end, wind, num_actions, stochastic, agent_type = 'Qlearning')
    steps6 = runner(alpha, epsilon, episodes, num_trials, start, end, wind, num_actions, stochastic, agent_type = 'e-sarsa')

    x4 = np.zeros(len(steps4)+1)
    x4[1:] = steps4
    y4 = np.zeros(episodes+1)
    y4[1:] = [i for i in range(episodes)]

    x5 = np.zeros(len(steps5)+1)
    x5[1:] = steps5
    y5 = np.zeros(episodes+1)
    y5[1:] = [i for i in range(episodes)]

    x6 = np.zeros(len(steps6)+1)
    x6[1:] = steps6
    y6 = np.zeros(episodes+1)
    y6[1:] = [i for i in range(episodes)]

    plt.figure()
    plt.plot(x4, y4, label="SARSA")
    plt.plot(x5, y5, label="Q-Learning")
    plt.plot(x6, y6, label="Expected SARSA")
    plt.title("Combined plots for task 5 (lr={}, epsilon={})".format(alpha, epsilon))
    plt.legend(loc='lower right')
    plt.xlabel("Time Steps")
    plt.ylabel("Episodes")
    plt.show()