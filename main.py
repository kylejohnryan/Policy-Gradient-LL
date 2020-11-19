import gym
import numpy as np
from agent_functionality import Agent

if __name__ == '__main__':
    agent = Agent(alpha = 0.0005, gamma = 0.99, n_actions = 4)
    
    env = gym.make('LunarLander-v2')
    score_history = []

    n_episodes = 2000

    for i in range(n_episodes):
        done = False
        score = 0
        observation = env.reset()

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.store_transition(observation, action, reward)
            observation = observation_
            score += reward
        score_history.append(score)

        agent.learn()

        avg_score = np.mean(score_history[-100:])
        print('Episode: ', i, 'Score: %.1f' % score, 'Average Score: %.1f' % avg_score)

    