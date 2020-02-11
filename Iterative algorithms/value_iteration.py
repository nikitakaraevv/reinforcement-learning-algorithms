#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Src: https://gym.openai.com/docs/#environments

import gym
import numpy as np


class ValueIteration:
    def __init__(self, env):
        self.env = env
        # defining the policy
        self.P = np.zeros(self.env.nS)

    # calculate the utility of every action
    def get_expected_utility(self, V, s, gamma):
        return [np.sum([p*(r + gamma * V[s_next])
                        for p, s_next, r, __ in self.env.P[s][a]])
                for a in range(self.env.nA)]

    def train(self, gamma=1.0, eps=1e-20):
        # creating the value table
        V = np.zeros(self.env.nS)
        V_next = np.copy(V)

        # defining the expected utility that we will update further
        exp_utility = np.zeros(self.env.nA)
        delta = float('inf')
        it = 0

        # value iteration algorithm. Iterates until convergence
        print(f"Started training")
        while True:
            it += 1
            delta = 0
            V = np.copy(V_next)
            # loop through all the states
            for s in range(self.env.nS):
                exp_utility = self.get_expected_utility(V, s, gamma)
                # choosing the optimal action
                a = np.argmax(exp_utility)
                self.P[s] = a
                V_next[s] = np.max(exp_utility)
                delta = max(delta, np.abs(V_next[s]-V[s]))

            # checking the convergence condition
            if delta <= (eps*(1-gamma)/gamma):
                print(f"Value iteration algorithm converged after {it} iterations")
                break

    def test(self, episodes=100, max_iterations=100):
        # in this code we suppose that success gives us 1 for the final reward
        # and failure gives as 0 at the end of the episode
        print(f"Started testing")
        rewards = []
        for i_episode in range(episodes):
            s = self.env.reset()
            for __ in range(max_iterations):
                a = self.P[s]
                s, r, done, info = self.env.step(a)
                if done:
                    rewards.append(r)
                    break
        print(
            f"Testing is completed with {int(np.mean(rewards)*100)}% of successful attempts")
        self.env.close()


if __name__ == '__main__':
    agent = ValueIteration(gym.make('FrozenLake-v0').unwrapped)
    agent.train()
    agent.test()
