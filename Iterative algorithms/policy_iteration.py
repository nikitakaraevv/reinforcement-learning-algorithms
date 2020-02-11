#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Src: https://gym.openai.com/docs/#environments

import gym
import numpy as np


class PolicyIteration:
    def __init__(self, env):
        self.env = env
        # defining the policy
        self.P = np.random.randint(self.env.nA, size=self.env.nS)

    # evaluate the policy
    def policy_evaluation(self, P, gamma, threshold=1e-10):
        V = np.zeros(self.env.nS)

        # updating the value table
        while True:
            V_new = np.copy(V)
            # loop through all the states
            for s in range(self.env.nS):
                a = P[s]
                # calculate the value for the current state with an action from the policy
                V[s] = np.sum([p*(r + gamma * V_new[s_next])
                               for p, s_next, r, __ in self.env.P[s][a]])
            if np.sum(np.fabs(V-V_new)) <= threshold:
                break
        return V

    def train(self, gamma=1.0, threshold=1e-10):
        P = self.P
        it = 0

        # policy iteration algorithm. Iterates until convergence
        print(f"Started training")

        while True:
            it += 1
            # update the value function
            V_new = self.policy_evaluation(P, gamma, threshold)

            P_new = np.zeros(self.env.nS)
            # loop through all the states
            for s in range(self.env.nS):
                P_new[s] = np.argmax([np.sum([p*(r + gamma * V_new[s_next])
                                              for p, s_next, r, __ in self.env.P[s][a]]) for a in range(self.env.nA)])
            if np.all(P == P_new):
                print(
                    f"Policy iteration algorithm converged after {it} iterations")
                break
            P = np.copy(P_new)

        self.P = P

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
    vi = PolicyIteration(gym.make('FrozenLake-v0').unwrapped)
    vi.train()
    vi.test()
