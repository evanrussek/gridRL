
import numpy as np
import numpy.random as nr
import itertools
import scipy.misc
import matplotlib.pyplot as plt
import pandas as pd


class MDP():
    def __init__(self,nstates,nactions,Rsa, Tsas,terminal_states, start_state = 0):
        self.nstates = nstates
        self.nactions = nactions
        self.Rsa = Rsa;
        self.Tsas = Tsas
        self.terminal_states = terminal_states
        self.start_state = start_state
        self.reset(start_state)
        
    def reset(self, state):
        # reset environment (e.g. at end of episode)
        self.state = state
        return self.state
        
    def step(self, action):
        # take action, produce next state and reward
        
        done = False
        s = self.state
        s_prime = nr.choice(np.arange(self.nstates),p=self.Tsas[s,action,:])
        r = self.Rsa[s,action]
        if np.isin(s_prime,self.terminal_states):
            done = True
        
        self.state = s_prime
        nchoices = self.nactions[self.state]
        
        return(self.state, r, nchoices, done)
