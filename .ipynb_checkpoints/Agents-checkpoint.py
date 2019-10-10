import numpy as np
import random
import itertools
import scipy.misc
import matplotlib.pyplot as plt

class TDagent():
    def __init__(self,params,nrows,ncols):
        self.params = params
        nstates = nrows*ncols
        self.nrows = nrows
        self.ncols = ncols
        self.nstates = nstates
        self.V = np.zeros(self.nstates)
        self.e = np.zeros(self.nstates)
        self.S = 'S'
        self.S_prime = 'S'

    def set_S(self,S):
        self.S = S
    
    def set_S_prime(self,S_prime):
        self.S_prime = S_prime
    
    def set_r(self,r):
        self.r = r
    
    def set_next_states(self,next_states):
        self.next_states = next_states
        
    def sample_action(self):
        # build Q values for each action
        Q = np.zeros(4);
        
        for c in range(4):
            next_state = int(self.next_states[c])
            Q[c] = self.V[next_state]
            
        # sample max q value, add some noise to break ties
        a = np.argmax(Q + np.random.normal(loc= 0, scale = .000001, size = Q.shape))
        
        # e-greedy noise
        if np.random.uniform() < self.params['epsilon']:
            a = np.floor(np.random.uniform(0,4))
            
        return a
    
    def update(self):
        self.e = self.update_e(self.S)
        self.V = self.update_V(self.S,self.r,self.S_prime)
        
        # doesn't return anything
    
    def update_e(self,S):   
        # update eligibility traces
        vgrad = np.eye(self.nstates)[S:S+1,:].flatten()
        e = self.params['gamma']*self.params['lam']*self.e + vgrad
        return e
    
    def update_V(self,S,r,S_prime):
        # update values 
        if S_prime == 'T':
            delta = r - self.V[S]
        else:
            delta = r + self.params['gamma']*self.V[S_prime] - self.V[S]
        V = self.V + self.params['alpha']*delta*self.e
        
        return V
    
    def render_V(self,ax):
        V_map = np.reshape(self.V, [self.nrows, self.ncols])
        ax.imshow(V_map,interpolation='none', vmin=np.min(V_map), vmax=np.max(V_map), aspect='equal')


class SRTD_agent():
    def __init__(self,params,nrows,ncols):
        self.params = params
        nstates = nrows*ncols
        self.nrows = nrows
        self.ncols = ncols
        self.nstates = nstates
        self.M = np.zeros([self.nstates,self.nstates])
        self.w = np.zeros(self.nstates)
        self.V = np.matmul(self.M,self.w)
        self.e = np.zeros(self.nstates)
        self.S = 'S'
        self.S_prime = 'S'
    
    def set_S(self,S):
        self.S = S
    
    def set_S_prime(self,S_prime):
        self.S_prime = S_prime
    
    def set_r(self,r):
        self.r = r
    
    def set_next_states(self,next_states):
        self.next_states = next_states
        
    def sample_action(self):
        
        self.V = np.matmul(self.M,self.w)
        
        # build Q values for each action
        Q = np.zeros(4);
        
        for c in range(4):
            next_state = int(self.next_states[c])
            Q[c] = self.V[next_state]
            
        # sample max q value, add some noise to break ties
        a = np.argmax(Q + np.random.normal(loc= 0, scale = .000001, size = Q.shape))
        
        # e-greedy noise
        if np.random.uniform() < self.params['epsilon']:
            a = np.floor(np.random.uniform(0,4))
            
        return a
                        
    def update(self):
        self.e = self.update_e(self.S)
        self.M = self.update_M(self.S,self.S_prime)
        self.w = self.update_w_RW(self.S,self.r)
        
    def update_e(self,S):   
        # update eligibility traces
        mgrad = np.eye(self.nstates)[S:S+1,:].flatten()
        e = self.params['gamma']*self.params['lam']*self.e + mgrad
        return e
        
    def update_M(self,S,S_prime):
        # update M using TD reulte with eligibility traces (check vectorization here)
        M = self.M;
        S_id_vec = np.eye(self.nstates)[S:S+1,:]

        if S_prime == 'T':
            delta = S_id_vec - self.M[S,:]
        else:
            delta = S_id_vec + self.params['gamma']*self.M[S_prime,:] - self.M[S,:]
            
        delta_mat = np.tile(delta,(self.nstates,1));
        
        M = self.M + self.params['alpha_sr']*delta_mat*self.e[:,None]
        
        return M
    
    def update_w_RW(self,S,r):
        # update w using rescorla wagner rule on reward
        w = self.w
        w[S] = w[S] + self.params['alpha_w']*(r - self.w[S])
        return w
    
    def render_V(self,ax):
        V_map = np.reshape(self.V, [self.nrows, self.ncols])
        ax.imshow(V_map,interpolation='none', vmin=np.min(V_map), vmax=np.max(V_map), aspect='equal')
        
    
    def render_M(self,ax):
        this_m = self.M[self.S,:];
        m_map = np.reshape(this_m,[self.nrows, self.ncols])
        ax.imshow(m_map,interpolation='none', vmin=np.min(m_map), vmax=np.max(m_map), aspect='equal')
        