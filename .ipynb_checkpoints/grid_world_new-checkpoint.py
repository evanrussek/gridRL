import numpy as np
import numpy.random as nr
import random
import scipy.misc

import matplotlib
import matplotlib.pyplot as plt


import time


class grid_world():
    # adjust to add objects?
    def __init__(self,reward_mtx,wall_mtx, start_pos, transition_noise):
        
        self.reward_mtx = reward_mtx
        self.wall_mtx = wall_mtx
      
        self.n_rows = reward_mtx.shape[0]
        self.n_cols = reward_mtx.shape[1]
        
        self.n_states = self.n_rows*self.n_cols
        self.state_mtx = np.reshape(np.arange(self.n_rows*self.n_cols), [self.n_rows, self.n_cols])
        
        self.start_pos = start_pos;
        self.start_state = self.pos_to_state(start_pos);
        
        
        # make Rs
        Rs = np.zeros(self.n_states)
        (rew_r, rew_c) = np.nonzero(reward_mtx)
        
        for i in range(len(rew_r)):
            state = self.pos_to_state(np.array([rew_r[i], rew_c[i]]))
            Rs[state] = reward_mtx[rew_r[i], rew_c[i]]
            
        self.Rs = Rs
        self.Rsa = np.tile(self.Rs, [4,1]).T
        
        # probaility that moves don't work
        self.transition_noise = transition_noise
        
        # reward in each state
        self.lookaheadmtx = self.lookaheadmtx()

        self.reset()
        
    def state_to_pos(self,state):
        return np.unravel_index(state,(self.n_rows,self.n_cols))
    
    def pos_to_state(self,pos):
        return self.state_mtx[int(pos[0]), int(pos[1])]
    
    def lookahead(self,S,direction):
        
        (old_y,old_x) = self.state_to_pos(S)
        new_y = old_y
        new_x = old_x
        
        if direction == 0 and new_y > 0: # up
            new_y -= 1
        if direction == 1 and new_y < self.n_rows-1: # down
            new_y += 1
        if direction == 2 and new_x > 0: # left
            new_x -= 1
        if direction == 3 and new_x < self.n_cols-1: # right
            new_x += 1  
    
    
        if self.wall_mtx[new_y, new_x] == 1:
            new_y = old_y
            new_x = old_x
            
        next_state = self.pos_to_state(np.array([new_y,new_x]))
            
        return (next_state, new_y, new_x)
    
    def lookaheadmtx(self):
        lookaheadmtx = np.zeros([self.n_states,4])
        for s in range(self.n_states):
            for a in range(4):
                s_prime,_,_ = self.lookahead(s,a)
                lookaheadmtx[s,a] = s_prime
        return lookaheadmtx
    
    def make_Tsas(self):
        T_sas = np.zeros([self.n_states,4,self.n_states])
        for s in np.arange(self.n_states):
            for a in np.arange(4):
                s_prime = int(self.lookaheadmtx[s,a])
                T_sas[s,a,s_prime] = 1 - self.transition_noise
                for sp in np.arange(4):
                    T_sas[s,a,self.lookaheadmtx[s,sp].astype(int)] = T_sas[s,a,self.lookaheadmtx[s,sp].astype(int)] + self.transition_noise/4
                
        # deal with terminal states a bit, or states with rewards - these take you back to start state...
        #T_sas[self.Rs != 0, :, :] = 0
        #T_sas[self.Rs != 0, :, self.start_state] = 1
        
        return T_sas
    
    def reset(self):
        # reset environment (e.g. at end of episode)
        self.agent_y = self.start_pos[0]
        self.agent_x = self.start_pos[1]
        self.state = self.start_state
        return self.state
    
    def moveAgent(self,direction):
        # 0 - up, 1 - down, 2 - left, 3 - right
        S = self.pos_to_state(np.array([self.agent_y, self.agent_x]))
        
        (next_state, new_y, new_x) = self.lookahead(S,direction)

        self.agent_x = new_x;
        self.agent_y = new_y;
    
    def step(self, action):
        # take action, produce next state and reward
        r = self.Rs[self.state]
        # trials end when you get reward or punishment # maybe change this
        
        # potential action failure
        if nr.rand() < self.transition_noise:
            action = nr.choice(4)
        
        # for now episodes end when reward is achieved
        if r != 0:
            done = True
            self.state = self.start_state
        else:
            done = False
            self.moveAgent(action)
            self.state = self.pos_to_state(np.array([self.agent_y, self.agent_x]))
        return (self.state, r, done)
    
    def render(self,ax):
        # draw current state of environment - include rewards and agent
        a = np.zeros([self.n_rows, self.n_cols])
        #a[self.agent_y, self.agent_x] = -1
        a[self.reward_mtx > 0] = 3
        a[self.reward_mtx < 0] = -3
        a[self.wall_mtx == 1] = -2
        a[self.agent_y, self.agent_x] = 1
        
        ax.imshow(a,interpolation='none', vmin=np.min(a), vmax=np.max(a), aspect='equal')
        
        for r in np.arange(self.n_rows):
            for c in np.arange(self.n_cols):
                if self.reward_mtx[r,c] != 0:
                    text = ax.text(c,r,self.reward_mtx[r,c],ha="center", va="center", color="r"  )
                    

        ax.imshow(a,interpolation='none', vmin=np.min(a), vmax=np.max(a), aspect='equal')
        text = ax.text(self.agent_x,self.agent_y,'A',ha="center", va="center", color="w"  )
        
    def render_mtx(self,mtx,ax):
        ax.imshow(mtx,interpolation='none', vmin=np.min(mtx), vmax=np.max(mtx), aspect='equal')
    
    def render_vec(self,vec,ax):
        r_map = np.reshape(vec,[self.n_rows, self.n_cols])
        #r_map[self.wall_mtx == 1] = 0
        ax.imshow(r_map,interpolation='none', cmap = 'Greys', vmin=np.min(r_map), vmax=np.max(r_map), aspect='equal')
        
    def render_sa_mtx(self,sa_mtx,ax, over_im = False):
        # plot function over actions as colored quiver arrows
        X_mtx = np.zeros([self.n_rows,self.n_cols,4])
        Y_mtx = np.zeros([self.n_rows,self.n_cols,4])
        X_mtx[:,:,2] = -1
        X_mtx[:,:,3] = 1
        Y_mtx[:,:,0] = 1
        Y_mtx[:,:,1] = -1

        X_mtx[self.wall_mtx == 1,:] = 0
        Y_mtx[self.wall_mtx == 1,:] = 0
        X_mtx[self.reward_mtx == 1,:] = 0
        Y_mtx[self.reward_mtx == 1,:] = 0

        for i in range(4):
            X,Y = np.meshgrid(np.arange(self.n_cols),np.arange(self.n_rows))
            dir_mtx = sa_mtx[:,i]
            dir_map =  np.reshape(dir_mtx,[self.n_rows, self.n_cols])

            if over_im:
                ax.quiver(X,Y,X_mtx[:,:,i],Y_mtx[:,:,i],dir_map, cmap = 'Reds',width=.005,linewidth=.005, scale_units = 'dots', scale = .07)
            else:
                ax.quiver(X,np.flipud(Y),X_mtx[:,:,i],Y_mtx[:,:,i],dir_map, cmap = 'inferno',width=.005,linewidth=.005, scale_units = 'dots', scale = .07)

        ax.set_xticks(np.arange(-.5, self.n_cols, 1), minor=True);
        ax.set_yticks(np.arange(-.5, self.n_rows, 1), minor=True);
        ax.grid(which='minor', color='k', linestyle='-', linewidth=2)

