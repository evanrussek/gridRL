import numpy as np
import random
import itertools
import scipy.misc
import matplotlib.pyplot as plt
import pandas as pd

class gameEnv():
    # adjust to add objects?
    def __init__(self,nrows,ncols,reward_loc, reward_mag,wall_loc, start_pos):
        self.nrows = nrows
        self.ncols = ncols
        self.start_pos = start_pos;
        self.start_state = self.pos_to_state(start_pos);
        # make reward vector
        reward_vec = np.zeros(self.nrows*self.ncols)
        for i in range(reward_loc.shape[0]):
            state = self.pos_to_state(reward_loc[i,:])
            reward_vec[state] = reward_mag[i]
        
        wall_states = np.zeros(wall_loc.shape[0])
        for i in range(len(wall_states)):
            wall_states[i] = self.pos_to_state(wall_loc[i,:])
        self.wall_states = wall_states.astype(int)
        self.wall_loc = wall_loc.astype(int)
        
        # reward in each state
        self.R = reward_vec 
        self.lookaheadmtx = self.lookaheadmtx();

        self.reset()
            
    def state_to_pos(self,state):
        return np.unravel_index(state,(self.nrows,self.ncols))
    
    def pos_to_state(self,pos):
        a = np.reshape(np.arange(self.nrows*self.ncols), [self.nrows, self.ncols])
        return a[int(pos[0]), int(pos[1])]
    
    def lookahead(self,S,direction):
        (old_y,old_x) = self.state_to_pos(S)
        new_y = old_y
        new_x = old_x
        if direction == 0 and new_y > 0: # up
            new_y -= 1
        if direction == 1 and new_y < self.nrows-1:
            new_y += 1
        if direction == 2 and new_x > 0:
            new_x -= 1
        if direction == 3 and new_x < self.ncols-1:
            new_x += 1  
    
        curr_state = self.pos_to_state(np.array([old_y,old_x]))
        next_state = self.pos_to_state(np.array([new_y,new_x]))
            
        
        if (next_state in self.wall_states):
            next_state = S
            new_y = old_y
            new_x = old_x
            
        if (curr_state in self.wall_states):
            next_state = S
            new_y = old_y
            new_x = old_x
        
        return (next_state, new_y, new_x)       
    
    def moveAgent(self,direction):
        # 0 - up, 1 - down, 2 - left, 3 - right
        S = self.pos_to_state(np.array([self.agent_y, self.agent_x]))
        
        (next_state, new_y, new_x) = self.lookahead(S,direction)

        self.agent_x = new_x;
        self.agent_y = new_y;
        
    def render(self,ax):
        # draw current state of environment - include rewards and agent
        a = np.zeros([self.nrows, self.ncols])
        a[self.agent_y, self.agent_x] = -1
        R_rc = np.reshape(self.R,[self.nrows,self.ncols]);
        a[R_rc > 0] = R_rc[R_rc > 0]
        
        for i in range(self.wall_loc.shape[0]):
            a[self.wall_loc[i,0], self.wall_loc[i,1]] = -2

        ax.imshow(a,interpolation='none', vmin=np.min(a), vmax=np.max(a), aspect='equal')
        
        
    def step(self, action):
        # take action, produce next state and reward
        r = self.R[self.state]
        # trials end when you get reward!
        if r > 0:
            done = True
            self.state = 'T';
        else:
            done = False
            self.moveAgent(action)
            self.state = self.pos_to_state(np.array([self.agent_y, self.agent_x]))
        return (self.state, r, done)
    
    def reset(self):
        # reset environment (e.g. at end of episode)
        self.agent_y = self.start_pos[0];
        self.agent_x = self.start_pos[1];
        self.state = self.pos_to_state(self.start_pos);
        return self.state
    
    def lookaheadmtx(self):
        nstates = self.nrows*self.ncols
        lookaheadmtx = np.zeros([nstates,4])
        for s in range(nstates):
            for a in range(4):
                s_prime,_,_ = self.lookahead(s,a)
                lookaheadmtx[s,a] = s_prime
        return lookaheadmtx
    
    def make_Tsas(self):
        nstates = self.nrows*self.ncols
        T_sas = np.zeros([nstates,4,nstates])
        for s in np.arange(nstates):
            for a in np.arange(4):
                s_prime = int(self.lookaheadmtx[s,a])
                T_sas[s,a,s_prime] = 1
        return T_sas
    
    def make_Tss(self,pol_mtx):
        nstates = self.nrows*self.ncols
        T_ss = np.zeros([nstates,nstates])
        for s in np.arange(nstates):
            for i in np.arange(4):
                s_prime = int(self.lookaheadmtx[s,i])
                T_ss[s,s_prime] = T_ss[s,s_prime] + pol_mtx[s,i]
        return T_ss
    
    def make_SR(self,Tss,gamma):
        nstates = self.nrows*self.ncols
        M = np.linalg.inv(np.eye(nstates) - gamma*Tss)
        return M
    
    def render_mtx(self,mtx,ax):
        ax.imshow(mtx,interpolation='none', vmin=np.min(mtx), vmax=np.max(mtx), aspect='equal')
    
    def render_mtx_row(self,mtx,row,ax):
        this_r = mtx[row,:];
        r_map = np.reshape(this_r,[self.nrows, self.ncols])
        ax.imshow(r_map,interpolation='none', vmin=np.min(r_map), vmax=np.max(r_map), aspect='equal')
        
    def make_csv(self,mtx,name):
        df = pd.DataFrame(mtx)
        df.to_csv(name)

        

        
    


    
                
                
                
                
        

