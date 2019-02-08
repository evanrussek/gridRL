class gameEnv():
    # adjust to add objects?
    def __init__(self,rewards,walls, start_pos):
        
        self.rewards = rewards
        self.walls = walls
        
        self.nrows = rewards.shape[0]
        self.ncols = rewards.shape[1]
        self.start_pos = start_pos;
        self.start_state = self.pos_to_state(start_pos);
        
        # make reward vector
        reward_vec = np.zeros(self.nrows*self.ncols)
        nz_rewards = np.nonzeros(rewards)
        for i in range(len(nz_rewards[0])):
            state = self.pos_to_state(reward_loc[i,:])
            reward_vec[state] = reward_mag[i]
        
        
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
    
    
        if self.walls[new_y, new_x] == 1:
            new_y = old_y
            new_x = old_x
            
        next_state = self.pos_to_state(np.array([new_y,new_x]))
            
        return (next_state, new_y, new_x)
    
    def lookaheadmtx(self):
        nstates = self.nrows*self.ncols
        lookaheadmtx = np.zeros([nstates,4])
        for s in range(nstates):
            for a in range(4):
                s_prime,_,_ = self.lookahead(s,a)
                lookaheadmtx[s,a] = s_prime
        return lookaheadmtx
    
    