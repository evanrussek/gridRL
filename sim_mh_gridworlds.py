import numpy as np
import numpy.random as nr
import random
#import itertools
import scipy.misc

import matplotlib
#matplotlib.use('qt5agg')
import matplotlib.pyplot as plt

#%matplotlib notebook
#%matplotlib qt 
#%matplotlib inline 
import time

# gridworldclass that we made
from gridworldclass import gameEnv
from IPython.display import clear_output
from scipy.special import gammaln
import mhfuncs as mh

# build environments
wall_x = np.array([]);
wall_y = np.array([]);
wall_loc = np.array([wall_y, wall_x]).T
wall_states = np.zeros(wall_loc.shape[0])

nenvs = 10
envs = []
for i in np.arange(nenvs):
    #gameEnv(nrows,ncols,reward_loc, reward_mag,wall_loc, start_pos)
    nrows = 7
    ncols = 5
    r_col = np.random.randint(0,ncols,1)[0]
    s_col = np.random.randint(0,ncols,1)[0]
    reward_loc = np.array([[0,r_col]])
    reward_mag = np.array([1])
    start_pos = np.array([nrows-1,s_col])
    envs1 = gameEnv(nrows,ncols,reward_loc, reward_mag,wall_loc, start_pos)
    envs.append(envs1)

nstates = envs[0].nrows*envs[0].ncols

nruns = 10
nsteps_per_env = 10
nsteps_per_run = nsteps_per_env*nenvs
which_env_list = np.repeat(np.arange(nenvs),nsteps_per_env)


# run simulation w/ prior learning for n runs
vt_run = np.zeros([nruns,nsteps_per_run])
pt_run = np.zeros([nruns,nsteps_per_run, nstates])
tt_run = np.zeros([nruns,nsteps_per_run,4])

for run in np.arange(nruns):
    theta0 = mh.sample_theta()
    pol_vec0 = mh.sample_policy(theta0,nstates)
    
    for j in np.arange(nenvs):
        (this_env_pol_trace,this_env_theta_trace,this_env_v_trace) = mh.run_nsteps(pol_vec0,theta0,envs[j],nsteps_per_env)
        pol_vec0 = this_env_pol_trace[-1]
        
        env_idx = (which_env_list == j)
        vt_run[run,env_idx] = this_env_v_trace
        pt_run[run,env_idx,:] = this_env_pol_trace
        tt_run[run,env_idx,:] = this_env_theta_trace
       

       


#plt.close('all')
#f,ax = plt.subplots(1)
#env.render(ax)
