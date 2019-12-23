import numpy as np
import numpy.random as nr
import scipy.misc
from IPython.display import clear_output
from scipy.special import gammaln

def run_nsteps(pol_vec0,theta0,env,nsteps):
    # run nsteps of metropolis hastings

    theta = np.copy(theta0)

    pol_vec = np.copy(pol_vec0)
    value_curr = value(env,pol_vec)

    pol_trace = np.zeros([nsteps,len(pol_vec)])
    theta_trace = np.zeros([nsteps,len(theta)])
    v_trace = np.zeros([nsteps])

    for step in np.arange(nsteps):
        if step % 20 == 0:
            clear_output()
            print('step: ', step)
        pol_trace[step,:] = pol_vec
        theta_trace[step,:] = theta
        v_trace[step] = value_curr
        (pol_vec, logp_curr, value_curr)  = policy_step(pol_vec,theta,env,v_trace[step])
        theta = latent_step(pol_vec,theta, value_curr)
  
    return (pol_trace,theta_trace,v_trace)


def sim_episode(env, policy_vec, max_step, show):
    d = False
    j = 0
    S = env.reset()
    while j < max_step:   
        if show == 1:
            # draw environment and pause
            env.render(ax)
            plt.pause(.02)        
        # increase counter
        j += 1
        # sample action given by pi for state S
        a = policy_vec[S]
        # take action A, observe s1, r, terminal?
        S_prime,r,d = env.step(a)
        # update S
        S = S_prime;
        if d == True:
            break
    return j

def dir_logp(theta):
    # log_p(theta|dir(alpha))
    alpha = np.array([5,5,5,5])
    return np.sum((alpha-1)*np.log(theta) - gammaln(alpha)) + gammaln(np.sum(alpha))

def cat_logp(theta,p_vec):
    # log_p(p_vec|categorical(theta))
    return np.sum(np.log(theta[p_vec.astype(int)]))

def value(env,p_vec):
    # evaluate p_vec for env
    return -1*sim_episode(env,p_vec,200,0)

def logp(val, p_vec, theta):
    # log posterior of value, p_vec and theta
    logp = val + cat_logp(theta,p_vec) + dir_logp(theta)
    return logp

def latent_step(pol_vec,theta0, value_curr):
    # step theta (propose and accept or reject)
    theta = np.copy(theta0)
    logp_curr = logp(value_curr,pol_vec,theta)
    curr_val, theta = theta, prop_theta(theta)
    logp_prop = logp(value_curr,pol_vec,theta)
    theta,accepted = metrop_select(logp_prop - logp_curr, theta, curr_val)
    if accepted:
        logp_curr = logp_prop
    
    return theta

def policy_step(pol_vec0,theta,env,value_curr):
    # step p_vec (gibbs metropolis)
    state_list = np.arange(pol_vec0.shape[0])
    nr.shuffle(state_list)
        
    pol_vec = np.copy(pol_vec0)
    
    logp_curr = logp(value_curr,pol_vec,theta) 
    
    nchoices = 4
    
    for s in state_list:
        curr_choice, pol_vec[s] = pol_vec[s], sample_except(nchoices, pol_vec[s])
        value_prop = value(env,pol_vec)
        logp_prop = logp(value_prop,pol_vec,theta)
        pol_vec[s], accepted = metrop_select(logp_prop - logp_curr, pol_vec[s], curr_choice)
        if accepted:
            logp_curr = logp_prop
            value_curr = value_prop
    
    return pol_vec, logp_curr, value_curr

def prop_theta(theta0): 
    # propose new theta
    scale = 30
    prop_theta = nr.dirichlet(scale*theta0)
    return prop_theta

def metrop_select(mr, q, q0):
    # accept or reject according to metrop hasting rule
    if np.isfinite(mr) and np.log(nr.uniform()) < mr:
        return q, True
    else:
        return q0, False

def sample_except(limit, excluded):
    # draw categorical sample less than limit, not picking exclded
    candidate = nr.choice(limit - 1)
    if candidate >= excluded:
        candidate += 1
    return candidate

def sample_policy(theta,nstates):
    pol_vec = nr.choice(4,size=nstates,p=theta)
    return(pol_vec)

def sample_theta():
    alpha = np.array([10,10,10,10])
    return nr.dirichlet(alpha)

