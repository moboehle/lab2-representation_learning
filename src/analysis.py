import os
import tensorflow as tf
from utils import load_me
from utils import save_me
from network import Network
from network import network 
import numpy as np
import sys
from joblib import Parallel, delayed
tf.logging.set_verbosity("WARN")

import multiprocessing




def analyze_subdir(path,subdir):
        sub_path = path+subdir+"/"
        print("starting for " + sub_path)
        world_params = load_me(path+"world_params.pkl")
        settings = load_me(path+"settings.pkl")
        dqn_args = load_me(path+"dqn_args.pkl")
        targets = load_me(path+"targets.pkl")
        with tf.Graph().as_default():
            dqn_agent = Network(dqn_args,settings,world_params,targets)
            activation_list,activations_var,_q_vals = dqn_agent.analyze_invariance(load_path=sub_path)
        with tf.Graph().as_default():
            dqn_agent = Network(dqn_args,settings,world_params,targets)
            games = dqn_agent.almost_greedy_play(sub_path,max_time=2500,epsilon=0.05)

      
        save_me(sub_path+"activation_list.pkl",activation_list)
        save_me(sub_path+"activations_var.pkl",activations_var)
        save_me(sub_path+"q_vals.pkl",_q_vals)
        var_mean = np.mean(activations_var[1:,1:,:])
        if subdir == ".":
            subdir = settings["num_total_updates"]
        return int(subdir), games, var_mean

if __name__ == "__main__":
    path = sys.argv[1]

    time_ = []
    games_ = []
    var_mean_ = []

    subdirs = next(os.walk(path))[1]
    subdirs.append(".")
    num_cpus = min(multiprocessing.cpu_count(),8)
    output = Parallel(n_jobs=num_cpus)(delayed(analyze_subdir)(path,subdir) for subdir in subdirs)
    time_,games_,var_mean_ = zip(*output)
        

    tmp = np.array(sorted(np.array([time_,games_,var_mean_]).T,key=lambda x : x[0]))
    
    np.savetxt(path+"time.csv",tmp[:,0])
    np.savetxt(path+"performance.csv",tmp[:,1])
    np.savetxt(path+"var_mean.csv",tmp[:,2])

    
    
    
    
    
    
    