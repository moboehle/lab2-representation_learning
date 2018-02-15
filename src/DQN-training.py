import os
import random
from copy import copy
from collections import deque
import numpy as np
import sys
import pickle
from network import *
from params import * 
from utils import *
import world 
tf.logging.set_verbosity("WARN")

import tensorflow as tf


if __name__ == "__main__":
    
    setter = {}
    world_updator = {}
    for arg in sys.argv[1:]:

        if arg.startswith("-I"):
            key, val = arg[2:].split("=")
            setter[key] = int(val)
        if arg.startswith("-S"):
            key, val = arg[2:].split("=")
            setter[key] = val
        if arg.startswith("-B"):
            key, val = arg[2:].split("=")
            setter[key] = bool(int(val))
            
        if arg.startswith("--I"):
            key, val = arg[3:].split("=")
            world_updator[key] = int(val)
        if arg.startswith("--S"):
            key, val = arg[3:].split("=")
            world_updator[key] = val
        if arg.startswith("--B"):
            key, val = arg[3:].split("=")
            world_updator[key] = bool(int(val))

         
        
    world_params = {"game_mode":'constant_reward',
                    "goal_reward" : 50,
                    "precision": 3,
                    "ignore_first": False,
                    "ignore_second" : True,
                    "first_in_second" : True
                   }

    dqn_args   = {"n_out_h1" : 16, 
                  "kernel_size_h1":(8,8), 
                  "strides_h1" : 4,
                  "padding" : "SAME",
                  "actvt_fct" : tf.nn.relu,
                  "n_out_h2" : 32,
                  "kernel_size_h2":(4,4),
                  "strides_h2" : 2,
                  "n_out_h3" : 128}

    settings = {"num_actions" : 8,
                "num_targets" : 9,
                "learning_rate"  : 5*10**(-4),
                "batch_size"     : 32 ,
                "training_start" : 128  ,
                "discount_rate"  : .9,
                "num_total_updates" : 500000,
                "learn_period" : 20,
                "eps_min" : 0.1,
                "eps_max" : 1.0,
                "exploratory_fraction" : 1/10,
                "max_memory" : 1500,
                "switch_period" : 500,
                "save_period" : 10000
               }
    
    settings.update(setter)
    world_params.update(world_updator)
    

    locals().update(settings)
    exploratory_steps = int(num_total_updates*exploratory_fraction)
    eps_decay_steps = num_total_updates-exploratory_steps


    settings["exploratory_steps"] = exploratory_steps
    settings["eps_decay_steps"] = eps_decay_steps

    
    targets = [{"vaxis" : vaxis, "haxis" : haxis,"vaxis2":vaxis2,"haxis2":haxis2} for vaxis,haxis,vaxis2,haxis2 in \
               np.random.randint(0,30,(settings["num_targets"],4))]
    save_path = "data/"
    
    if world_params["both_in_first"]:
        save_path = save_path + "square/" 
    
    elif world_params["first_in_second"]:
        save_path = save_path + "double-ellipse/"
       
    else:
        save_path = save_path + "disjoint/"
        
     
    if num_targets> 1:
        save_path = save_path + "multi_target/"
    else: 
        save_path = save_path + "single_target/"        
        
    
    with tf.Graph().as_default():
        netwk = Network(dqn_args,settings,world_params,targets)
        netwk.save_settings(save_path)
        netwk.train()

    