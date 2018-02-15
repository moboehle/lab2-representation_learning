import tensorflow as tf
from utils import *
import numpy as np
from params import * 
import world
import os
from collections import deque


def network(name,x, n_out_h1=None, kernel_size_h1 = None, strides_h1 = None, padding = None,\
            actvt_fct = None, n_out_h2 = None, kernel_size_h2 = None, strides_h2 = None,\
           n_out_h3 = None, initializer = None,num_actions = None, num_targets = 1):
    
    '''Creates a network with its variables in the scope network/{name}.
       Useful for creating an online learning network and a network for target prediction.
       Returns the nodes of the network and their names within their scope. These names can
       be used for easily transferring the variable values from the online network to the predicting network.
    '''    
    
    with tf.variable_scope("network/"+name,reuse=tf.AUTO_REUSE) as scope:
        
        ### First two layers are convolutional layers.
        h_1 = tf.layers.conv2d(x, filters = n_out_h1, kernel_size = kernel_size_h1, strides = strides_h1,\
                               padding = padding, activation = actvt_fct, kernel_initializer = initializer)
        #h_2 = tf.layers.conv2d(x, filters = n_out_h2, kernel_size = kernel_size_h2, strides = strides_h2,\
        #                       padding = padding, activation = actvt_fct, kernel_initializer = initializer)
        
        
        ### Third layer is a fully connected layer. Therefore pass the flattened h_2 into the network.
        h_3 = tf.layers.dense(tf.layers.Flatten()(h_1), n_out_h3, activation=actvt_fct,kernel_initializer=initializer)
        
        ### Finally the last fully connected layer. This will give the q-value estimate for each action respectively
        #   at the state that was passed into the first layer. No acitvation function needed.
        
        q_values = [tf.layers.dense(h_3, num_actions,kernel_initializer=initializer) for _ in range(num_targets)]
            
            
        
        ### Collect all the variables that are defined within this scope.
        theta    = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope.name)
        ### Get their names within their scope: This allows for using the names to pass from
        #   online to target by just using the relative names.
        theta_names = {var.name[len(scope.name):]: var for var in theta}
        
        return q_values, theta_names,[h_1,h_3]
        
    
    
    
    
class Network:
        
    save_path = None
    range_x  = int(FRAME_DIM/2-MARGIN/2)
    range_y  = int(FRAME_DIM/2-MARGIN/2)
    
    
    def __init__(self,dqn_args,settings,world_params,targets):

        #Setting up number of actions, number of targets and so on.
        vars(self).update(settings)
        #setting up the world
        self.set_up_world_and_actions(world_params)
        
        self.settings = settings
        self.world_params = world_params
        self.targets = targets
        self.dqn_args = dqn_args
        
        self.name="DQN"
        self.create_network()
        self.set_up_training()
        
        self.shape = (self.range_x+1,self.range_y+1,dqn_args["n_out_h3"])
        self.parameter_space = np.array([(haxis,vaxis) for haxis in np.arange(1,self.range_x+1)\
                                          for vaxis in np.arange(1,self.range_y+1)])
        
        
    ################################################################################################
    
    ##########################
    ##### SETUP FUNCTIONS ####
    ##########################
    def create_network(self):

        
        initializer = tf.contrib.layers.variance_scaling_initializer()
        self.x = tf.placeholder(tf.float32, shape=[None, FRAME_DIM,FRAME_DIM,self.my_world.frame.shape[-1]])
        
        self.learner_out  , self.learner_vars, self.learner_layers   = \
                                            network("learner"  ,self.x,**self.dqn_args,\
                                                    num_actions=self.num_actions,num_targets=self.num_targets,\
                                                    initializer=initializer)
        self.predictor_out, self.predictor_vars, self.predictor_layers = \
                                            network("predictor",self.x,**self.dqn_args,\
                                                    num_actions=self.num_actions,num_targets=self.num_targets,\
                                                    initializer=initializer)

        self.update_operations = [predictor_var.assign(self.learner_vars[var_name])
                            for var_name, predictor_var in self.predictor_vars.items()]

        self.update_predictor = tf.group(*self.update_operations)

        # List of tensorflow variables in last layer if only last layer is to be trained. #
        self.last_layer_vars = [tf.trainable_variables(scope="network/learner/dense_{}/".format(i+1)) 
                                for i in range(self.num_targets)]


    
    def set_up_training(self):
            
        ## TRAINING VARIABLES ##
        with tf.variable_scope("training",reuse=tf.AUTO_REUSE):

            self.act_idx = tf.placeholder(tf.int32, shape=[None])
            self.y = tf.placeholder(tf.float32, shape=[None, 1])

            self.q_value_per_target = [tf.reduce_sum(output * tf.one_hot(self.act_idx, self.num_actions),
                                                     axis=1, keepdims=True) for output in self.learner_out]

            self.loss_per_target = [tf.clip_by_value(tf.squared_difference(self.y,q_value,"loss"),0,100) 
                                    for q_value in self.q_value_per_target]

            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            self.training_step_per_target = [self.optimizer.minimize(loss, global_step=self.global_step) 
                                             for loss in self.loss_per_target]
            self.last_layer_trainer =  [self.optimizer.minimize(loss, global_step=self.global_step,var_list=var_list) 
                                        for loss,var_list in zip(self.loss_per_target,self.last_layer_vars)]


        #### MEMORY BUFFER ####
        self.memory = [deque([],maxlen=self.max_memory) for _ in range(self.num_targets)]

    def set_up_world_and_actions(self,world_params):
        self.my_world = world.world(**world_params)
        def increase_height():
            self.my_world.change_height(1)
        def decrease_height():
            self.my_world.change_height(-1)
        def increase_width():
            self.my_world.change_width(1)
        def decrease_width():
            self.my_world.change_width(-1)
        def increase_height2():
            self.my_world.change_height(1,ellipse=1)
        def decrease_height2():
            self.my_world.change_height(-1,ellipse=1)
        def increase_width2():
            self.my_world.change_width(1,ellipse=1)
        def decrease_width2():
            self.my_world.change_width(-1,ellipse=1)
        #List of actions to use for the network.
        self.actions = [increase_height,decrease_height,increase_width,decrease_width,\
                  increase_height2,decrease_height2,increase_width2,decrease_width2]

    
    ################################################################################################
    
    #####################
    #### GREEDY PLAY ####
    #####################
    
    
    
    def almost_greedy_play(self,load_path = None, max_time = 10000,epsilon=.1):
        '''
        In order to assess performance, restore network from previous checkpoint under load_path
        and let play for max_time steps. Use small epsilon and count number of completed games.
        '''
        over = False
        game_length = 0
        finished   = 0
        game_length_over_t = []
        time = []
        t = 0 

        
        if load_path is None:
            load_path = self.load_path

        with tf.Session() as sess:
            if os.path.isfile("./"+load_path+self.name+".index"):
                new_saver = tf.train.Saver()
                new_saver.restore(sess, tf.train.latest_checkpoint(load_path))
                print("Retrieved saved parameters")
            else:
                print("WARNING: Not a valid path: "+load_path)
                return -1
            mean = 0
            for current_target in range(0,self.settings["num_targets"]):
                self.my_world.set_target(**self.targets[current_target])
                self.my_world.restart()
                state = self.my_world.get_frame()
                self.current_target = current_target
                while True:
                        t +=1


                        #TRAINING OVER? 
                        if t >= max_time:
                            
                            break


                        #Reset world if goal is reached  
                        if over:            
                            finished +=1
                            self.my_world.restart()
                            state = self.my_world.get_frame()

                        #act and observe
                        state, reward, over, action_idx = self.act_and_observe(state,epsilon)
                        self.actions[action_idx]()
                        
                mean+=finished
                finished=0
                over=False
                t=0
        mean/=self.settings["num_targets"]
        print("\nPlayed enough already. Finished {:.2f} games on average in {} time steps .".format(mean,max_time))
                
            
        return mean
    
    ################################################################################################
    
    ##################
    #### TRAINING ####
    ##################
    def train(self,last_layer_only=False, verbose = True,extra_training_time = 0):

        
        #### VARIABLES DURING GAME PLAY####

        over = True
        game_length = 0
        finished   = -1
        game_length_over_t = []
        time = []
        t = 0 
        
        if self.save_path is None:
            
            print("Please set up the save path first with save_settings.")
            return
            
        self.current_target = 0
        
        
        with tf.Session() as sess:

            #if os.path.isfile("./"+self.load_path+self.name+".index"):
            #    self.saver.restore(sess, self.load_path)
            #    print("Retrieving saved parameters")
            #else:
            #    if last_layer_only:
            #        print("WARNING: Not loading previous parameters but still only training last layer..")
            self.init.run()


            #start with online being the same as predictor network
            self.update_predictor.run()

            if verbose: print("Starting learning process - Filling memory.",end="")

            
            
            #### STARTING ACTUAL TRAINING ####
            while True:
                t +=1

                num_updates = self.global_step.eval()
                
                #TRAINING OVER? 
                if num_updates - extra_training_time >= self.num_total_updates:
                    self.saver.save(sess, self.save_path+self.name)
                    print("\nTrained enough already.")
                    break

                #Reset world if goal is reached  
                if over:            
                    finished +=1
                    self.my_world.restart()
                    state = self.my_world.get_frame()


                epsil = self.get_curr_eps(num_updates-extra_training_time)
                
                #act and observe
                next_state, reward, over, action_idx = self.act_and_observe(state,epsil)

                #Place observations in memory.
                self.memory[self.current_target].append((state, action_idx, reward, next_state, not over))
                
                state = self.my_world.get_frame()


                if np.any([len(mem) < self.training_start for mem in self.memory ]):


                    #skip first "training_start" steps until memory is filled enough
                    self.current_target+=1
                    self.current_target%=self.num_targets
                    self.my_world.set_target(**self.targets[self.current_target])
                    continue

                
                #### LEARN FROM MEMORY ####
                

                # Sample memories and use the target DQN to produce the target Q-Value
                mem_x, mem_action, mem_rewards, mem_next_state, continues=\
                sample_memories(self.memory[self.current_target],self.batch_size)

                #For target estimate use the predictor #bootstrappingFutureRewards..
                next_q_values = self.predictor_out[self.current_target].eval(feed_dict={self.x: mem_next_state})
                #Learn according to greedy policy.

                max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)

                #Target to achieve. 
                y_val = mem_rewards + continues * self.discount_rate * max_next_q_values

                #Perform training steps and calculate the loss
                if last_layer_only:
                    the_trainer = self.last_layer_trainer[self.current_target]
                else:
                    the_trainer = self.training_step_per_target[self.current_target]

                _, prediction_loss = sess.run([the_trainer,\
                                               self.loss_per_target[self.current_target]], 
                                              feed_dict={self.x: mem_x, self.act_idx: mem_action, self.y: y_val})

                
                
                self.save_switch_update(sess,num_updates)
                
                

                #Print progress!
                if verbose and num_updates % 200 == 0:
                    print("\r Update {}/{} finished games:{} width {:.1f} height {:.1f} reward {:.2f}".format(
                     num_updates, self.num_total_updates, finished,self.my_world.haxis,self.my_world.vaxis,reward), end="")

                    
    ################################################################################################                
                    
    ###########################
    ##### ANALYSIS FUNCTIONS ####
    ###########################
                    
    def analyze_invariance(self,load_path = None):
        
        '''
        In order to analyze the representations the network learns, 
        record and store the outputs for different points in state space.
        Record q_values, last hidden layer output and variance of last hidden layer over second dimension.
        '''
        if load_path is None:
            load_path = self.load_path

        with tf.Session() as sess:
     
            if os.path.isfile("./"+load_path+self.name+".index"):
                saver = tf.train.Saver()
                saver.restore(sess, tf.train.latest_checkpoint(load_path))
                #print("Retrieved saved parameters")
            else:
                print("WARNING: Not a valid path: "+load_path)
                return -1
            
            #Stride with which to go through state space
            sparse_space = 8
            
            #List of last layer activations for each point in sparse state space 
            activation_list = np.zeros((self.range_x+1,self.range_y+1,
                                        self.parameter_space[::sparse_space].shape[0],self.shape[-1]))
            #variance of the activations with respect to second dimension (square, ellipse...)
            activations_var = np.zeros(self.shape)
            #Output of the network for each point in sparse state space
            _q_vals = np.zeros((self.range_x+1,self.range_y+1,
                                       self.parameter_space[::sparse_space].shape[0],self.settings["num_targets"],
                                       self.settings["num_actions"]))
            
            
            #for each ellipse in first dimension... 
            for count,ellipse in enumerate(self.parameter_space):

                
                self.my_world.haxis = ellipse[0]
                self.my_world.vaxis = ellipse[1]
                
                self.my_world.set_frame(which="first")

                activation_list *=0

                #... calculate the activation for all other forms and use that to calc variance of activations
                for idx, ellipse2 in enumerate(self.parameter_space[::sparse_space]):

                    
                    self.my_world.haxis2 = ellipse2[0]
                    self.my_world.vaxis2 = ellipse2[1]
                    
                    self.my_world.set_frame(which="second")
                    state = self.my_world.get_frame()

                    activation_list[ellipse[0]][ellipse[1]][idx], _q_vals[ellipse[0]][ellipse[1]][idx]  =\
                        sess.run([self.predictor_layers[1],self.predictor_out], feed_dict={self.x: [state]})
                    
                    
                activations_var[ellipse[0]][ellipse[1]] = activation_list[ellipse[0]][ellipse[1]].var(axis=0)
                #print("\rDone with {0:4d} / {1:4d} points in parameter space"
                #.format(count,self.parameter_space.shape[0]),end="")

        return activation_list,activations_var,_q_vals

    
    ################################################################################################                
                    
    ###########################
    ##### Helper FUNCTIONS ####
    ###########################
    def save_settings(self,save_path, load_path = None):
    
    
        self.save_path = save_path
        self.load_path = save_path if load_path is None else load_path

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver(max_to_keep=None)

        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)


        save_me(self.save_path+"settings.pkl",self.settings)
        save_me(self.save_path+"world_params.pkl",self.world_params)
        save_me(self.save_path+"targets.pkl",self.targets)
        save_me(self.save_path+"dqn_args.pkl",self.dqn_args)
        
        
        
    def get_curr_eps(self,num_updates):
        '''
        Decay scheme for epsilon in epsilon greedy.
        '''
        return 1 if num_updates  < self.exploratory_steps else  \
                max(self.eps_min, self.eps_max - (self.eps_max-self.eps_min) * 
                    (num_updates-self.exploratory_steps)/self.eps_decay_steps)
            
    def act_and_observe(self,state,epsil):
                    
        #Perform actions according to online network and our custom epsilon greedy policy.
        q_values = self.learner_out[self.current_target].eval(feed_dict={self.x: [state]})
        action_idx = eps_greed(epsil,q_values,self.num_actions)
        #Perform action
        self.actions[action_idx]()
        #observe world...
        return self.my_world.get_frame(), self.my_world.get_reward(), self.my_world.game_over() , action_idx
            
    def save_switch_update(self,sess,num_updates):
                    
                    # Update the predicting network every once in a while 
                    if num_updates % self.learn_period == 0:
                        self.update_predictor.run()

                    # save tuned variables.
                    if num_updates % self.switch_period == 0:
                        self.current_target+=1
                        self.current_target%=self.num_targets
                        self.my_world.set_target(**self.targets[self.current_target])

                    if num_updates % self.save_period == 0:
                        #backing up state of network to later track progress
                        self.saver.save(sess, self.save_path+"{}/".format(self.global_step.eval()-1)+self.name)    