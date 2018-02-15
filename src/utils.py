import numpy as np

import os
import tensorflow as tf
import random 
import pickle 




def calc_variances(sess, my_world, save_path,parameter_space,\
                             shape,layer,input_layer,conv_layer= False, change_dim = "second" ):
        
        '''
        Calculate for each node in the given layer how much it varies for each point in state space for
        one dimension (e.g. the ellipse) when the second dimension (e.g. the square) is changed.
        '''
        
        static_dim = "first" if change_dim=="second" else "first"
    
        #adequate shape if conv layer is used
        if conv_layer:
            activation_list = np.zeros((parameter_space[::8].shape[0],20,20,shape[-1]))
            activations_var = np.zeros((shape[0],shape[1],20,20,shape[2]))
        else:
            activation_list = np.zeros((parameter_space[::8].shape[0],shape[-1]))
            activations_var = np.zeros(shape)
       
            
        saver = tf.train.Saver()
        if os.path.isfile(save_path+"DQN.index"):
            new_saver = tf.train.import_meta_graph(save_path+"DQN.meta")
            new_saver.restore(sess, tf.train.latest_checkpoint(save_path))
        else:
            print("not a valid save_path")
            return

        #For each geometric form...
        for count,ellipse in enumerate(parameter_space):
            
            if change_dim == "second":
                my_world.haxis = ellipse[0]
                my_world.vaxis = ellipse[1]
            else:
                my_world.haxis2 = ellipse[0]
                my_world.vaxis2 = ellipse[1]
            my_world.set_frame(which=static_dim)
            
            activation_list *=0

            #... calculate the activation for all other forms and use that to calc variance of activations
            for idx, ellipse2 in enumerate(parameter_space[::8]):
                
                if change_dim == "second":
                    my_world.haxis2 = ellipse2[0]
                    my_world.vaxis2 = ellipse2[1]
                else:
                    my_world.haxis = ellipse2[0]
                    my_world.vaxis = ellipse2[1]                

                my_world.set_frame(which=change_dim)
                state = my_world.get_frame()
                activation_list[idx]  = layer.eval(feed_dict={input_layer: [state]})

            activations_var[ellipse[0]][ellipse[1]] = activation_list.var(axis=0)
            print("\rDone with {0:4d} / {1:4d} points in parameter space".format(count,parameter_space.shape[0]),end="")

        print()
        return activations_var


def sample_memories(mem,batch_size):
    
    '''
    Sample memories from the replay buffer. Returns a batch of batch_size that includes the 
    remembered state-action-reward-next_state-game_over tuple.
    '''
    
    memo_batch  = np.array(random.sample(mem,batch_size))
    
    mem_states       = np.stack(np.array(memo_batch[:,0]))
    mem_actions      = np.stack(np.array(memo_batch[:,1]))
    mem_rewards      = np.stack(np.array(memo_batch[:,2]))
    mem_nxt_state    = np.stack(np.array(memo_batch[:,3]))
    mem_final_state  = np.stack(np.array(memo_batch[:,4]))

    
    return (mem_states, mem_actions, mem_rewards.reshape(-1, 1), mem_nxt_state, mem_final_state.reshape(-1, 1))



def eps_greed(eps,q_values,num_actions):
    if np.random.rand() < eps:
            # explore
            return np.random.randint(num_actions) 
    else:
        # exploit
        return np.argmax(q_values) 
    
def save_me(path_to_file,obj):
    
    with open(path_to_file,"wb+") as file:
        pickle.dump(obj,file)
        
def load_me(path_to_file):
    
    with open(path_to_file,"rb") as file:
        obj = pickle.load(file)
    return obj