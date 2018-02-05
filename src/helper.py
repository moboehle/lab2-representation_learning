import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors as clrs
import os
import tensorflow as tf
from network import * 
import random 
import pickle 

def plot_variances(train_both,train_single):
    fig, axes = plt.subplots(8,16,figsize=(10,5))
    axes = axes.flatten()
    fig.suptitle("Variance for each dense layer node if only first dimension is relevant",fontsize=18)
    _min,_max = (np.min([np.log(train_both[:,:,:]+1),np.log(train_single[:,:,:]+1)]),\
                 np.max([np.log(train_both[:,:,:]+1),np.log(train_single[:,:,:]+1)]))
    cmap=plt.cm.Reds
    norm = clrs.Normalize(_min,_max)

    for idx,plot in enumerate(range(train_single.shape[-1])):
        im = axes[idx].imshow(np.log(train_single[:,:,idx]+1),cmap=cmap,norm=norm)
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])

    plt.subplots_adjust(bottom=0.1, right=0.9, top=0.9)
    cax = plt.axes([1.05, 0.25, 0.025, 0.45])
    cbar = plt.colorbar(im,cax=cax)
    cbar.set_label(r"$\log$(variance+1) per position")

    plt.show()
    
    
    fig, axes = plt.subplots(8,16,figsize=(10,5))
    axes = axes.flatten()
    fig.suptitle("Variance for each dense layer node if both dimensions are relevant",fontsize=18)

    for idx,plot in enumerate(range(train_both.shape[-1])):
        im = axes[idx].imshow(np.log(train_both[:,:,idx]+1),cmap=cmap,norm=norm)
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])

    plt.subplots_adjust(bottom=0.1, right=0.9, top=0.9)
    cax = plt.axes([1.05, 0.25, 0.025, 0.45])
    cbar = plt.colorbar(im,cax=cax)
    cbar.set_label(r"$\log$(variance+1) per position")

    

def plot_qs_and_actions(qvalues,num_actions,targets,range_x,range_y):
    

    X, Y = np.meshgrid(range(1,range_x+1),range(1,range_y+1))
    
    for qvalue,target in zip(qvalues,targets):
        fig = plt.figure(figsize=(10,14))

        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312,projection="3d")
        ax3 = fig.add_subplot(313)
        cmap=plt.cm.jet

        norm = clrs.BoundaryNorm(np.arange(-.5,num_actions+.5), cmap.N)

        im = ax1.imshow(np.argmax(qvalue[1:,1:],axis=2),norm=norm,cmap=cmap,origin="lower")
        ax1.set_title("Action per state")
        ax1.set_xlim(1,range_x-1)
        ax1.set_ylim(1,range_y-1)
        ax1.set_xlabel("Width")
        ax1.set_ylabel("Height")
        cbar = fig.colorbar(im,ticks=np.arange(num_actions),spacing="uniform",ax=ax1)

        
        #TODO instead of scatter use vlines and hlines
        ax1.scatter(target["vaxis"],target["haxis"],\
                    color="red",label="target area",marker="s",s=1000,facecolors='none', edgecolors='r',lw=2)
        ax3.scatter(target["vaxis"],target["haxis"],\
                    color="red",label="target area",marker="s",s=1000,facecolors='none', edgecolors='r',lw=2)
        ax1.legend(bbox_to_anchor=[1.1,1.15])
        cbar.ax.set_yticklabels(['Increase width D=1', 'decrease width D=1',\
                                 'increase height D=1','decrease height D=1',\
                                'Increase width D=2', 'decrease width D=2',\
                                 'increase height D=2','decrease height D=2']);


        

        max_q = np.max(qvalue[1:,1:],axis=2)
        surf = ax2.plot_surface(X, Y, max_q, rstride=1, cstride=1, cmap=cm.coolwarm,
                linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5, ax = ax2)
        ax2.set_ylabel("Height")
        ax2.set_xlabel("Width")
        ax2.set_title("Max Q value per state")

        im = ax3.imshow(max_q,origin="lower",cmap=cm.coolwarm)
        ax3.set_ylabel("Height")
        ax3.set_xlabel("Width")
        fig.colorbar(im,ax=ax3)



        plt.show()
        
        

def calc_variances_over_2ndD(sess, my_world, save_path,parameter_space,shape,last_layer,input_layer):
    
    
    activation_list = np.zeros((parameter_space[::8].shape[0],shape[-1]))

    activations_var = np.zeros(shape)
    saver = tf.train.Saver()
    if os.path.isfile(save_path+".index"):
        saver.restore(sess, save_path)
    else:
        print("not a valid save_path")
        return

    #For each geometric form...
    for count,ellipse in enumerate(parameter_space):
        my_world.haxis = ellipse[0]
        my_world.vaxis = ellipse[1]
        my_world.set_frame(which="first")
        activation_list *=0
        
        #... calculate the activation for all other forms and use that to calc variance of activations
        for idx, ellipse2 in enumerate(parameter_space[::8]):
            my_world.haxis2 = ellipse2[0]
            my_world.vaxis2 = ellipse2[1]
            my_world.set_frame(which="second")
            state = my_world.get_frame()
            activation_list[idx]  = last_layer.eval(feed_dict={input_layer: [state]})

        activations_var[ellipse[0]][ellipse[1]] = activation_list.var(axis=0)
        print("\rDone with {0:4d} / {1:4d} points in parameter space".format(count,parameter_space.shape[0]),end="")
        
        
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


def epsilon_greedy(q_values, num_updates,exploratory_steps=0,eps_decay_steps=0,num_actions=0,eps_min=0.1,eps_max=1,**kwargs):
    '''
    Custom epsilon greedy policy. Learn completely random in the first "exploratory_steps" number of steps to
    explore the state space more or less equally. Then decay to minimal epsilon to become more and more greedy.
    '''
    
    #First completely random, then decaying linearly from eps_max to eps_min
    epsilon = 1 if num_updates< exploratory_steps else  \
                max(eps_min, eps_max - (eps_max-eps_min) * (num_updates-exploratory_steps)/eps_decay_steps)

    if np.random.rand() < epsilon:
        # explore
        return np.random.randint(num_actions) 
    else:
        # exploit
        return np.argmax(q_values) 


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