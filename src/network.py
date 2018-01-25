import tensorflow as tf
from params import * 


x = tf.placeholder(tf.float32, shape=[None, FRAME_DIM,FRAME_DIM,1])

def network(name, n_out_h1=None, kernel_size_h1 = None, strides_h1 = None, padding = None,\
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
        
        return q_values, theta_names
        
    