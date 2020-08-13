import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_graphics as tfg
import numpy as np


class proba_pos_sym:
    """
        
        
        Constructor Args:
        
        threshold: float32 that corresponds to the threshold above which an individual would be positive
        
       
    """
    def __init__(self, threshold):
            self.threshold = threshold
    
    

    def positive_fn(self, vload, pospar):
        """
            does not depent on pospar for now...
            to be written
            
            return: pos_fn proba of a positive test result
        """
        matrix_threshold = self.threshold * tf.ones(vload[...,:].shape)
        pos_fn_bool = tf.math.greater_equal(vload, matrix_threshold) #create tensor of boolean
        pos_fn_var = tf.Variable(pos_fn_bool)
        pos_fn = tf.dtypes.cast(pos_fn_var, tf.float32) #convert boolean to float
        #pos_fn = tf.random.uniform(
        #                 vload[...,:].shape, minval=0, maxval=1.0, dtype=tf.dtypes.float32, seed=None, name=None
        #                 )
        #       print('printing pos_fn')
#        print(vload[...,:].shape)
#        print(vload)
#        print(pos_fn)
        return pos_fn



    

    def symptom_fn(self, vload, sympar):
        """
            does not depend on sympar for now...
            to be written
            
            return: sym_fn proba of exhibiting symptoms (constant = 0.6  for now)
            
        
            """
        sym_fn = 0.65* tf.ones(vload[...,:].shape)
        return sym_fn
