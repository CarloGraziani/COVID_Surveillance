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
        matrix_threshold = self.threshold* tf.ones(vload[...,:].shape)
        return tf.math.greater_equal(vload, matrix_threshold)
        


    

    def symptom_fn(self, vload, sympar):
        """
            does not depent on sympar for now...
            to be written
            
            return: sym_fn proba of exhibiting symptoms
            """
        return 0.6
