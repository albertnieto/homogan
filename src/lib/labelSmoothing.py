import tensorflow as tf
import numpy as np
#Smoothing labels posive between 0.9-1.0 and negative between 0.0-0.1

__all__ = ['smooth_pos_and_trick', 'smooth_neg_and_trick']

def smooth_pos_and_trick(y):
    tensor = np.random.uniform(0.9,1,y.shape[0])
    to_subst = np.ceil(y.shape[0]*0.05)
    targe = np.random.choice(y.shape[0], int(to_subst))
    for idx in targe:
        tensor[idx] = abs(tensor[idx]-1)
    return tf.convert_to_tensor(tensor, dtype=tf.float32)

def smooth_neg_and_trick(y):
    tensor = np.random.uniform(0,0.1, y.shape[0])
    to_subst = np.ceil(y.shape[0]*0.05)
    targe = np.random.choice(y.shape[0], int(to_subst))
    for idx in targe:
        tensor[idx] = abs(tensor[idx]-1)
    return tf.convert_to_tensor(tensor, dtype=tf.float32)