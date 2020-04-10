from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras import layers, models
from keras.optimizers import Adam
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

from easydict import EasyDict as edict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import glob
import imghdr
import imageio
import PIL
import pathlib
import time
import os
import json

from src.dataset.dataset import DatasetCeleba
import src.lib
from src.training import *

def main(config_file='config.json'):
  #Load attributes as EasyDict from file as "a"
  with open(config_file) as f:
    a = edict(json.loads(f.read()))
  
  training_dataset = DatasetCeleba(a["celebaParam"]).getDataset()
  g, d = create_network(**a.network_params)
  network = define_gan(g, d)

  with tf.device('/device:GPU:0'):
    train(g, d, network, training_dataset, **a.training_params)

if __name__ == "__main__":
  main()