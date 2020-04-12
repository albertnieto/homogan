from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import tensorflow as tf

import json

from easydict import EasyDict as edict
from src.dataset.dataset import DatasetCeleba
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