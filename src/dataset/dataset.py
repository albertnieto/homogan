import os
import tensorflow as tf
import pandas as pd
import subprocess
import itertools
import math
import numpy as np
import functools
import skimage.io
import matplotlib.pyplot as plt

from src.dataset.misc import *
from src.dataset.celebaWrapper import CelebA
from src.lib.noise_plot import *
'''Load and prepare the dataset
  Download dataset
  Male Female attribute is "Male" in datasaet, -1 female, 1 male
  CelebA dataset Wrapper
  Prepare dataset
'''

class DatasetCeleba():
  def __init__(self, params):
    def filter_features(feats):
      return [i for i in feats if len(i)==2]

    def multilabeling_features(feats):
      return [i for i in feats if len(i)==1]

    self.params = params
    self.dataset_folder = params["dataset_folder"]
    self.filter_features = filter_features(params["celeba_features"])
    self.multilabeling_features = multilabeling_features(params["celeba_features"])
    self.celeba_features = feat_name(self.filter_features) + feat_name(self.multilabeling_features)

    if not os.path.exists(self.dataset_folder):
      download_celeba(params["kaggle"])
    
    self.celeba = CelebA(selected_features=self.celeba_features, main_folder=self.dataset_folder)
    self.dataset = self.generate_dataset()

  def getDataset(self):
    return self.dataset

  def parse_attributes(self):
    feat_df = self.celeba.attributes
    # Add path to image_id
    feat_df['image_id'] = feat_df['image_id'].apply(
      lambda x: self.dataset_folder + '/img_align_celeba/img_align_celeba/' + x)

    # Filter dataset if filters
    if self.filter_features:
      feat_df = filtered_dataframe(feat_df, self.filter_features)

    # Enable multilabeling features if 
    if self.multilabeling_features:
      feat_df = multilabeled_features(feat_df, self.multilabeling_features)
      for ff in self.filter_features:
        feat_df = feat_df.drop(ff[0], axis=1)

    image_list = feat_df['image_id'].tolist()
    feat_df = feat_df.drop('image_id', axis=1)

    return feat_df, image_list
  
  def generate_dataset(self):
    df, image_list = self.parse_attributes()
    #Create data set and map it. Could be improved if we can load images 
    # previously and avoid mapping it later.
    training_images = tf.data.Dataset.from_tensor_slices((image_list, df.values.tolist()))
    training_dataset = training_images.map(map_training_data)
    #Shuffle and batch
    # TODO add params json
    training_dataset = training_dataset.shuffle(3000).batch(100)
    return training_dataset
