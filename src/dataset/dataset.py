import os
import tensorflow as tf
import pandas as pd
import subprocess
import itertools
import math
import numpy as np
import functools
from src.dataset.celebaWrapper import CelebA
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
    def map_training_data(filename, labels):
      #Images are loaded and decoded
      image_string = tf.io.read_file(filename)
      image_decoded = tf.image.decode_jpeg(image_string, channels=3)
      image = tf.cast(image_decoded, tf.float32)

      #Reshaping, normalization and optimization goes here
      image = tf.image.resize(image, (128, 128),
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      
      # mean, std = tf.reduce_mean(image), tf.math.reduce_std(image)
      # image = (image-mean)/std # Normalize the images to [0, 1]
      image = image/255
      return image, labels
      
    df, image_list = self.parse_attributes()
    #Create data set and map it. Could be improved if we can load images 
    # previously and avoid mapping it later.
    training_images = tf.data.Dataset.from_tensor_slices((image_list, df.values.tolist()))
    training_dataset = training_images.map(map_training_data)
    #Shuffle and batch
    training_dataset = training_dataset.shuffle(3000).batch(100)
    return training_dataset

def download_celeba(k):
  os.environ['KAGGLE_USERNAME'] = k["kaggleUser"]
  os.environ['KAGGLE_KEY'] = k["kagglePass"]
  rc = subprocess.call("./docs/download_celeba.sh")

def feat_name(feats):
  ret = []
  if len(feats) > 0:
    ret += [i[0] for i in feats]
  return ret

def filtered_dataframe(df, features):
  for i in features:
    df = df[getattr(df, i[0]) == i[1]]
  return df

def dict_of_smallest_label_in_df(labels, dataframe):
  ret_value = 99999999999
  ret_label = ''

  for label in labels:
    i0 = len(dataframe[getattr(dataframe, label)==0].index)
    i1 = len(dataframe[getattr(dataframe, label)==1].index)
    if i0 < i1 and i0 < ret_value:
      ret_value = i0
      ret_label = (label, 0)
    if i1 < i0 and i1 < ret_value:
      ret_value = i1
      ret_label = (label, 1)
  
  return {"value": ret_value, "label": ret_label}

def multilabeled_features(df, features):
  def conjunction(*conditions):
    return functools.reduce(np.logical_and, conditions)

  def unpack_dict(d):
    return list(d.items())[0][0], list(d.items())[0][1]

  labels = feat_name(features)
  f = dict_of_smallest_label_in_df(labels, df)

  min_feature = f["label"][0]
  min_feature_value = f["label"][1]
  min_value = f["value"]
  
  inv_min_feature_value = 0 if min_feature_value == 1 else 1

  reduced_labels = labels
  reduced_labels.remove(min_feature)
  rl_size = len(reduced_labels)
  # iterations = list(itertools.permutations(reduced_labels))

  min_value_split = min_value // (rl_size**2)

  feat_df = df[getattr(df, min_feature)==min_feature_value]
  df_aux = df[getattr(df, min_feature)==inv_min_feature_value]
  
  bits = ['0', '1']
  query_list = []

  for i in itertools.product(bits, repeat = rl_size):
    for j, value in enumerate(reduced_labels):
      query_list.append({value:i[j]})

  query_composite_list = [query_list[x:x+rl_size] for x in range(0, len(query_list), rl_size)]

  for label_query in query_composite_list:
    ql = []
    for c in label_query:
      k, v = unpack_dict(c)
      ql.append(df_aux[getattr(df_aux, k) == v])

    new_query = df_aux[conjunction(*ql)]
    # chapuza
    new_query = df_aux[df_aux.index.isin(new_query.index)]
    feat_df = pd.concat([feat_df, new_query[:min_value_split]])

  return feat_df
# x = {**x, **y}