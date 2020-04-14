import os
import sys
import tensorflow as tf
import pandas as pd
import subprocess
import itertools
import numpy as np
import functools

from src.dataset.misc import *
from src.lib.noise_plot import *

__all__ = ['map_training_data', 'download_celeba', 'feat_name', 
  'filtered_dataframe', 'dict_of_smallest_label_in_df', 'multilabeled_features',]

def assert_exit(condition, err_message):
  try:
    assert condition
  except AssertionError:
    print(err_message)
    sys.exit(0)

def map_training_data(image, labels):
  # Images are loaded and decoded
  image = tf.io.read_file(image)
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.cast(image, tf.float32)

  # Reshaping, normalization and optimization goes here
  image = tf.image.resize(image, (128, 128),
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  # mean, std = tf.reduce_mean(image), tf.math.reduce_std(image)
  # image = (image-mean)/std # Normalize the images to [0, 1]
  image = image/255
  return image, labels

def download_celeba(k, folder):
  def blockPrint():
    sys.stdout = open(os.devnull, 'w')
  def enablePrint():
    sys.stdout = sys.__stdout__

  assert_exit(k["kaggleUser"] is not None, "Please write a kaggle username in config.json")
  assert_exit(k["kagglePass"] is not None, "Please write a kaggle password in config.json")

  os.environ['KAGGLE_USERNAME'] = k["kaggleUser"]
  os.environ['KAGGLE_KEY'] = k["kagglePass"]

  blockPrint()

  try: subprocess.call("./scripts/download_celeba.sh")
  except: pass

  enablePrint()

  assert_exit(os.path.exists(folder), "Dataset could not be downloaded")

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
    new_query = df_aux[df_aux.index.isin(new_query.index)]
    feat_df = pd.concat([feat_df, new_query[:min_value_split]])

  return feat_df
