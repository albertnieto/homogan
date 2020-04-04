import os
import pandas as pd
import subprocess
import itertools
import math

from dataset.celebaWrapper import CelebA
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

    self.dataset_folder = params["dataset_folder"]
    self.filter_features = filter_features(params["celeba_features"])
    self.multilabeling_features = multilabeling_features(params["celeba_features"])
    self.celeba_features = feat_name(self.filter_features) + feat_name(self.multilabeling_features)

    if not os.path.exists(self.dataset_folder):
      download_celeba(params["kaggle"])
    
    self.celeba = CelebA(selected_features=self.celeba_features, main_folder=self.dataset_folder)
    self.dataframe, self.image_list = self.parse_attributes()
    self.images_used = max(params["num_img_training"], len(self.image_list)) 

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

    image_list = feat_df['image_id'].tolist()

    return feat_df, image_list

  def parse_function(self, filename, labels):
    #Images are loaded and decoded
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)

    #Reshaping, normalization and optimization goes here
    image = tf.image.resize(image, (IMG_HEIGHT, IMG_WIDTH),
                              method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    # mean, std = tf.reduce_mean(image), tf.math.reduce_std(image)
    # image = (image-mean)/std # Normalize the images to [0, 1]
    image = image/255
    return image, labels
  
  def prepare(self):
    #Create data set and map it. Could be improved if we can load images 
    # previously and avoid mapping it later.
    training_images = (tf.data.Dataset.from_tensor_slices(
                        (list(joined['image_id'][:self.images_used]), 
                          list(df[:self.images_used]))))

    training_dataset = training_images.map(this.parse_function)

    #Shuffle and batch
    training_dataset = training_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


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

def dict_smallest_feature(labels, dataframe):
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
  def query(dataframe, column, operation, value): 
    return operation(dataframe[column], value)

  def add_queries(dataframe, *b):
    return dataframe[(np.logical_and(*b))]

  labels = feat_name(features)
  f = dict_smallest_feature(labels, df)

  min_feature = f["label"][0]
  min_feature_value = f["label"][1]
  min_value = f["value"]
  inv_value = 0 if min_value == 1 else 1

  reduced_labels = labels
  reduced_labels.remove(min_feature)
  rl_size = len(reduced_labels)
  iterations = list(itertools.permutations(reduced_labels))

  df_1 = df[getattr(df, min_feature)==min_value]
  df_2 = df[getattr(df, min_feature)==inv_value]

# >>> c = {"one": 1, "two": 2}
# >>> for k,v in c.items():
# ...    exec("%s=%s" % (k,v))
  bits = ['0', '1']
  query_list = []

  # for values in iterations:
  for i in itertools.product(bits, repeat = rl_size):
    for j, value in enumerate(reduced_labels):
      query_list.append({value:i[j]})

  query_composite_list = [query_list[x:x+rl_size] for x in range(0, len(query_list), rl_size)]

  for query in query_composite_list:
    ql = []query(dataframe, column, operation, value)
    ql.append(df_2, )
    for y in query:
      x = {**x, **y}
    for k,v in c.items():
      exec("%s=%s" % (k,v))
