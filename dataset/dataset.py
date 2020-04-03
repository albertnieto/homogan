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
    self.celeba_features = self.feat_name(self.filter_features) + self.feat_name(self.multilabeling_features)

    if not os.path.exists(self.dataset_folder):
      self.download_celeba(params["kaggle"])
    
    self.celeba = CelebA(selected_features=self.celeba_features, main_folder=self.dataset_folder)
    self.dataframe, self.image_list = self.parse_attributes()
    self.images_used = max(params["num_img_training"], len(self.image_list)) 

  def download_celeba(self, k):
    os.environ['KAGGLE_USERNAME'] = k["kaggleUser"]
    os.environ['KAGGLE_KEY'] = k["kagglePass"]
    rc = subprocess.call("./docs/download_celeba.sh")

  def dict_smallest_feature(self, labels, dataframe):
    def smallest_feature_from_iteration(log, small_label, small_value):
      log = [set(i) for i in log]
      

    smallest_iteration_value = 99999999999
    smallest_iteration = ''
    iteration_log = []
    label = ''

    iterations = list(itertools.permutations(labels))
    num_features = math.sqrt(len(iterations))

    for iteration in iterations:  # equals to len(multilabeling_features)**2
      query = dataframe
      for iter_label in iteration:     # for each label of iteration
        for i in range (0,1):
          dataframe[getattr(dataframe, iter_label)==i] # check binary state

      occurrences = query.size
      
      if smallest_iteration_value > occurrences:
        smallest_iteration_value = occurrences
        smallest_iteration = iteration

      iteration_log.append((iteration, occurrences))

    value, label = smallest_feature_from_iteration(iteration_log, 
                                smallest_iteration, smallest_iteration_value)

    return {"value": value, "label": label}

  def feat_name(self, feats):
        ret = []
        if len(feats) > 0:
          ret += [i[0] for i in feats]
        return ret

  def filtered_dataframe(self, df):
    for i in self.filter_features:
      df = df[getattr(df, i[0]) == i[1]]
    return df

  def multilabeled_features(self, df):
    labels = self.feat_name(self.multilabeling_features)
    smallest_feature = get_smallest_feature(labels, df)

  def parse_attributes(self):
    feat_df = self.celeba.attributes
    print(self.celeba_features)
    # Add path to image_id
    feat_df['image_id'] = feat_df['image_id'].apply(
      lambda x: self.dataset_folder + '/img_align_celeba/img_align_celeba/' + x)

    # Filter dataset if filters
    if self.filter_features:
      feat_df = filtered_dataframe(feat_df)

    # Enable multilabeling features if 
    if self.multilabeling_features:
      feat_df = multilabeled_features(feat_df)

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