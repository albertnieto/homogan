import os
import pandas as pd
import subprocess

from dataset.celebaWrapper import CelebA
'''Load and prepare the dataset
  Download dataset
  Male Female attribute is "Male" in datasaet, -1 female, 1 male
  CelebA dataset Wrapper
  Prepare dataset
'''

class DatasetCeleba():

  def __init__(self, params):
    def get_features(feats):
      return [i[0] for i in feats]

    def join_images_path(x):
      return (lambda x: self.dataset_folder + 
                          '/img_align_celeba/img_align_celeba/' + x)

    download_celeba(params["kaggle"])
    
    self.features = get_features(params["celeba_features"])
    self.dataset_folder = params["dataset_folder"]
    self.celeba = CelebA(selected_features=self.features, main_folder=self.dataset_folder)
    self.dataframe, self.image_list = parse_attributes(params["celeba_features"])

  def parse_attributes(feats):
    feat_df = self.celeba.attributes

    for i in feats:
      feat_df = feat_df[getattr(feat_df, i[0]) == i[1]]

    feat_df['image_id'] = feat_df['image_id'].apply(join_images_path)
    image_list = feat_df['image_id'].tolist()

    return feat_df, image_list

  def download_celeba(k):
    os.environ['KAGGLE_USERNAME'] = k[kaggleUser]
    os.environ['KAGGLE_KEY'] = k[kagglePass]
    rc = subprocess.call("src/download_celeba.sh")

  def parse_function(filename, labels):
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
  
  def prepare():
    #Create data set and map it. Could be improved if we can load images 
    # previously and avoid mapping it later.
    training_images = (tf.data.Dataset.from_tensor_slices(
                        (list(joined['image_id'][:NUM_IMAGES_USED]), 
                          list(joined['Male'][:NUM_IMAGES_USED]))))

    training_dataset = training_images.map(parse_function)

    #Shuffle and batch
    training_dataset = training_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)