import pandas as pd

'''Load and prepare the dataset
  Download dataset
  Male Female attribute is "Male" in datasaet, -1 female, 1 male
  CelebA dataset Wrapper
  Prepare dataset
'''

class DatasetCeleba():

  def __init__(self, features=None, num_images=0):
    self.features = ['Male']
    self.celeba = CelebA(selected_features=features, main_folder=dataset_folder)

    feat_df = celeba.attributes
    feat_df['image_id'] = feat_df['image_id'].apply(
      lambda x: dataset_folder + '/img_align_celeba/img_align_celeba/'+x)
    feat_female = feat_df[feat_df.Male == 0][:5000]

    image_list = feat_male['image_id'].tolist()
    image_list = image_list + feat_female['image_id'].tolist()
    joined = pd.concat([feat_male,feat_female])

    def parse_function(filename, labels):
      #Images are loaded and decoded
      image_string = tf.io.read_file(filename)
      image_decoded = tf.image.decode_jpeg(image_string, channels=3)
      image = tf.cast(image_decoded, tf.float32)

      #Reshaping, normalization and optimization goes here
      image = tf.image.resize(image, (IMG_HEIGHT, IMG_WIDTH), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      
      # mean, std = tf.reduce_mean(image), tf.math.reduce_std(image)
      # image = (image-mean)/std # Normalize the images to [0, 1]
      image = image/255
      return image, labels
    
    def prepare():
      #Labels and images are created
      # labels = celeba.attributes.drop('image_id', 1)
      # labels = labels.applymap(lambda x: 1 if x else 0) 
      # labels = tf.constant(labels.values, dtype=tf.dtypes.float32)


      #Create data set and map it. Could be improved if we can load images previously
      # and avoid mapping it later.
      training_images = (tf.data.Dataset.from_tensor_slices((list(joined['image_id'][:NUM_IMAGES_USED]), list(joined['Male'][:NUM_IMAGES_USED]))))

      training_dataset = training_images.map(parse_function)

      #Shuffle and batch
      training_dataset = training_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)