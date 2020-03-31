from __future__ import absolute_import, division, print_function, unicode_literals
from IPython import display

import tensorflow as tf
from tensorflow.keras import layers, models
from keras.optimizers import Adam
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import IPython.display as display
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import glob
import imghdr
import imageio
import PIL
import pathlib
import time
import os
import json


print(tf.__version__)


# ### Load and prepare the dataset

# ####Download dataset

# Male Female attribute is "Male" in datasaet, -1 female, 1 male
# 

# ####CelebA dataset Wrapper

# Dataset wrapper for easy handling of attributes, paths and partitioning.

class CelebA():
    '''Wraps the celebA dataset, allowing an easy way to:
             - Select the features of interest,
             - Split the dataset into 'training', 'test' or 'validation' partition.
    '''
    def __init__(self, main_folder='/content/celeba-dataset', selected_features=None, drop_features=[]):
        self.main_folder = main_folder
        self.images_folder   = os.path.join(main_folder, 'img_align_celeba/')
        self.attributes_path = os.path.join(main_folder, 'list_attr_celeba.csv')
        self.partition_path  = os.path.join(main_folder, 'list_eval_partition.csv')
        self.selected_features = selected_features
        self.features_name = []
        self.__prepare(drop_features)

    def __prepare(self, drop_features):
        '''do some preprocessing before using the data: e.g. feature selection'''
        # attributes:
        if self.selected_features is None:
            self.attributes = pd.read_csv(self.attributes_path)
            self.num_features = 40
        else:
            self.num_features = len(self.selected_features)
            self.selected_features = self.selected_features.copy()
            self.selected_features.append('image_id')
            self.attributes = pd.read_csv(self.attributes_path)[self.selected_features]

        # remove unwanted features:
        for feature in drop_features:
            if feature in self.attributes:
                self.attributes = self.attributes.drop(feature, axis=1)
                self.num_features -= 1
            
        self.attributes.set_index('image_id', inplace=True)
        self.attributes.replace(to_replace=-1, value=0, inplace=True)
        self.attributes['image_id'] = list(self.attributes.index)
    
        self.features_name = list(self.attributes.columns)[:-1]
    
        # load ideal partitioning:
        self.partition = pd.read_csv(self.partition_path)
        self.partition.set_index('image_id', inplace=True)
    
    def split(self, name='training', drop_zero=False):
        '''Returns the ['training', 'validation', 'test'] split of the dataset'''
        # select partition split:
        if name is 'training':
            to_drop = self.partition.where(lambda x: x != 0).dropna()
        elif name is 'validation':
            to_drop = self.partition.where(lambda x: x != 1).dropna()
        elif name is 'test':  # test
            to_drop = self.partition.where(lambda x: x != 2).dropna()
        else:
            raise ValueError('CelebA.split() => `name` must be one of [training, validation, test]')

        partition = self.partition.drop(index=to_drop.index)
            
        # join attributes with selected partition:
        joint = partition.join(self.attributes, how='inner').drop('partition', axis=1)

        if drop_zero is True:
            # select rows with all zeros values
            return joint.loc[(joint[self.features_name] == 1).any(axis=1)]
        elif 0 <= drop_zero <= 1:
            zero = joint.loc[(joint[self.features_name] == 0).all(axis=1)]
            zero = zero.sample(frac=drop_zero)
            return joint.drop(index=zero.index)

        return joint

    def split_all(self, drop_zero=False):

        ret = []
        for i in range(3): 
            to_drop = self.partition.where(lambda x: x != i).dropna()
            partition = self.partition.drop(index=to_drop.index)
            joint = partition.join(self.attributes, how='inner').drop('partition', axis=1)

            if drop_zero is True:
                ret.append(joint.loc[(joint[self.features_name] == 1).any(axis=1)])
            elif 0 <= drop_zero <= 1:
                zero = joint.loc[(joint[self.features_name] == 0).all(axis=1)]
                zero = zero.sample(frac=drop_zero)
                ret.append(joint.drop(index=zero.index))

            ret.append(joint) 

        return ret[0], ret[1], ret[2]


# ####Prepare dataset

celeba = CelebA(selected_features=["Male"])

data_dir = pathlib.Path(celeba.images_folder + "")
image_count = len(list(data_dir.glob('*/*.jpg')))
image_list = list(data_dir.glob('*/*.jpg'))
image_list = [str(x) for x in image_list]
NUM_IMAGES_USED = 10000
print(len(list(data_dir.glob('*/*.jpg'))))


IMG_HEIGHT = 128
IMG_WIDTH = 128
BUFFER_SIZE = 3000
BATCH_SIZE = 64
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)
CLASS_NAMES = celeba.features_name

img_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
noise_shape = (100,)


def _parse_function(filename, labels):
    #Images are loaded and decoded
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)
    #img = load_image(filename)
    #Reshaping, normalization and optimization goes here
    image = tf.image.resize(image, (IMG_HEIGHT, IMG_WIDTH), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = image/255.0 # Normalize the images to [-1, 1]
    return image, labels


#Labels and images are created
labels = celeba.attributes.drop('image_id', 1)
labels = labels.applymap(lambda x: 1 if x else 0) 
labels = tf.constant(labels.values, dtype=tf.dtypes.float32)


#Create data set and map it. Could be improved if we can load images previously
# and avoid mapping it later.
training_images = (tf.data.Dataset.from_tensor_slices((image_list[:10000], labels[:10000])))


training_dataset = training_images.map(_parse_function)


#Shuffle and batch
training_dataset = training_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


# for batch in training_dataset:
#   break

with tf.device('/device:GPU:0'):    
    for batch in training_dataset:
        break
    print(batch[0][0].device)


print(f"Device being used {tf.test.gpu_device_name()}")


# #Test

class Generator(tf.keras.Model):

    def __init__(self, generator_input_shape=(100,)):
        super(Generator, self).__init__()
        self.generator_input_shape = generator_input_shape
        self.dense1 = layers.Dense(1024, activation="relu")
        self.dense2 = layers.Dense(128*8*8, activation="relu")
        self.reshape = layers.Reshape((8,8,128))

        self.unconv1 = layers.Conv2DTranspose(128, kernel_size=(2,2) ,  strides=(2,2) , use_bias=False)
        self.conv1 = layers.Conv2D( 64  , ( 1 , 1 ) , activation='relu' , padding='same', name="block_4")

        self.unconv2 = layers.Conv2DTranspose(32, kernel_size=(2,2) ,  strides=(2,2) , use_bias=False)
        self.conv2 = layers.Conv2D( 64  , ( 1 , 1 ) , activation='relu' , padding='same', name="block_5")
                
        self.unconv3 = layers.Conv2DTranspose(32, kernel_size=(2,2) ,  strides=(2,2) , use_bias=False)
        self.conv3 = layers.Conv2D( 64  , ( 1 , 1 ) , activation='relu' , padding='same', name="block_6")

        self.unconv4 = layers.Conv2DTranspose(64, kernel_size=(2,2) ,  strides=(2,2) , use_bias=False)
        self.conv4 = layers.Conv2D( 128  , ( 1 , 1 ) , activation='relu' , padding='same', name="block_7")

        self.img = layers.Conv2D( 3 , ( 1 , 1 ) , activation='sigmoid' , padding='same', name="final_block")

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.reshape(x)

        x = self.unconv1(x)
        x = self.conv1(x)

        x = self.unconv2(x)
        x = self.conv2(x)

        x = self.unconv3(x)
        x = self.conv3(x)

        x = self.unconv4(x)
        x = self.conv4(x)

        return self.img(x)

#https://github.com/tensorflow/tensorflow/issues/25036
    def model(self):
        x = layers.Input(shape=self.generator_input_shape)
        return models.Model(inputs=[x], outputs=self.call(x))


generator = Generator().model()
generator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.00007, 0.5))
generator.summary()


class Discriminator(tf.keras.Model):
    def __init__(self, input_shape, output_shape=1):
        super(Discriminator, self).__init__()
        self.discriminator_input_shape = input_shape
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv1')
        self.conv2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv2')
        self.maxpooling1 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')
        
        self.conv3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv1')
        self.conv4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv2')
        self.maxpooling2 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')
        
        self.conv5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv1')
        self.conv6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv2')
        self.maxpooling3 = layers.MaxPooling2D((2, 2), strides=(1, 1), name='block4_pool')

        
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(1024,      activation="relu")
        self.dense2 = layers.Dense(output_shape,   activation='sigmoid')
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.maxpooling1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpooling2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.maxpooling3(x)

        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)
    
    #https://github.com/tensorflow/tensorflow/issues/25036
    def model(self):
        x = layers.Input(shape=self.discriminator_input_shape)
        return models.Model(inputs=[x], outputs=self.call(x))


discriminator = Discriminator((IMG_HEIGHT, IMG_WIDTH, 3)).model()
discriminator.compile(loss = 'binary_crossentropy', optimizer = tf.keras.optimizers.Adam(0.00007, 0.5), metrics   = ['accuracy'])
discriminator.summary()


#https://fairyonice.github.io/My-first-GAN-using-CelebA-data.html
def get_noise(nsample=1, nlatent_dim=100):
        noise = np.random.normal(0, 1, (nsample,nlatent_dim))
        return(noise)

def plot_generated_images(noise,path_save=None,titleadd=""):
        imgs = generator.predict(noise)
        fig = plt.figure(figsize=(20,10))
        for i, img in enumerate(imgs):
                ax = fig.add_subplot(1,nsample,i+1)
                ax.imshow(img)
        fig.suptitle("Generated images "+titleadd,fontsize=30)
        
        if path_save is not None:
                plt.savefig(path_save,
                                        bbox_inches='tight',
                                        pad_inches=0)
                plt.close()
        else:
                plt.show()

checkpoint_dir = './training_checkpoints'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator=generator,
                                 discriminator=discriminator)

manager = tf.train.CheckpointManager(checkpoint, directory = checkpoint_dir, max_to_keep=3)


nsample = 4
noise = get_noise(nsample=nsample, nlatent_dim=noise_shape[0])
plot_generated_images(noise)


z = layers.Input(shape=noise_shape)
img = generator(z)

# For the combined model we will only train the generator
discriminator.trainable = False

# The valid takes generated images as input and determines validity
valid = discriminator(img)

# The combined model  (stacked generator and discriminator) takes
# noise as input => generates images => determines validity 
combined = models.Model(z, valid)
combined.compile(loss='binary_crossentropy', optimizer=tf.optimizers.Adam(0.00007, 0.5))
combined.summary()


def train(models, X_train, noise_plot, dir_result="./results_GAN/images",
        save_model_folder = "./results_GAN/models", epochs=100, batch_size=1024):
    '''
    models     : tuple containins three tensors, (combined, discriminator, generator)
    X_train    : np.array containing images (Nsample, height, width, Nchannels)
    noise_plot : np.array of size (Nrandom_sample_to_plot, hidden unit length)
    dir_result : the location where the generated plots for noise_plot are saved 
    
    '''
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    logdir = './logs/func/'
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    writer = tf.summary.create_file_writer(logdir)

    tf.summary.trace_on(graph=True, profiler=True)
    
    
    combined, discriminator, generator = models
    nlatent_dim = noise_plot.shape[1]
    history = []
    cycle = 0
    batches = int(10000/64)+1
    for i in range(epochs):
        x = 0
        for image_batch in X_train:
            #batch = image_batch.numpy()
            # ---------------------
            #  Train Discriminator
            # ---------------------
            if cycle % 10 == 0:
              print(f"Batch: {x}/{NUM_IMAGES_USED/batch_size}")
            x += 1      
            cycle += 1
            images = image_batch[0]

            noise = get_noise(images.shape[0], nlatent_dim)
            # Generate a half batch of new images
            gen_imgs = generator.predict(noise)

            
            # Train the discriminator q: better to mix them together?
            d_loss1 = discriminator.train_on_batch(images, np.ones((images.shape[0], 1)))
            d_loss2 = discriminator.train_on_batch(gen_imgs, np.zeros((images.shape[0], 1)))
            d_loss = 0.5 * np.add(d_loss1, d_loss2)


            # ---------------------
            #  Train Generator
            # ---------------------

            noise = get_noise(batch_size, nlatent_dim)

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = (np.array([1] * batch_size)).reshape(batch_size,1)
            
            # Train the generator
            g_loss = combined.train_on_batch(noise, valid_y)

            if cycle % 10 == 0:
              with writer.as_default():
                    tf.summary.scalar('Disc loss real', d_loss1[0], step=cycle)
                    tf.summary.scalar('Disc loss fake', d_loss2[0], step=cycle)
                    tf.summary.scalar('Gen Loss', g_loss, step=cycle)
            history.append({"D":d_loss[0],"G":g_loss})
        

        # summarize loss on this batch
        print('Epoch: %d,  Loss: D_real = %.3f, D_fake = %.3f,  G = %.3f' %   (i+1, d_loss1[0], d_loss2[0], g_loss))
        # evaluate the model performance
        if (i+1) % 1 == 0:
          plot_generated_images(noise_plot, titleadd="Epoch {}".format(i))
          plot_generated_images(noise_plot, path_save=dir_result+"/image_{:05.0f}.png".format(i), titleadd="Epoch {}".format(epoch))
        if (i+1) % 20 == 0:
          # Save the model every 10 i
            checkpoint.save(file_prefix = checkpoint_prefix)
                                    
    return(history)

dir_result="./results_GAN/images"

if not os.path.exists(dir_result):
    os.makedirs(dir_result)



X_train = training_dataset


# total = 0
# for batch in training_dataset:
#   total += 1
#   print(total)
#   print(batch.shape)


_models = combined, discriminator, generator          


with tf.device('/device:GPU:0'):
    start_time = time.time()
    history = train(_models, X_train, noise, dir_result=dir_result, epochs=2200, batch_size=64)
    end_time = time.time()
    print("-"*10)
    print("Time took: {:4.2f} min".format((end_time - start_time)/60))
