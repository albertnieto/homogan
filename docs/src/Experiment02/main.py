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
tf.keras.backend.set_floatx('float32')

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
    def __init__(self, main_folder='C:\\Users\\Jordi\\Anaconda3\\celeba-dataset\\', selected_features=None, drop_features=[]):
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


####Prepare dataset

celeba = CelebA()

data_dir = pathlib.Path(celeba.images_folder + "")
image_count = len(list(data_dir.glob('*/*.jpg')))
image_list = list(data_dir.glob('*/*.jpg'))
image_list = [str(x) for x in image_list]
len(image_list)
print(len(list(data_dir.glob('*/*.jpg'))))


IMG_HEIGHT = 128
IMG_WIDTH = 128
BUFFER_SIZE = 3000
BATCH_SIZE = 16
NUM_IMAGES_USED = 10000
STEPS_PER_EPOCH = np.ceil(NUM_IMAGES_USED/BATCH_SIZE)
CLASS_NAMES = celeba.features_name

img_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
noise_shape = (100,)


def _parse_function(filename, labels):
    #Images are loaded and decoded
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)

    #Reshaping, normalization and optimization goes here
    image = tf.image.resize(image, (IMG_HEIGHT, IMG_WIDTH), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    mean, std = tf.reduce_mean(image), tf.math.reduce_std(image)
    image = (image-mean)/std # Normalize the images to [0, 1]
    
    return image, labels


#Labels and images are created
labels = celeba.attributes.drop('image_id', 1)
labels = labels.applymap(lambda x: 1 if x else 0) 
labels = tf.constant(labels.values, dtype=tf.dtypes.float32)


#Create data set and map it. Could be improved if we can load images previously
# and avoid mapping it later.
training_images = (tf.data.Dataset.from_tensor_slices((image_list[:NUM_IMAGES_USED], labels[:NUM_IMAGES_USED])))


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


#Test

class Generator(tf.keras.Model):
    def __init__(self, generator_input_shape=(256,)):
        super(Generator, self).__init__()
        self.generator_input_shape = generator_input_shape
        self.dense1 = layers.Dense(1024, use_bias=False)
        self.batchNorm1 = layers.BatchNormalization()
        self.leakyRelu1 = layers.LeakyReLU()

        self.dense2 = layers.Dense(128*8*8, use_bias=False)
        self.batchNorm2 = layers.BatchNormalization()
        self.leakyRelu2 = layers.LeakyReLU()
        self.reshape = layers.Reshape((8,8,128))

        self.unconv1 = layers.Conv2DTranspose(128, kernel_size=(5,5) ,  strides=(2,2), padding="same", use_bias=False)
        self.conv1 = layers.Conv2D( 64  , ( 1 , 1 ) , padding='same', name="block_4", use_bias=False)
        self.batchNorm3 = layers.BatchNormalization()
        self.leakyRelu3 = layers.LeakyReLU()

        self.unconv2 = layers.Conv2DTranspose(32, kernel_size=(5,5) ,  strides=(2,2), padding="same", use_bias=False)
        self.conv2 = layers.Conv2D( 64  , ( 1 , 1 ) , padding='same', name="block_5", use_bias=False)
        self.batchNorm4 = layers.BatchNormalization()
        self.leakyRelu4 = layers.LeakyReLU()
                
        self.unconv3 = layers.Conv2DTranspose(32, kernel_size=(3,3) ,  strides=(2,2), padding="same", use_bias=False)
        self.conv3 = layers.Conv2D( 64  , ( 1 , 1 ) , padding='same', name="block_6", use_bias=False)
        self.batchNorm5 = layers.BatchNormalization()
        self.leakyRelu5 = layers.LeakyReLU()

        self.unconv4 = layers.Conv2DTranspose(32, kernel_size=(3,3) ,  strides=(2,2), padding="same", use_bias=False)
        self.conv4 = layers.Conv2D( 128  , ( 1 , 1 ) , padding='same', name="block_7", use_bias=False)
        self.batchNorm6 = layers.BatchNormalization()
        self.leakyRelu6 = layers.LeakyReLU()

        self.img = layers.Conv2D( 3 , ( 1 , 1 ) , activation='tanh' , padding='same', name="final_block")

    def call(self, inputs):
        x = self.leakyRelu1(self.batchNorm1(self.dense1(inputs)))
        x = self.leakyRelu2(self.batchNorm2(self.dense2(x)))
        x = self.reshape(x)

        x = self.unconv1(x)
        x = self.leakyRelu3(self.batchNorm3(self.conv1(x)))

        x = self.unconv2(x)
        x = self.leakyRelu4(self.batchNorm4(self.conv2(x)))

        x = self.unconv3(x)
        x = self.leakyRelu5(self.batchNorm5(self.conv3(x)))

        x = self.unconv4(x)
        x = self.leakyRelu6(self.batchNorm6(self.conv4(x)))

        return self.img(x)

    #https://github.com/tensorflow/tensorflow/issues/25036
    def model(self):
        x = layers.Input(shape=self.generator_input_shape)
        return models.Model(inputs=[x], outputs=self.call(x))


generator = Generator().model()
noise = tf.random.normal([1, 256])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')

generator.summary()


class Discriminator(tf.keras.Model):
    def __init__(self, input_shape, output_shape=1):
        super(Discriminator, self).__init__()
        self.discriminator_input_shape = input_shape
        self.conv1 = layers.Conv2D(32, (5, 5), padding='same', input_shape=[128, 128, 3], name='block1_conv1')
        self.batchNorm1 = layers.BatchNormalization()
        self.leakyRelu1 = layers.LeakyReLU()
        self.dropout1 = layers.Dropout(0.3)

        self.conv2 = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', name='block1_conv2')
        self.batchNorm2 = layers.BatchNormalization()
        self.leakyRelu2 = layers.LeakyReLU()
        self.dropout2 = layers.Dropout(0.3)

        self.conv3 = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', name='block1_conv3')
        self.batchNorm3 = layers.BatchNormalization()
        self.leakyRelu3 = layers.LeakyReLU()
        self.dropout3 = layers.Dropout(0.3)

        self.conv4 = layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same', name='block1_conv4')
        self.batchNorm4 = layers.BatchNormalization()
        self.leakyRelu4 = layers.LeakyReLU()
        self.dropout4 = layers.Dropout(0.3)

        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(output_shape)
    
    def call(self, inputs):
        x = self.dropout1(self.leakyRelu1(self.batchNorm1(self.conv1(inputs))))
        x = self.dropout2(self.leakyRelu2(self.batchNorm2(self.conv2(x))))
        x = self.dropout3(self.leakyRelu3(self.batchNorm3(self.conv3(x))))
        x = self.dropout4(self.leakyRelu4(self.batchNorm4(self.conv4(x))))

        x = self.flatten(x)
        return self.dense1(x)
    
    #https://github.com/tensorflow/tensorflow/issues/25036
    def model(self):
        x = layers.Input(shape=self.discriminator_input_shape)
        return models.Model(inputs=[x], outputs=self.call(x))


discriminator = Discriminator((IMG_HEIGHT, IMG_WIDTH, 3)).model()
decision = discriminator(generated_image)
print (decision)

discriminator.summary()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = 0.5 * (real_loss + fake_loss)
    #print(real_loss)
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

manager = tf.train.CheckpointManager(checkpoint, directory = checkpoint_dir, max_to_keep=3)

EPOCHS = 100
noise_dim = 256
num_examples_to_generate = 4

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    gen_loss = 0
    disc_loss = 0
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        # tf.print(real_output[0])
        # tf.print(fake_output[0])
        gen_loss = generator_loss(fake_output)
        #tf.print(gen_loss)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    #print(gen_loss)
    return gen_loss, disc_loss

def generate_and_save_images(model, epoch, test_input, titleadd = ""):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(10,5))

    for i, img in enumerate(predictions):
        ax = fig.add_subplot(1,4,i+1)
        ax.imshow(img)
        fig.suptitle("Generated images "+titleadd,fontsize=30)

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

def train(dataset, epochs):
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    gen_losses = []
    disc_losses = []

    logdir = '.\\logs\\func\\'
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    writer = tf.summary.create_file_writer(logdir)

    tf.summary.trace_on(graph=True, profiler=True)

    for epoch in range(epochs):
        start = time.time()
        i = 0
        for image_batch in dataset:
            genL, discL = train_step(image_batch)

            i += 1
            if i % 100 == 0:
                print(f"Batch {i}/{STEPS_PER_EPOCH}")
                gen_losses.append(genL.numpy())
                disc_losses.append(discL.numpy())

            with writer.as_default():
                tf.summary.trace_export(
                    name="pou",
                    step=epoch,
                    profiler_outdir=logdir)

        # Produce images for the GIF as we go
        #display.clear_output(wait=True)

        xs = range(1, len(gen_losses) + 1)
        plt.plot(xs, gen_losses)
        plt.plot(xs, disc_losses)

        generate_and_save_images(generator,
                                 epoch + 1,
                                 seed,titleadd="Epoch {}".format(epoch))

        # Save the model every 15 epochs
        if (epoch + 1) % 20 == 0 or epoch == epochs-1:
            checkpoint.save(file_prefix = checkpoint_prefix)


        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                           epochs,
                           seed)

print(manager.latest_checkpoint)

with tf.device('/device:GPU:0'):
    train(training_dataset, EPOCHS)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))