from __future__ import absolute_import, division, print_function, unicode_literals
#from IPython import display

import tensorflow as tf
from tensorflow.keras import layers, models
from keras.optimizers import Adam
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
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

from data_importer import CelebA
import GAN as gan

print(tf.__version__)
tf.keras.backend.set_floatx('float32')

# ### Load and prepare the dataset

# ####Download dataset

# Male Female attribute is "Male" in datasaet, -1 female, 1 male

# ####CelebA dataset Wrapper

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
BATCH_SIZE = 100
NUM_IMAGES_USED = 10000
noise_dim = 256
STEPS_PER_EPOCH = np.ceil(NUM_IMAGES_USED/BATCH_SIZE)
CLASS_NAMES = celeba.features_name

img_shape = (IMG_HEIGHT, IMG_WIDTH, 3)

def _parse_function(filename, labels):
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

# print(f"Device being used {tf.test.gpu_device_name()}")

generator = gan.Generator(noise_dim)
generator.summary()
discriminator = gan.Discriminator()
discriminator.summary()

#Smoothing labels posive between 0.9-1.0 and negative between 0.0-0.1
def smooth_pos_and_trick(y):
    tensor = np.random.uniform(0.9,1,y.shape[0])
    to_subst = np.ceil(y.shape[0]*0.05)
    targe = np.random.choice(y.shape[0], int(to_subst))
    for idx in targe:
        tensor[idx] = abs(tensor[idx]-1)
    return tf.convert_to_tensor(tensor, dtype=tf.float32)

def smooth_neg_and_trick(y):
    tensor = np.random.uniform(0,0.1, y.shape[0])
    to_subst = np.ceil(y.shape[0]*0.05)
    targe = np.random.choice(y.shape[0], int(to_subst))
    for idx in targe:
        tensor[idx] = abs(tensor[idx]-1)
    return tf.convert_to_tensor(tensor, dtype=tf.float32)

checkpoint_dir = './training_checkpoints'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator=generator,
                                 discriminator=discriminator)

manager = tf.train.CheckpointManager(checkpoint, directory = checkpoint_dir, max_to_keep=3)

EPOCHS = 100
num_examples_to_generate = 4

# def train(dataset, epochs):
#     checkpoint.restore(manager.latest_checkpoint)
#     if manager.latest_checkpoint:
#         print("Restored from {}".format(manager.latest_checkpoint))
#     else:
#         print("Initializing from scratch.")

#     logdir = '.\\logs\\func\\'
#     if not os.path.exists(logdir):
#         os.makedirs(logdir)

#     writer = tf.summary.create_file_writer(logdir)

#     tf.summary.trace_on(graph=True, profiler=True)

#     cycle = 0
#     start_train = time.time()
#     for epoch in range(epochs):
#         start = time.time()
#         i = 0
#         for image_batch in dataset:
#             genL, discL, gen_grad, disc_grad, real_acc, fake_acc = train_step(image_batch, epoch)

#             i += 1
#             cycle += 1
#             if i % 100 == 0:
#                 print(f"Batch {i}/{STEPS_PER_EPOCH}")

#                 with writer.as_default():
#                     tf.summary.scalar('real acc', real_acc, step=cycle)
#                     tf.summary.scalar('fake acc', fake_acc, step=cycle)
#                     tf.summary.scalar('Gen Loss', genL, step=cycle)
#                     tf.summary.scalar('Disc Loss', discL, step=cycle)
#         if (time.time() - start_train) > 10*60: #Log cada 10 min
#             start_train = time.time()
#             with writer.as_default():
#                 for grad in gen_grad:
#                     tf.summary.histogram('Gen_grad', grad, step=cycle)
#                 for grad in disc_grad:
#                     tf.summary.histogram('Disc_grad', grad, step=cycle)

#         generate_and_save_images(generator,
#                                  epoch + 1,
#                                  tf.random.normal([num_examples_to_generate,1,1, noise_dim]),
#                                  titleadd="Epoch {}".format(epoch))

#         # Save the model every 20 epochs
#         if (epoch + 1) % 20 == 0 or epoch == epochs-1:
#             checkpoint.save(file_prefix = checkpoint_prefix)


#         print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

#     # Generate after the final epoch
#     generate_and_save_images(generator,
#                            epochs,
#                            tf.random.normal([num_examples_to_generate,1,1, noise_dim]))

# print(manager.latest_checkpoint)

def train(g_model, d_model, gan_model, dataset, latent_dim=100, n_epochs=100, n_batch=128):
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

    # manually enumerate epochs
    cycle = 0
    start = time.time()
    for i in range(n_epochs):
        x = 0
        # enumerate batches over the training set
        for image_batch in dataset:
            if cycle % 10 == 0:
              print(f"Batch: {x}/{NUM_IMAGES_USED/n_batch}")
            x += 1
            cycle += 1
            # get randomly selected 'real' samples
            X_real = image_batch
            y_real = tf.ones(n_batch,1)
            # update discriminator model weights
            d_loss1, _ = d_model.train_on_batch(X_real, y_real)
            # generate 'fake' examples
            X_fake, y_fake = gan.generate_fake_samples(g_model, latent_dim, n_batch)
            # update discriminator model weights
            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
            # prepare points in latent space as input for the generator
            X_gan = gan.generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = np.ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
        
            if cycle % 10 == 0:
              with writer.as_default():
                    tf.summary.scalar('Disc loss real', d_loss1, step=cycle)
                    tf.summary.scalar('Disc loss fake', d_loss2, step=cycle)
                    tf.summary.scalar('Gen Loss', g_loss, step=cycle)
        # summarize loss on this batch
        print('Epoch: %d,  Loss: D_real = %.3f, D_fake = %.3f,  G = %.3f' %   (i+1, d_loss1, d_loss2, g_loss))
        # evaluate the model performance
        if (i+1) % 1 == 0:
            gan.summarize_performance(i, g_model, d_model, image_batch, latent_dim)     
        if (i+1) % 20 == 0:
          # Save the model every 10 epochs
            checkpoint.save(file_prefix = checkpoint_prefix)
    print ('Total time for training {} epochs is {} sec'.format(n_epochs, (time.time()-start)))

# create the gan
theGan = gan.define_gan(generator, discriminator)

with tf.device('/device:GPU:0'):
    train(generator, discriminator, theGan, training_dataset, 256, 100,100)

# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))