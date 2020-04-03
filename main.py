from __future__ import absolute_import, division, print_function, unicode_literals
#from IPython import display

import tensorflow as tf
from tensorflow.keras import layers, models
from keras.optimizers import Adam
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

from easydict import EasyDict as edict
from simplejson import loads

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import glob
import imghdr
import imageio
import PIL
import pathlib
import time
import os
import json

from dataset.dataset import DatasetCeleba
import lib
import GAN as gan

print(tf.__version__)
tf.keras.backend.set_floatx('float32')

def train(g_model, 
          d_model, 
          gan_model, 
          dataset, 
          latent_dim    =100,
          n_epochs      =100, 
          train_GEN     =1, 
          train_DISC    =1):

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
            n_batch = image_batch[0].shape[0]
            if cycle % 50 == 0:
              print(f"Batch: {x}/{NUM_IMAGES_USED/n_batch}")
            x += 1
            cycle += 1
            # get randomly selected 'real' samples
            X_real = image_batch
            y_real = tf.ones(n_batch,1)
            # smoothing
            y_real = smooth_pos_and_trick(y_real)
            if cycle % train_DISC == 0:
              # update discriminator model weights
              d_loss1, _ = d_model.train_on_batch(X_real, y_real)
              # generate 'fake' examples
              X_fake, y_fake = gan.generate_fake_samples(g_model, latent_dim, n_batch)
              # smoothing
              y_fake = smooth_neg_and_trick(y_fake)
              # update discriminator model weights
              d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
            if cycle % train_GEN == 0:
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
        if (i+1) % 2 == 0:
            gan.summarize_performance(i, g_model, d_model, image_batch, latent_dim)     
        if (i+1) % 20 == 0:
          # Save the model every 10 epochs
            checkpoint.save(file_prefix = checkpoint_prefix)
    print ('Total time for training {} epochs is {} sec'.format(n_epochs, (time.time()-start)))


def main(config_file='config.json'):
  #Load attributes as EasyDict from file as "a"
  a = edict(loads(config_file))

  training_dataset = DatasetCeleba()

  g = generator_from_experiment(a.architecture_used, a.noise_dim)
  d = discriminator_from_experiment(a.architecture_used)
  network = define_gan(g, d)

  checkpoint_dir = './training_checkpoints'

  if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
  checkpoint = tf.train.Checkpoint(generator=g,
                                  discriminator=d)

  manager = tf.train.CheckpointManager(checkpoint, directory = checkpoint_dir, 
                                        max_to_keep=3)

  with tf.device('/device:GPU:0'):
      train(g, d, network, training_dataset, a.NOISE_DIM, a.EPOCHS, 
            a.TRAIN_TIMES_G, TRAIN_TIMES_D)


if __name__ == "__main__":
  main()