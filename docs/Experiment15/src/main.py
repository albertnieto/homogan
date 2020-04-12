from __future__ import absolute_import, division, print_function, unicode_literals
#from IPython import display

import tensorflow as tf
from tensorflow.keras import layers, models
from keras.optimizers import Adam
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
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

from data_importer import CelebA
import GAN as gan

print(tf.__version__)
tf.keras.backend.set_floatx('float32')
def main(dataset_folder = "/content/celeba-dataset"):

    # ### Load and prepare the dataset

    # ####Download dataset

    # Male Female attribute is "Male" in datasaet, -1 female, 1 male

    # ####CelebA dataset Wrapper

    ####Prepare dataset

    features = ['Eyeglasses', 'No_Beard', 'Bald', 'Male']
    celeba = CelebA(selected_features=features, main_folder=dataset_folder)

    feat_df = celeba.attributes
    feat_df['image_id'] = feat_df['image_id'].apply(
    lambda x: dataset_folder + '/img_align_celeba/img_align_celeba/'+x)

    bald_people = feat_df[feat_df.Bald == 1]
    print("bald_people " + str(len(bald_people)))
    bald_people_with_g_with_b = bald_people[(bald_people["Eyeglasses"]==1) & (bald_people["No_Beard"]==0)]
    bald_people_with_g_without_b = bald_people[(bald_people["Eyeglasses"]==1) & (bald_people["No_Beard"]==1)]
    bald_people_without_g_with_b = bald_people[(bald_people["Eyeglasses"]==0) & (bald_people["No_Beard"]==0)]
    bald_people_without_g_without_b = bald_people[(bald_people["Eyeglasses"]==0) & (bald_people["No_Beard"]==1)]
    
    print("bald_people_with_g_with_b " + str(len(bald_people_with_g_with_b)))
    print("bald_people_with_g_without_b " + str(len(bald_people_with_g_without_b)))
    print("bald_people_without_g_with_b " + str(len(bald_people_without_g_with_b)))
    print("bald_people_without_g_without_b " + str(len(bald_people_without_g_without_b)))

    haired_people = feat_df[(feat_df.Bald == 0) & (feat_df['Male'] == 1)]
    print("haired_people " + str(len(haired_people)))
    haired_people_with_g_with_b = haired_people[(haired_people["Eyeglasses"]==1) & (haired_people["No_Beard"]==0)]
    haired_people_with_g_without_b = haired_people[(haired_people["Eyeglasses"]==1) & (haired_people["No_Beard"]==1)]
    haired_people_without_g_with_b = haired_people[(haired_people["Eyeglasses"]==0) & (haired_people["No_Beard"]==0)]
    haired_people_without_g_without_b = haired_people[(haired_people["Eyeglasses"]==0) & (haired_people["No_Beard"]==1)]

    print("haired_people_with_g_with_b " + str(len(haired_people_with_g_with_b)))
    print("haired_people_with_g_without_b " + str(len(haired_people_with_g_without_b)))
    print("haired_people_without_g_with_b " + str(len(haired_people_without_g_with_b)))
    print("haired_people_without_g_without_b " + str(len(haired_people_without_g_without_b)))

    image_list = bald_people
    image_list = pd.concat([image_list, haired_people_with_g_with_b[:1136]])
    image_list = pd.concat([image_list, haired_people_with_g_without_b[:1136]])
    image_list = pd.concat([image_list, haired_people_without_g_with_b[:1136]])
    image_list = pd.concat([image_list, haired_people_without_g_without_b[:1136]])
    
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    BUFFER_SIZE = 3000
    BATCH_SIZE = 50
    NUM_IMAGES_USED = len(image_list)
    noise_dim = 256
    STEPS_PER_EPOCH = np.ceil(NUM_IMAGES_USED/BATCH_SIZE)
    CLASS_NAMES = celeba.features_name
    
    print('Total images ' + str(NUM_IMAGES_USED))

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

        return image, tf.convert_to_tensor(labels, dtype=tf.int32)

    #Create data set and map it. Could be improved if we can load images previously
    # and avoid mapping it later.
    labels = (list(image_list[features[0]][:NUM_IMAGES_USED]), list(image_list[features[1][:NUM_IMAGES_USED]]), list(image_list[features[2][:NUM_IMAGES_USED]]))
    image_id_with_labels = (list(image_list['image_id'][:NUM_IMAGES_USED]), labels)
    training_images = (tf.data.Dataset.from_tensor_slices(image_id_with_labels))

    training_dataset = training_images.map(_parse_function)
    
    #Shuffle and batch
    training_dataset = training_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    # print(f"Device being used {tf.test.gpu_device_name()}")


    num_features = 3
    generator = gan.Generator(noise_dim, num_features = num_features)
    generator.summary()
    discriminator = gan.Discriminator(num_features= num_features)
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

    EPOCHS = 100
    train_GEN = 3 #Train every batch
    train_DISC = 1 #Train every batch

    def train(g_model, d_model, gan_model, dataset,start_epoch = 0, latent_dim=100, n_epochs=100, train_GEN = 1, train_DISC = 1):
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
        cycle = 0 + (start_epoch*(STEPS_PER_EPOCH))
        start = time.time()
        for i in range(start_epoch, n_epochs):
            x = 0
            start_epoch = time.time()
            # enumerate batches over the training set
            for image_batch in dataset:
                n_batch = image_batch[0].shape[0]
                if cycle % 50 == 0:
                    print("Batch: " + str(x) +"/" + str(NUM_IMAGES_USED/n_batch))
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
                    X_fake = gan.generate_fake_samples(g_model, latent_dim, n_batch)
                    # create 'fake' class labels (0)
                    y_fake = np.zeros((n_batch, 1))
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
                            tf.summary.flush(writer=writer)
            # summarize loss on this batch
            print('Epoch: %d,  Loss: D_real = %.3f, D_fake = %.3f,  G = %.3f' %   (i+1, d_loss1, d_loss2, g_loss))
            print('Total time for epoch {} epochs is {} sec'.format(i, (time.time()-start_epoch)))
            # evaluate the model performance
            if (i+5) % 1 == 0:
                gan.summarize_performance(i, g_model, d_model, image_batch, latent_dim)     
            if (i+1) % 10 == 0:
            # Save the model every 10 epochs
                checkpoint.save(file_prefix = checkpoint_prefix)
        print ('Total time for training {} epoch is {} sec'.format(i, (time.time()-start)))

    checkpoint_dir = './training_checkpoints'

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator=generator,
                                    discriminator=discriminator)

    manager = tf.train.CheckpointManager(checkpoint, directory = checkpoint_dir, max_to_keep=3)

    # create the gan
    theGan = gan.define_gan(generator, discriminator)

    with tf.device('/device:GPU:0'):
        train(generator, discriminator, theGan, training_dataset, 0, noise_dim, EPOCHS, train_GEN, train_DISC)

    # checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
if __name__ == "__main__":
    main()