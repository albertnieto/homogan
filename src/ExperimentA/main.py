from __future__ import absolute_import, division, print_function, unicode_literals
#from IPython import display

import tensorflow as tf
from tensorflow.keras import layers, models
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
#print(len(list(data_dir.glob('*/*.jpg'))))


IMG_HEIGHT = 128
IMG_WIDTH = 128
BUFFER_SIZE = 3000
BATCH_SIZE = 16
NUM_IMAGES_USED = 10000
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

# print(f"Device being used {tf.test.gpu_device_name()}")

# generator_optimizer = tf.keras.optimizers.Adam(1e-4)
# discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

theGAN = gan.GAN()
generated_image = theGAN.gen_Draw()

plt.imshow(generated_image[0, :, :, 0], cmap='gray')
plt.show()
#generator.summary()

# discriminator = gan.Discriminator((IMG_HEIGHT, IMG_WIDTH, 3)).model()

#discriminator.summary()


checkpoint_dir = './training_checkpoints'

# if not os.path.exists(checkpoint_dir):
#     os.makedirs(checkpoint_dir)

# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
#                                  discriminator_optimizer=discriminator_optimizer,
#                                  generator=generator,
#                                  discriminator=discriminator)

# manager = tf.train.CheckpointManager(checkpoint, directory = checkpoint_dir, max_to_keep=3)

EPOCHS = 20
noise_dim = 256
num_examples_to_generate = 4


# def generate_and_save_images(model, epoch, test_input, titleadd = ""):
#     # Notice `training` is set to False.
#     # This is so all layers run in inference mode (batchnorm).
#     predictions = model(test_input, training=False)

#     fig = plt.figure(figsize=(10,5))

#     for i, img in enumerate(predictions):
#         ax = fig.add_subplot(1,4,i+1)
#         ax.imshow(img)
#         fig.suptitle("Generated images "+titleadd,fontsize=30)

#     plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
#     #plt.show()

def train(dataset, epochs):
    # checkpoint.restore(manager.latest_checkpoint)
    # if manager.latest_checkpoint:
    #     print("Restored from {}".format(manager.latest_checkpoint))
    # else:
    #     print("Initializing from scratch.")

    logdir = '.\\logs\\func\\'
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    writer = tf.summary.create_file_writer(logdir)

    tf.summary.trace_on(graph=True, profiler=True)

    cycle = 0
    start_train = time.time()
    for epoch in range(epochs):
        start = time.time()
        i = 0
        for image_batch in dataset:
            discL, genL, acc, gen_grad, disc_grad = theGAN.train_model(BATCH_SIZE,image_batch,cycle)

            i += 1
            cycle += 1
            if i % 100 == 0:
                print(f"Batch {i}/{STEPS_PER_EPOCH}")

                with writer.as_default():
                    tf.summary.scalar('Accuracy', acc, step=cycle)
                    tf.summary.scalar('Gen Loss', genL, step=cycle)
                    tf.summary.scalar('Disc Loss', discL, step=cycle)
        # if (time.time() - start_train) > 10*60: #Log cada 10 min
        #     start_train = time.time()
                    for grad in gen_grad:
                        tf.summary.histogram('Gen_grad', grad, step=cycle)

        theGAN.gen_Draw()
        # generate_and_save_images(generator,
        #                          epoch + 1,
        #                          seed,titleadd="Epoch {}".format(epoch))

        # Save the model every 20 epochs
        # if (epoch + 1) % 20 == 0 or epoch == epochs-1:
        #     checkpoint.save(file_prefix = checkpoint_prefix)


        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    # generate_and_save_images(generator,
    #                        epochs,
    #                        seed)

# print(manager.latest_checkpoint)

with tf.device('/device:GPU:0'):
    train(training_dataset, EPOCHS)

# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))