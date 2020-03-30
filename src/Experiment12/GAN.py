import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Concatenate
from tensorflow.keras import Model
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn
from numpy.random import randint

def Generator(latent_dim, n_classes = 2):
      in_label = Input(shape=(1,))
      # embedding for categorical input
      li = Embedding(n_classes, 50)(in_label)
      # linear multiplication
      n_nodes = 8 * 8
      li = Dense(n_nodes)(li)
      # reshape to additional channel
      li = Reshape((8, 8, 1))(li)
      
      n_nodes = 128 * 8 * 8
      # image generator input
      in_lat = Input(shape=(1,1,latent_dim))
      x = Conv2DTranspose(2048, kernel_size=(1,1), padding="same")(in_lat)
      x = LeakyReLU(alpha=0.2)(x)
      x = Reshape((8, 8, 128))(x)

      # merge image gen and label input
      merge = Concatenate()([x, li])

      # upsample to 16x16
      x = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(merge)
      x = LeakyReLU(alpha=0.2)(x)
      # upsample to 32x32
      x = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(x)
      x = LeakyReLU(alpha=0.2)(x)
      # upsample to 64x64
      x = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(x)
      x = LeakyReLU(alpha=0.2)(x)
      # upsample to 128x128
      x = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(x)
      x = LeakyReLU(alpha=0.2)(x)
      # output layer 128x128x3
      out = Conv2D(3, (5,5), activation='tanh', padding='same')(x)

      # define model
      model = Model([in_lat, in_label], out)
      return model

#input of G
def generate_latent_points(latent_dim, n_samples, n_classes=2):
      # generate points in the latent space
      x_input = randn(latent_dim * n_samples)
      # reshape into a batch of inputs for the network
      x_input = x_input.reshape(n_samples,1,1, latent_dim)
      # generate labels
      labels = randint(0, n_classes, n_samples)
      return [x_input, labels]

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
      # generate points in latent space
      x_input, x_labels = generate_latent_points(latent_dim, n_samples)
      # predict outputs
      images = g_model.predict([x_input,x_labels])
      # create 'fake' class labels (0)
      y = np.zeros((n_samples, 1))
      return [images, x_labels], y

def Discriminator(in_shape=(128,128,3), n_classes = 2):
  	  # label input 
      in_label = Input(shape=(1,))
      # embedding for categorical input
      emb = Embedding(n_classes, 25)(in_label)
      # scale up to image dimensions with linear activation
      n_nodes = in_shape[0] * in_shape[1]
      emb = Dense(n_nodes)(emb)
      # reshape to additional channel
      li = Reshape((in_shape[0], in_shape[1], 1))(emb)
      # image input
      in_image = Input(shape=in_shape)	
      # concat label as a channel
      merge = Concatenate()([in_image, li])
      # normal
      x = Conv2D(128, (5,5), padding='same', input_shape=in_shape)(merge)
      x = LeakyReLU(alpha=0.2)(x)
      # downsample to 64x64
      x = Conv2D(128, (5,5), strides=(2,2), padding='same')(x)
      x = LeakyReLU(alpha=0.2)(x)
      # downsample to 32x32
      x = Conv2D(128, (5,5), strides=(2,2), padding='same')(x)
      x = LeakyReLU(alpha=0.2)(x)
      # downsample to 16x16
      x = Conv2D(128, (5,5), strides=(2,2), padding='same')(x)
      x = LeakyReLU(alpha=0.2)(x)
      # downsample to 8x8
      x = Conv2D(128, (5,5), strides=(2,2), padding='same')(x)
      x = LeakyReLU(alpha=0.2)(x)
      # classifier
      x = Flatten()(x)
      x = Dropout(0.4)(x)
      out = Dense(1, activation='sigmoid')(x)
      model = Model([in_image, in_label], out)
      # compile model
      opt = Adam(lr=0.0002, beta_1=0.5)
      model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
      return model

def define_gan(g_model, d_model):
      # make weights in the discriminator not trainable
      d_model.trainable = False
      # get noise and label inputs from generator model
      gen_noise, gen_label = g_model.input
      # get image output from the generator model
      gen_output = g_model.output
      # connect image output and label input from generator as inputs to discriminator
      gan_output = d_model([gen_output, gen_label])
      # define gan model as taking noise and label and outputting a classification
      model = Model([gen_noise, gen_label], gan_output)
      # compile model
      opt = Adam(lr=0.0002, beta_1=0.5)
      model.compile(loss='binary_crossentropy', optimizer=opt)
      return model

# create and save a plot of generated images
def show_generated(generated,epoch, n=3):
      plt.figure(figsize=(10,10))
      for i in range(n * n):
        gender = "Male" + str(generated[1][i])
        if generated[1][i] == 0:
          gender = "Female" + str(generated[1][i])
        plt.subplot(n, n, i + 1, title = gender)
        plt.imshow(generated[0][i])
        plt.axis('off')
      plt.savefig('image_at_epoch_{:04d}.png'.format(epoch+1))
      plt.show()

# evaluate the discriminator and plot generated images
def summarize_performance(epoch, g_model, d_model, image_batch, latent_dim, n_samples=100):
      n_samples = image_batch[0].shape[0]
      # prepare real samples
      X_real = image_batch
      y_real = np.ones((n_samples, 1))
      # evaluate discriminator on real examples
      _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
      # prepare fake examples
      x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
      # evaluate discriminator on fake examples
      _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
      # summarize discriminator performance
      print('>Accuracy [real: %.0f%%, fake: %.0f%%]' % (acc_real*100, acc_fake*100))
      # show plot
      show_generated(x_fake, epoch)  