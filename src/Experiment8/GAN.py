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
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn
from numpy.random import randint

def Generator(latent_dim):
      model = Sequential()
      
      n_nodes = 128 * 8 * 8
      model.add(Dense(n_nodes, input_dim=latent_dim))
      model.add(LeakyReLU(alpha=0.2))
      model.add(Reshape((8, 8, 128)))
      # upsample to 16x16
      model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
      model.add(LeakyReLU(alpha=0.2))
      # upsample to 32x32
      model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
      model.add(LeakyReLU(alpha=0.2))
      # upsample to 64x64
      model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
      model.add(LeakyReLU(alpha=0.2))
      # upsample to 128x128
      model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
      model.add(LeakyReLU(alpha=0.2))
      # output layer 128x128x3
      model.add(Conv2D(3, (5,5), activation='tanh', padding='same'))
      return model

#input of G
def generate_latent_points(latent_dim, n_samples):
      # generate points in the latent space
      x_input = randn(latent_dim * n_samples)
      # reshape into a batch of inputs for the network
      x_input = x_input.reshape(n_samples, latent_dim)
      return x_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
      # generate points in latent space
      x_input = generate_latent_points(latent_dim, n_samples)
      # predict outputs
      X = g_model.predict(x_input)
      # create 'fake' class labels (0)
      y = np.zeros((n_samples, 1))
      return X, y

def Discriminator(in_shape=(128,128,3)):
      model = Sequential()
      # normal
      model.add(Conv2D(128, (5,5), padding='same', input_shape=in_shape))
      model.add(LeakyReLU(alpha=0.2))
      # downsample to 64x64
      model.add(Conv2D(128, (5,5), strides=(2,2), padding='same'))
      model.add(LeakyReLU(alpha=0.2))
      # downsample to 32x32
      model.add(Conv2D(128, (5,5), strides=(2,2), padding='same'))
      model.add(LeakyReLU(alpha=0.2))
      # downsample to 16x16
      model.add(Conv2D(128, (5,5), strides=(2,2), padding='same'))
      model.add(LeakyReLU(alpha=0.2))
      # downsample to 8x8
      model.add(Conv2D(128, (5,5), strides=(2,2), padding='same'))
      model.add(LeakyReLU(alpha=0.2))
      # classifier
      model.add(Flatten())
      model.add(Dropout(0.4))
      model.add(Dense(1, activation='sigmoid'))
      # compile model
      opt = Adam(lr=0.0002, beta_1=0.5)
      model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
      return model

def define_gan(g_model, d_model):
      # make weights in the discriminator not trainable
      d_model.trainable = False
      # connect them
      model = Sequential()
      # add generator
      model.add(g_model)
      # add the discriminator
      model.add(d_model)
      # compile model
      opt = Adam(lr=0.0002, beta_1=0.5)
      model.compile(loss='binary_crossentropy', optimizer=opt)
      return model

# create and save a plot of generated images
def show_generated(generated,epoch, n=5):
      plt.figure(figsize=(10,10))
      for i in range(n * n):
          plt.subplot(n, n, i + 1)
          plt.imshow(generated[i])
          plt.axis('off')
      plt.savefig('image_at_epoch_{:04d}.png'.format(epoch+1))
      plt.show()    

# evaluate the discriminator and plot generated images
def summarize_performance(epoch, g_model, d_model, image_batch, latent_dim, n_samples=100):
      # prepare real samples
      X_real = image_batch
      y_real = np.ones((image_batch[0].shape[0], 1))
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