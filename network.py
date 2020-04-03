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
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras import Model
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn
from numpy.random import randint

#input of G
def generate_latent_points(latent_dim, n_samples, n_classes=2):
      # generate points in the latent space
      x_input = randn(latent_dim * n_samples)
      # reshape into a batch of inputs for the network
      x_input = x_input.reshape(n_samples, latent_dim)
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