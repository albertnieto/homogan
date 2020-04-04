import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

import Generator as Gen
import Discriminator as Discrim

class GAN():
    def __init__(self, img_height = 128, img_width = 128, channels = 3, latent_dim = (1,1,256), sampling_period = 100):
        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        self.img_shape = (self.img_height, self.img_width, self.channels)
        self.latent_dim = latent_dim
        self.sampling_period = sampling_period
        self.gLoss = 0
        self.dLoss = 0

        optimizer = Adam(1e-4)

        self.discriminator = Discrim.Discriminator(self.img_shape).model()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        self.generator = Gen.Generator(self.latent_dim).model()
        inp = Input(shape=self.latent_dim)
        gen_img = self.generator(inp)

        #This ensures that when we combine our networks we only train the Generator.
        self.discriminator.trainable = False
        validity = self.discriminator(gen_img)

        self.combined = Model(inp, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


    def train_model(self, batch_size, imgs, cycle):
        valid = tf.ones((batch_size, 1))
        fake = tf.zeros((batch_size, 1))   
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

            noise = tf.random.normal([batch_size,self.latent_dim[0], self.latent_dim[1], self.latent_dim[2]])
            gen_imgs = self.generator(noise) 

            #Train the Discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            self.dLoss = 0.5 * tf.math.add(d_loss_real, d_loss_fake)
            acc = 100*self.dLoss[1]

            #Train the Generator
            noise = tf.random.normal([batch_size,self.latent_dim[0], self.latent_dim[1], self.latent_dim[2]])
            self.gLoss = self.combined.train_on_batch(noise, valid)

        grads_of_gen = gen_tape.gradient(self.gLoss, self.generator.trainable_variables)
        grads_of_disc = disc_tape.gradient(self.dLoss, self.discriminator.trainable_variables)

        if cycle % self.sampling_period == 0:
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (cycle, self.dLoss[0], acc, self.gLoss))

        return self.dLoss[0], self.gLoss, acc, grads_of_gen, grads_of_disc
    
    def gen_Draw(self, number_of_images = 1):
        noise = tf.random.normal([number_of_images,self.latent_dim[0], self.latent_dim[1], self.latent_dim[2]])
        return self.generator.predict(noise)
    
    # def get_grads(self):
    #     gen_grads = K.gradients(self.generator.output, self.generator.input)
    #     disc_grads = K.gradients(self.discriminator.output, self.discriminator.input)

    #     return gen_grads, disc_grads