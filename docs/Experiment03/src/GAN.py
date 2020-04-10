import tensorflow as tf
from tensorflow.keras import layers, models

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
        #self.conv1 = layers.Conv2D( 64  , ( 1 , 1 ) , padding='same', name="block_4", use_bias=False)
        self.batchNorm3 = layers.BatchNormalization()
        self.leakyRelu3 = layers.LeakyReLU()

        self.unconv2 = layers.Conv2DTranspose(32, kernel_size=(5,5) ,  strides=(2,2), padding="same", use_bias=False)
        #self.conv2 = layers.Conv2D( 64  , ( 1 , 1 ) , padding='same', name="block_5", use_bias=False)
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
        x = self.leakyRelu3(self.batchNorm3(x))

        x = self.unconv2(x)
        x = self.leakyRelu4(self.batchNorm4(x))

        x = self.unconv3(x)
        x = self.leakyRelu5(self.batchNorm5(self.conv3(x)))

        x = self.unconv4(x)
        x = self.leakyRelu6(self.batchNorm6(self.conv4(x)))

        return self.img(x)

    #https://github.com/tensorflow/tensorflow/issues/25036
    def model(self):
        x = layers.Input(shape=self.generator_input_shape)
        return models.Model(inputs=[x], outputs=self.call(x))

class Discriminator(tf.keras.Model):
    def __init__(self, input_shape, output_shape=1):
        super(Discriminator, self).__init__()
        self.discriminator_input_shape = input_shape
        self.conv1 = layers.Conv2D(32, (5, 5), padding='same', input_shape=[128, 128, 3], name='block1_conv1')
        self.batchNorm1 = layers.BatchNormalization()
        self.leakyRelu1 = layers.LeakyReLU()
        self.dropout1 = layers.Dropout(0.5)

        # self.conv2 = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', name='block1_conv2')
        # self.batchNorm2 = layers.BatchNormalization()
        # self.leakyRelu2 = layers.LeakyReLU()
        # self.dropout2 = layers.Dropout(0.5)

        self.conv3 = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', name='block1_conv3')
        self.batchNorm3 = layers.BatchNormalization()
        self.leakyRelu3 = layers.LeakyReLU()
        self.dropout3 = layers.Dropout(0.5)

        self.conv4 = layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same', name='block1_conv4')
        self.batchNorm4 = layers.BatchNormalization()
        self.leakyRelu4 = layers.LeakyReLU()
        self.dropout4 = layers.Dropout(0.5)

        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(output_shape)
    
    def call(self, inputs):
        x = self.dropout1(self.leakyRelu1(self.batchNorm1(self.conv1(inputs))))
        # x = self.dropout2(self.leakyRelu2(self.batchNorm2(self.conv2(x))))
        x = self.dropout3(self.leakyRelu3(self.batchNorm3(self.conv3(x))))
        x = self.dropout4(self.leakyRelu4(self.batchNorm4(self.conv4(x))))

        x = self.flatten(x)
        return self.dense1(x)
    
    #https://github.com/tensorflow/tensorflow/issues/25036
    def model(self):
        x = layers.Input(shape=self.discriminator_input_shape)
        return models.Model(inputs=[x], outputs=self.call(x))