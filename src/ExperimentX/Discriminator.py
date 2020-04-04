import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, Flatten, Dense, Input, LeakyReLU
from tensorflow.keras.models import Model

class Discriminator(tf.keras.Model):
    def __init__(self, input_shape, output_shape=1):
        super(Discriminator, self).__init__()
        self.discriminator_input_shape = input_shape
        self.conv1 = Conv2D(32, (5, 5), padding='same', input_shape=[128, 128, 3], name='block1_conv1')
        self.batchNorm1 = BatchNormalization()
        self.leakyRelu1 = LeakyReLU()
        self.dropout1 = Dropout(0.5)

        self.conv3 = Conv2D(128, (3, 3), strides=(2, 2), padding='same', name='block1_conv3')
        self.batchNorm3 = BatchNormalization()
        self.leakyRelu3 = LeakyReLU()
        self.dropout3 = Dropout(0.5)

        self.conv4 = Conv2D(256, (3, 3), strides=(2, 2), padding='same', name='block1_conv4')
        self.batchNorm4 = BatchNormalization()
        self.leakyRelu4 = LeakyReLU()
        self.dropout4 = Dropout(0.5)

        self.flatten = Flatten()
        self.dense1 = Dense(output_shape)
    
    def call(self, inputs):
        x = self.dropout1(self.leakyRelu1(self.batchNorm1(self.conv1(inputs))))
        x = self.dropout3(self.leakyRelu3(self.batchNorm3(self.conv3(x))))
        x = self.dropout4(self.leakyRelu4(self.batchNorm4(self.conv4(x))))

        x = self.flatten(x)
        return self.dense1(x)
    
    #https://github.com/tensorflow/tensorflow/issues/25036
    def model(self):
        x = Input(shape=self.discriminator_input_shape)
        model = Model(inputs=[x], outputs=self.call(x))
        model.summary()
        return model