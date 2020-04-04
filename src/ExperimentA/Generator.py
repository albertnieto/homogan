import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Dropout, Input, LeakyReLU
from tensorflow.keras.models import Model

class Generator(tf.keras.Model):
    def __init__(self, generator_input_shape=(1,1,256)):
        super(Generator, self).__init__()
        self.generator_input_shape = generator_input_shape
        
        self.unconv1 = Conv2DTranspose(1024, kernel_size=(1,1), padding="same", use_bias=False)
        self.batchNorm1 = BatchNormalization()
        self.leakyRelu1 = LeakyReLU()

        self.unconv2 = Conv2DTranspose(128, kernel_size=(8,8), padding="valid", use_bias=False)
        self.batchNorm2 = BatchNormalization()
        self.leakyRelu2 = LeakyReLU()

        self.unconv3 = Conv2DTranspose(128, kernel_size=(5,5) ,  strides=(2,2), padding="same", use_bias=False)
        self.batchNorm3 = BatchNormalization()
        self.leakyRelu3 = LeakyReLU()

        self.unconv4 = Conv2DTranspose(32, kernel_size=(5,5) ,  strides=(2,2), padding="same", use_bias=False)
        self.batchNorm4 = BatchNormalization()
        self.leakyRelu4 = LeakyReLU()
                
        self.unconv5 = Conv2DTranspose(32, kernel_size=(3,3) ,  strides=(2,2), padding="same", use_bias=False)
        self.conv1 = Conv2D( 64  , ( 1 , 1 ) , padding='same', name="block_6", use_bias=False)
        self.batchNorm5 = BatchNormalization()
        self.leakyRelu5 = LeakyReLU()

        self.unconv6 = Conv2DTranspose(32, kernel_size=(3,3) ,  strides=(2,2), padding="same", use_bias=False)
        self.conv2 = Conv2D( 128  , ( 1 , 1 ) , padding='same', name="block_7", use_bias=False)
        self.batchNorm6 = BatchNormalization()
        self.leakyRelu6 = LeakyReLU()

        self.img = Conv2D( 3 , ( 1 , 1 ) , activation='tanh' , padding='same', name="final_block")

    def call(self, inputs):
        x = self.leakyRelu1(self.batchNorm1(self.unconv1(inputs)))

        x = self.leakyRelu2(self.batchNorm2(self.unconv2(x)))

        x = self.unconv3(x)
        x = self.leakyRelu3(self.batchNorm3(x))

        x = self.unconv4(x)
        x = self.leakyRelu4(self.batchNorm4(x))

        x = self.unconv5(x)
        x = self.leakyRelu5(self.batchNorm5(self.conv1(x)))

        x = self.unconv6(x)
        x = self.leakyRelu6(self.batchNorm6(self.conv2(x)))

        return self.img(x)

    #https://github.com/tensorflow/tensorflow/issues/25036
    def model(self):
        x = Input(shape=self.generator_input_shape)
        model = Model(inputs=[x], outputs=self.call(x))
        model.summary()
        return model