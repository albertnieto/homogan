from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras import Model
from numpy.random import randn
from numpy.random import randint

def Generator(latent_dim=256, num_features=1):
      in_label = Input(shape=(num_features,))
      n_nodes = 8 * 8
      li = Dense(n_nodes)(in_label)
      # reshape to additional channel
      li = Reshape((8, 8, 1))(li)
      
      # image generator input
      in_lat = Input(shape=(1,1,latent_dim))
      x = Conv2DTranspose(8192, kernel_size=(1,1), padding="same")(in_lat)
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

def Discriminator(num_features=1, in_shape=(128,128,3)):
  	  # label input 
      in_label = Input(shape=(num_features,))
      # scale up to image dimensions with linear activation
      emb = Dense(in_shape[0] * in_shape[1])(in_label)
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