from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization, ZeroPadding2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D,ZeroPadding2D
from keras import backend as K
import json
import os
import tensorflow as tf
from keras.layers.core import Activation, Reshape, Permute
from keras import models


img_w = 256
img_h = 256
n_labels = 4
kernel = 3
tf.keras.backend.set_image_data_format('channels_last')
print(tf.keras.backend.image_data_format())
if K.image_data_format() == 'channels_first':
    inputshape = (3, img_w, img_h)
else:
    inputshape = (img_w, img_h, 3)


encoding_layers = [
        #Input(padding=(0, 0), input_shape=inputshape, data_format='channels_last'),
       # Input(input_shape=inputshape, data_format='channels_last'),
ZeroPadding2D(padding=(0, 0), input_shape=inputshape, data_format='channels_last'),
        Convolution2D(32, kernel, padding='same',activation='relu'),
        Convolution2D(32, kernel, padding='same',activation='relu'),
        MaxPooling2D((2, 2)),

        Convolution2D(64, kernel, padding='same',activation='relu'),
        Convolution2D(64, kernel, padding='same',activation='relu'),
        MaxPooling2D((2, 2)),

        Convolution2D(128, kernel, padding='same',activation='relu'),
        Convolution2D(128, kernel, padding='same',activation='relu'),
        MaxPooling2D((2, 2)),

        Convolution2D(256, kernel, padding='same',activation='relu'),
        Convolution2D(256, kernel, padding='same',activation='relu'),
        MaxPooling2D((2, 2)),

        Convolution2D(512, kernel, padding='same',activation='relu'),
        Convolution2D(512, kernel, padding='same',activation='relu'),
    ]
autoencoder = models.Sequential()
autoencoder.encoding_layers = encoding_layers
for l in autoencoder.encoding_layers:
    autoencoder.add(l)
    print(l.input_shape, l.output_shape, l)

decoding_layers = [
    UpSampling2D(),
    Convolution2D(256, kernel, padding='same',activation='relu'),
    Convolution2D(256, kernel, padding='same',activation='relu'),


    UpSampling2D(),
    Convolution2D(128, kernel, padding='same',activation='relu'),
    Convolution2D(128, kernel, padding='same',activation='relu'),


    UpSampling2D(),
    Convolution2D(64, kernel, padding='same',activation='relu'),
    Convolution2D(64, kernel, padding='same',activation='relu'),

    UpSampling2D(),
    Convolution2D(32, kernel, padding='same',activation='relu'),
    Convolution2D(32, kernel, padding='same',activation='relu'),


    Convolution2D(n_labels, 1, 1, padding='valid'),
]


autoencoder.decoding_layers = decoding_layers
for l in autoencoder.decoding_layers:
    autoencoder.add(l)
autoencoder.summary()
autoencoder.add(Reshape(target_shape=(n_labels, img_h * img_w), input_shape=inputshape))
autoencoder.add(Permute((2, 1)))
autoencoder.add(Activation('softmax'))
with open('model_unet.json', 'w') as outfile:
    outfile.write(json.dumps(json.loads(autoencoder.to_json()), indent=2))
autoencoder.summary()
'''
def get_UNet(img_rows, img_cols,channels):


    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(n_labels, (1, 1), padding='valid', activation='softmax')(conv9)
    model = Model(inputs=[inputs], outputs=[conv10])

    return model



#get_UNet(256,256,3).summary()
#with open('model_unet.json', 'w') as outfile:
    #outfile.write(json.dumps(json.loads(get_UNet(256,256).to_json()), indent=2))
'''