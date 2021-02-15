from keras import models
from keras.layers.core import Activation, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D,ZeroPadding2D
from keras.layers.normalization import BatchNormalization
import json
from keras import backend as K
import tensorflow as tf
import os
import tensorflow as tf


'''https://github.com/imlab-uiip/keras-segnet'''





tf.config.list_physical_devices(device_type=None)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices(('GPU'))))

print(tf.test.is_built_with_cuda)
import os


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")
with tf.device('/device:GPU:0'):
    from tensorflow.python.client import device_lib

    print(tf.config.experimental.list_physical_devices)
    img_w = 256
    img_h = 256
    n_labels = 4

    kernel = 3
    from tensorflow.python.client import device_lib

    print(tf.config.list_physical_devices('GPU'))

    print(device_lib.list_local_devices())


    def get_available_gpus():
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']


    print(get_available_gpus())
    # print(device_lib.list_local_devices())
    tf.keras.backend.set_image_data_format('channels_last')
    print(tf.keras.backend.image_data_format())
    if K.image_data_format() == 'channels_first':
        inputshape = (3, img_w, img_h)
    else:
        inputshape = (img_w, img_h, 3)

    print(inputshape)
    encoding_layers = [
        ZeroPadding2D(padding=(0, 0), input_shape=inputshape, data_format='channels_last'),
        # Convolution2D(64, kernel, padding='same', input_shape=inputshape,data_format='channels_first'),
        Convolution2D(64, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(64, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),

        Convolution2D(128, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(128, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2), padding='same'),

        Convolution2D(256, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(256, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(256, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2), padding='same'),

        Convolution2D(512, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2), padding='same'),

        Convolution2D(512, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2), padding='same'),
    ]

    autoencoder = models.Sequential()
    autoencoder.encoding_layers = encoding_layers

    for l in autoencoder.encoding_layers:
        autoencoder.add(l)
        print(l.input_shape, l.output_shape, l)

    decoding_layers = [
        UpSampling2D(),
        Convolution2D(512, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(),
        Convolution2D(512, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(256, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(),
        # ZeroPadding2D(((1, 0), (0, 0))),
        Convolution2D(256, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(256, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(128, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(),
        Convolution2D(128, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(64, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(),
        Convolution2D(64, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(n_labels, 1, 1, padding='valid'),
        BatchNormalization(),
    ]
    autoencoder.decoding_layers = decoding_layers
    for l in autoencoder.decoding_layers:
        autoencoder.add(l)
    autoencoder.summary()
    autoencoder.add(Reshape(target_shape=(n_labels, img_h * img_w), input_shape=inputshape))
    autoencoder.add(Permute((2, 1)))
    autoencoder.add(Activation('softmax'))
    with open('model_MultiSeg.json', 'w') as outfile:
        outfile.write(json.dumps(json.loads(autoencoder.to_json()), indent=2))

    autoencoder.summary()
