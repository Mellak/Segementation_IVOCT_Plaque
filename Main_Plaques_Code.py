import numpy as np
import pandas as pd
import sys
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from numpy import linalg
import math
from keras import optimizers
import os
from keras.preprocessing.image import ImageDataGenerator
os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN, device=cpu, floatX=float32, optimizer=fast_compile'
from keras import backend as K
import tensorflow as tf
from keras import models
from keras.optimizers import SGD, Adam
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import dice_help
import cv2
import Unet_model
import HRNet

# TODO: Merci Monsieur de ne changer vers un chemin global
path ='C:/Users/youne/OneDrive/Documents/Visual Studio 2015/Projects/Simulation/Simulation/Images/SegNetData/Lumiere_without_plaque'


#img_w et img_h, la taille de l'image d'entrée
img_w = 256
img_h = 256
n_labels = 4 # n_labels : le nombre des classes qu'on a plus une du background.
n_train = 2999 # nombre d'image d'entrainement
n_test = 299 # nombre d'image de test


# Une fonction d'aide pour préparer les masques
def label_map(labels):
    print(labels.shape)
    label_map = np.zeros([img_h, img_w, n_labels])
    for r in range(img_h):
        for c in range(img_w):
            label_map[r, c, labels[r][c]] = 1
    print(label_map.shape)
    return label_map

# TODO: Merci Monsieur de ne pas changer les noms de dossier... ils sont la comme mode.

#Une fonction pour préparer les données
# mode : soit préparation des données d'entrainement ou de test.
def prep_data(mode):
    assert mode in {'/MTestC', '/MTrainC'}, \
        'mode should be either \'test\' or \'train\''
    data = []
    label = []
    df = pd.read_csv(path + mode + '.csv')
    n = n_train if mode == '/MTrainC' else n_test
    for i, item in df.iterrows():
        if i >= n:
            break
        if( i < n/(n_labels-1)):
            img, mask = [imread(path + mode + '/' + item[2])], np.clip(imread(path + mode + '/' + item[1], ), 0,1)
            img = img[0]
            data.append(img)
            label.append(label_map(mask))
            sys.stdout.write('\r')
            sys.stdout.write(mode + ": [%-20s] %d%%" % ('=' * int(20. * (i + 1) / n - 1) + '>',
                                                        int(100. * (i + 1) / n)))
            sys.stdout.flush()
        elif  n/(n_labels-1)<=i< (2* n/(n_labels-1)) :
            img, mask2 = [imread(path + mode + '/' + item[2])], np.clip(imread(path + mode + '/' + item[1], ), 0,2)
            img = img[0]
            data.append(img)
            label.append(label_map(mask2))
            sys.stdout.write('\r')
            sys.stdout.write(mode + ": [%-20s] %d%%" % ('=' * int(20. * (i + 1) / n - 1) + '>',
                                                        int(100. * (i + 1) / n)))
            sys.stdout.flush()
        else :
            img, mask3 = [imread(path + mode + '/' + item[2])], np.clip(imread(path + mode + '/' + item[1], ), 0, 3)
            img = img[0]
            data.append(img)
            label.append(label_map(mask3))
            sys.stdout.write('\r')
            sys.stdout.write(mode + ": [%-20s] %d%%" % ('=' * int(20. * (i + 1) / n - 1) + '>',
                                                        int(100. * (i + 1) / n)))
    sys.stdout.write('\r')
    sys.stdout.flush()
    data, label = np.array(data), np.array(label).reshape((n, img_h * img_w, n_labels))

    print (mode + ': OK')
    print ('\tshapes: {}, {}'.format(data.shape, label.shape))
    print ('\ttypes:  {}, {}'.format(data.dtype, label.dtype))
    print ('\tmemory: {}, {} MB'.format(data.nbytes / 1048576, label.nbytes / 1048576))

    return data, label



''' Utilisation des models 


# Pour utiliser le SegNet Copier :
with open('model_MultiSeg.json') as model_file:
   autoencoder = models.model_from_json(model_file.read())

# Pour utiliser le HRNet Copier :
autoencoder = HRNet.seg_hrnet(batch_size=None, height=256, width=256, channel=3, classes=4)

#Pour utiliser le Unet Copier :
autoencoder = Unet_model.get_UNet(256,256)

#Pour utiliser le Unet++  Copier :
autoencoder = Unetpp.Unetpp()

'''

#Pour utiliser le SegNet
with open('model_MultiSeg.json') as model_file:
   autoencoder = models.model_from_json(model_file.read())

autoencoder.summary()

#J'ai changé l'optimizer vers Adam puisqu'il donne des meuilleurs résultats.
#optimizer = SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False)
optimizer = Adam(lr=1E-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

autoencoder.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=[dice_help.dice_coefv3])
#autoencoder.compile(loss=dice_help.dice_coef_loss, optimizer=optimizer, metrics=[dice_help.dice_coef])
print('Compiled: OK')

# TODO: Merci Monsieur de ne pas changer les noms de dossier... ils sont la comme mode.
train_data, train_label = prep_data('/MTrainC')
nb_epoch = 15
batch_size = 8
history = autoencoder.fit(train_data, train_label, batch_size=batch_size, epochs=nb_epoch, verbose=1)
autoencoder.save_weights('model_weights.hdf50')

#Si vous avez déja entrainer votre model on peut utiliser directement les poids
#autoencoder.load_weights('model_weights.hdf50')


# Lancement du test :
test_data, test_label = prep_data('/MTestC')
score = autoencoder.evaluate(test_data, test_label, verbose=0)
print('Test loss:', score[0])
print('Dice Coef:', score[1])
output = autoencoder.predict(test_data, verbose=0)
output = output.reshape((output.shape[0], img_h, img_w, n_labels))


# Sauvegarder les résultats.
# Merci de faire attention, qu'il y'a un décalage entre le test et les images résultats de 1.
# TODO: Merci Monsieur de changer save savePath le dossier ou vous voulez sauvegarder
# TODO: Merci Monsieur de changer save testPath le dossier contenant vos images de test
savePath='C:/Users/youne/OneDrive/Documents/Visual Studio 2015/Projects/Simulation/Simulation/Images/SegNetData/Lumiere_without_plaque/Prediction'
testPath = os.listdir('C:/Users/youne/OneDrive/Documents/Visual Studio 2015/Projects/Simulation/Simulation/Images/SegNetData/Lumiere_without_plaque/MinTest')

for file in testPath:
    img = np.squeeze(output[n,:,:,:])
    imsave(savePath + '/' + 'Pred' +str(n)+'.bmp', img)
    img = np.squeeze(output[n, :, :, 1])
    imsave(savePath + '/' + 'out1' + str(n) + '.bmp', img)
    img = np.squeeze(output[n, :, :, 2])
    imsave(savePath + '/' + 'out2' + str(n) + '.bmp', img)
    img = np.squeeze(output[n, :, :, 3])
    imsave(savePath + '/' + 'out3' + str(n) + '.bmp', img)
    n = n + 1
