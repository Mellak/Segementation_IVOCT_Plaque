import keras.backend as K
import numpy as np

# Subroutine that computes the Dice coefficient
# from true and predicted binary images
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return 2 * K.sum(y_true_f * y_pred_f) / (K.sum(y_true_f) + K.sum(y_pred_f))
# INSERT CODE

# Subroutine that computes the Dice coefficient loss
def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)

#Dice coefficient pour une image a plusieurs canaux

def dice_coefv3(y_true, y_pred):
  intersection = K.sum(y_true * y_pred, axis=-1)
  union = K.sum(y_true, axis=-1) + K.sum(y_pred, axis=-1)
  dice = 2*intersection/union
  return dice

def dice_coef_lossv3(y_true, y_pred):
    return 1.-dice_coefv3(y_true, y_pred)