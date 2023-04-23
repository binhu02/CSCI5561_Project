from keras import layers
import keras
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras import callbacks
from scipy.ndimage.filters import gaussian_filter
import tensorflow as tf
import random
from tqdm import tqdm
from skimage.util import random_noise

DATASET_PATH = 'dataset'
TRAIN_X_PATH = os.path.join(DATASET_PATH, 'X_train_input')
VALID_X_PATH = os.path.join(DATASET_PATH, 'X_test_input')
TRAIN_Y_PATH = os.path.join(DATASET_PATH, 'X_train_target')
VALID_Y_PATH = os.path.join(DATASET_PATH, 'X_test_target')

LEARN_RATE = 3e-4
SSIM_SCALE = 0.1
MAE_SCALE = 0.25
LR_FACTOR = 0.8
NUM_EPOCHS = 1

NUM_TRAINING = 30000
NUM_TESTING = 5000
BATCH_SIZE = 8

def SSIMLoss(y_true, y_pred):
  return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

def MSE_SSIM(y_true, y_pred):
    return SSIM_SCALE * SSIMLoss(y_true, y_pred) + (1 - SSIM_SCALE) * tf.reduce_mean(tf.math.squared_difference(y_true, y_pred))

def MSE_MAE(y_true, y_pred):
    return MAE_SCALE * tf.reduce_mean(tf.math.abs(y_true - y_pred)) +  (1 - MAE_SCALE) * tf.reduce_mean(tf.math.squared_difference(y_true, y_pred))

DATASET_PATH = 'BRATS_2018_ALL'
TRAIN_X_PATH = os.path.join(DATASET_PATH, 'X_train_input')
VALID_X_PATH = os.path.join(DATASET_PATH, 'X_test_input')
TRAIN_Y_PATH = os.path.join(DATASET_PATH, 'X_train_target')
VALID_Y_PATH = os.path.join(DATASET_PATH, 'X_test_target')

def image_loader(mode, x_or_y, index):
    """
    Return the numpy image given the information.
    mode: 'train', 'valid' or 'test'
    x_or_y: 'x' or 'y', 'x' stands for the input and 'y' stands for the target
    index: int, the index of the image
    """
    if mode == 'train':
        if x_or_y == 'x':
            filepath = os.path.join(TRAIN_X_PATH, os.path.basename(TRAIN_X_PATH)+'_'+str(index)+'.npy')
        if x_or_y == 'y':
            filepath = os.path.join(TRAIN_Y_PATH, os.path.basename(TRAIN_Y_PATH)+'_'+str(index)+'.npy')
    elif mode == 'valid':
        if x_or_y == 'x':
            filepath = os.path.join(VALID_X_PATH, os.path.basename(VALID_X_PATH)+'_'+str(index)+'.npy')
        if x_or_y == 'y':
            filepath = os.path.join(VALID_Y_PATH, os.path.basename(VALID_Y_PATH)+'_'+str(index)+'.npy')
    else:
        raise ValueError("The first or the second parameter is not valid")
        
    if not isinstance(index, int):
        raise TypeError("Index should be an integer")
    
    if mode == 'train' and (index < 0 or index > 35339):
            raise IndexError("Image index out of range 0 - 35339")
            
    if mode == 'valid' and (index < 0 or index > 8834):
            raise IndexError("Image index out of range 0 - 8834")
    
    return np.load(filepath)

def gen_noisy_image(x, mode):
    mean = 0
    sigma = random.uniform(0, 2)
    scale = random.uniform(0, 0.2)
    img = image_loader(mode, 'x', x)
    noisy_img = np.zeros(img.shape)
    for x in range(img.shape[2]):
        random_dist = (1 - (scale * np.random.normal(mean, sigma, (img.shape[0], img.shape[1]))))
        noisy_img[:, :, x] = img[:, :, x] * random_dist
    return np.clip(noisy_img, 0., 1.)

class Data_Generator(keras.utils.Sequence):
  
  def __init__(self, image_num, mode, batch_size):
    self.image_num = image_num
    self.mode = mode
    self.batch_size = batch_size
    
  def __len__(self):
    return self.image_num // self.batch_size
  
  def __getitem__(self, idx):
    batch_x = [gen_noisy_image(x, self.mode) for x in range(idx * self.batch_size, (idx+1) * self.batch_size)]
    batch_y = [gen_noisy_image(x, self.mode) for x in range(idx * self.batch_size, (idx+1) * self.batch_size)]
    
    return np.array(batch_x), np.array(batch_y)

# get some data in main memory for visualizing
def get_data(n_samples_train = 500, n_samples_test = 250, overwrite = True):
  if not overwrite:
    x_train = np.load("x_train.npy")
    x_train_target = np.load("x_train_targ.npy")
    x_test = np.load("x_test.npy")
    x_test_target = np.load("x_test_targ.npy")
  else:
    x_train = np.zeros((n_samples_train, 240, 240, 4))
    x_train_target = np.zeros((n_samples_train, 240, 240, 4))
    x_test = np.zeros((n_samples_test, 240, 240, 4))
    x_test_target = np.zeros((n_samples_test, 240, 240, 4))
    for x in tqdm(range(x_train.shape[0]), 'Train data'):
      x_train[x] = gen_noisy_image(x, 'train')
      x_train_target[x] = gen_noisy_image(x, 'train')
    for x in tqdm(range(x_test.shape[0]), 'Test data'):
      x_test_target[x] = gen_noisy_image(x, 'valid')
      x_test[x] = gen_noisy_image(x, 'valid')
    np.save("./x_train.npy", x_train)
    np.save("./x_train_targ.npy", x_train_target)
    np.save("./x_test.npy", x_test)
    np.save("./x_test_targ.npy", x_test_target)
  return x_train, x_train_target, x_test, x_test_target

training_generator = Data_Generator(NUM_TRAINING, 'train', BATCH_SIZE)
testing_generator = Data_Generator(NUM_TESTING, 'valid', BATCH_SIZE)

x_train, x_train_target, x_test, x_test_target = get_data()

n = 5
offset = 200
for i in range(1, n + 1):
    ax = plt.subplot(2, n, i)
    ax.imshow(x_test[i + offset])
    ax2 = plt.subplot(2, n, i+n)
    ax2.imshow(x_test_target[i + offset])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
plt.tight_layout()
plt.show()

def approx_gelu(x):
    return 0.5 * x * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))

#activation_func = keras.activations.gelu(approximate=True)
def autoencoder_model(input_shape, activation_func=approx_gelu, loss_func='mse', learn_rate=LEARN_RATE):

    input_img = keras.Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), activation=approx_gelu, padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (5, 5), activation=approx_gelu, padding='same')(x)
    x = layers.MaxPooling2D((3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(16, (7, 7), activation=approx_gelu, padding='same')(x)
    encoded = layers.MaxPooling2D((4, 4), padding='same')(x)

    x = layers.Conv2D(16, (7, 7), activation=approx_gelu, padding='same')(encoded)
    x = layers.UpSampling2D((4, 4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (5, 5), activation=approx_gelu, padding='same')(x)
    x = layers.UpSampling2D((3, 3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), activation=approx_gelu, padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    decoded = layers.Conv2D(4, (3, 3), activation='tanh', padding='same')(x)

    autoencoder = keras.Model(input_img, decoded)
    optimizer = keras.optimizers.Adam(learning_rate=learn_rate, amsgrad=True)
    autoencoder.compile(optimizer=optimizer, loss=loss_func)
    autoencoder.summary()

    return autoencoder

autoencoder = autoencoder_model(x_train[0].shape)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='loss', factor=LR_FACTOR, patience=1, min_lr=1e-6)
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=20, min_delta=1e-4, restore_best_weights=False)

from keras.callbacks import TensorBoard

autoencoder.fit(training_generator,
                steps_per_epoch = NUM_TRAINING // BATCH_SIZE,
                epochs = NUM_EPOCHS,
                validation_data = testing_generator,
                validation_steps = NUM_TESTING // BATCH_SIZE,
                callbacks=[TensorBoard(log_dir='/log/autoencoder'), reduce_lr, early_stop])

x_test_out = autoencoder.predict(x_test)
n = 5
for i in range(1, n + 1):
    ax = plt.subplot(3, n, i)
    ax.imshow(x_test_target[i + offset])
    ax2 = plt.subplot(3, n, i+n)
    ax2.imshow(x_test[i + offset])
    ax3 = plt.subplot(3, n, i+2*n)
    ax3.imshow(x_test_out[i + offset])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
plt.tight_layout()
plt.show()

autoencoder.save('denoising-model')
