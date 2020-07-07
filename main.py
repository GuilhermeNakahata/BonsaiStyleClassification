import pickle
import warnings

import keras as keras
import numpy
import numpy as numpy
from keras.callbacks import ModelCheckpoint
from keras.datasets import cifar10
from keras.legacy import layers
from matplotlib import pyplot
from nets.nn import Sequential
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from tensorflow.python.keras.applications.resnet import ResNet50

warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split, KFold

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Nadam, Adam, Adadelta
from keras.utils import to_categorical, plot_model
from keras.layers import Dropout, Flatten, Input, Dense
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras.models import model_from_json
from keras.models import Sequential
from keras import optimizers, models, initializers

from keras import layers
from keras import models

import random
import tensorflow as tf
import cv2 as cv
import os
import glob

# Pegando os diretórios da base de dados.
chokkan_dir = glob.glob(os.path.join('Chokkan/', '*'))
fukunagashi_dir = glob.glob(os.path.join('Fukunagashi/', '*'))
kengai_dir = glob.glob(os.path.join('Kengai/', '*'))
literatti_dir = glob.glob(os.path.join('Literatti/', '*'))
moyogi_dir = glob.glob(os.path.join('Moyogi/', '*'))
shakan_dir = glob.glob(os.path.join('Shakan/', '*'))

# Compilando todos os caminhos.
X_path = chokkan_dir + fukunagashi_dir + kengai_dir + literatti_dir + moyogi_dir + shakan_dir

X = []

# Tamanho da imagem escolhido foi de 224x224, a maioria das redes neurais prontas utilizam o 224x224
for f in X_path:
    try:
        X.append(np.array(cv.resize(cv.imread(f), (224, 224), interpolation=cv.INTER_AREA)))
    except:
        print(f)

X = np.array(X)

# Normalização dividido pela quantidade de pixel no RGB.
X = X / 255

# One-Hot-Encondig.
l_chokkan = np.zeros(len(chokkan_dir))
l_chokkan_string = ['chokkan' for i in range(len(chokkan_dir))]
l_fukunagashi = np.ones(len(fukunagashi_dir))
l_fukunagashi_string = ['fukunagashi' for i in range(len(fukunagashi_dir))]
l_kengai = 2 * np.ones(len(kengai_dir))
l_kengai_string = ['kengai' for i in range(len(kengai_dir))]
l_literatti = 3 * np.ones(len(literatti_dir))
l_literatti_string = ['literatti' for i in range(len(literatti_dir))]
l_moyogi = 4 * np.ones(len(moyogi_dir))
l_moyogi_string = ['moyogi' for i in range(len(moyogi_dir))]
l_shakan = 5 * np.ones(len(shakan_dir))
l_shakan_string = ['shakan' for i in range(len(shakan_dir))]

y_string = np.concatenate(
    (l_chokkan_string, l_fukunagashi_string, l_kengai_string, l_literatti_string, l_moyogi_string, l_shakan_string))

y = np.concatenate((l_chokkan, l_fukunagashi, l_kengai, l_literatti, l_moyogi, l_shakan))

# Finalização da categorização.
y = to_categorical(y, 6)


# Evita problemas de overfitting (Aumenta o data_set)
datagen = ImageDataGenerator(
    zoom_range=0.1,  # Aleatory zoom
    rotation_range=15,
    width_shift_range=0.1,  # horizontal shift
    height_shift_range=0.1,  # vertical shift
    horizontal_flip=True,
    vertical_flip=True)


def evaluate_model(X_train, X_val, y_train, y_val):

    datagen.fit(X_train)

    googlenet_base = tf.keras.applications.InceptionV3(input_shape=(224,224,3), include_top=False, weights='imagenet')
    x = googlenet_base.output
    x = GlobalAveragePooling2D(name='avg_pool_2D')(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    predictions = Dense(6, activation='softmax', name='classifcation_softmax_6')(x)
    model = Model(inputs=googlenet_base.input, outputs=predictions)

    for layer in googlenet_base.layers:
            layer.trainable = False

    model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.RMSprop(lr=2e-5),
                      metrics=['accuracy'])

    history = model.fit(X_train,y_train,
                            validation_data=(X_val, y_val),
                            epochs=2,
                            verbose=1)

    _, val_acc = model.evaluate(X_val, y_val, verbose = 1)

    return model, val_acc

n_folds = 10
cv_scores, model_history = list(), list()
for index in range(n_folds):
    print('-----------------------------------------------------------------')
    print('Epoca ' + index)
    # split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, random_state = np.random.randint(1,1000, 1)[0])
    # evaluate model
    model, test_acc = evaluate_model(X_train, X_val, y_train, y_val)
    print('>%.3f' % test_acc)
    cv_scores.append(test_acc)
    model_history.append(model)
    print('-----------------------------------------------------------------')

print('Estimated Accuracy %.3f (%.3f)' % (np.mean(cv_scores), np.std(cv_scores)))


