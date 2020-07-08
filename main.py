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



# ----------------
# - Treina a CNN -
# ----------------
def evaluate_model(X_train, X_val, y_train, y_val, datagen):
    datagen.fit(X_train)

    googlenet_base = tf.keras.applications.InceptionV3(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
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

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=100,
                        verbose=1)

    _, val_acc = model.evaluate(X_val, y_val, verbose=1)

    return model, val_acc, history


# -----------------------
# - Predição de imagens -
# -----------------------

def PredizerImagem(model, X_train, X_val, y_train, y_val):

    # Evita problemas de overfitting (Parece que modifica ligeiramente a imagem)
    datagen = ImageDataGenerator(
        zoom_range=0.1,  # Aleatory zoom
        rotation_range=15,
        width_shift_range=0.1,  # horizontal shift
        height_shift_range=0.1,  # vertical shift
        horizontal_flip=True,
        vertical_flip=True)

    datagen.fit(X_train)

    pred = model.predict(X_val)
    pred = np.argmax(pred, axis=1)
    pred = pd.DataFrame(pred).replace(
        {0: 'chokkan', 1: 'fukunagashi', 2: 'kengai', 3: 'literatti', 4: 'moyogi', 5: 'shakan'})

    y_val_string = np.argmax(y_val, axis=1)
    y_val_string = pd.DataFrame(y_val_string).replace(
        {0: 'chokkan', 1: 'fukunagashi', 2: 'kengai', 3: 'literatti', 4: 'moyogi', 5: 'shakan'})

    mis_class = []
    for i in range(len(y_val_string)):
        if (not y_val_string[0][i] == pred[0][i]):
            mis_class.append(i)

        if (len(mis_class) == 8):
            break
        # if(len(mis_class)==8):
        #     break
        # else:
        #     mis_class.append(i)

    mis_class1 = []
    for i in range(len(y_val_string)):
        if (not y_val_string[0][i] == pred[0][i]):
            mis_class1.append(i)

    print(len(mis_class1))

    count = 0
    fig, ax = plt.subplots(3, 2)
    fig.set_size_inches(5, 5)
    for i in range(3):
        for j in range(2):
            ax[i, j].imshow(X_val[mis_class[count]][:, :, ::-1])
            ax[i, j].set_title("Predicted style : " + str(pred[0][mis_class[count]]) + "\n" + "Actual style : " + str(
                y_val_string[0][mis_class[count]]))
            plt.tight_layout()
            count += 1

    plt.show()

    return True

# ------------------
# - Abre o DataSet -
# ------------------

def AbreDataSet():
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
    l_fukunagashi = np.ones(len(fukunagashi_dir))
    l_kengai = 2 * np.ones(len(kengai_dir))
    l_literatti = 3 * np.ones(len(literatti_dir))
    l_moyogi = 4 * np.ones(len(moyogi_dir))
    l_shakan = 5 * np.ones(len(shakan_dir))

    y = np.concatenate((l_chokkan, l_fukunagashi, l_kengai, l_literatti, l_moyogi, l_shakan))

    # Finalização da categorização.
    y = to_categorical(y, 6)

    return X, y

# ---------------------
# - Reparte o DataSet -
# ---------------------

def ReparteDataSet(X,y,r_state):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, stratify=y, random_state=r_state)
    return X_train, X_val, y_train, y_val

# ------------------------
# - Verificar a precisao -
# ------------------------

def VerificarPrecisao(model, X_val, y_val):

    loss1, accuracy1 = model.evaluate(X_val, y_val, steps = 20)

    print("--------Precisão Atual---------")
    print("Initial loss: {:.2f}".format(loss1))
    print("Initial accuracy: {:.2f}".format(accuracy1))
    print("---------------------------")

# -------------------------------
# - Imprime exemplos de estilos -
# -------------------------------

def ImprimirExemplos(label, arrayImages):
    fig, ax = plt.subplots(3, 3)
    fig.set_size_inches(10, 10)
    for i in range(3):
        for j in range(3):
            r = random.randint(0, len(label))
            ax[i, j].imshow(arrayImages[r][:, :, ::-1])
            ax[i, j].set_title('Style: ' + label[r])

    plt.tight_layout()
    plt.show()

    return True

# inceptionv3 = keras.models.load_model('GoogleNet10EpochsK-Folds-1.tf')
# print("Modelo carregado!")
#
# X1,y1 = AbreDataSet()
# X_train1, X_val1, y_train1, y_val1 = ReparteDataSet(X1,y1,42)
#
# PredizerImagem(inceptionv3,X_train1, X_val1, y_train1, y_val1)
# VerificarPrecisao(inceptionv3,X_val1,y_val1)

datagen = ImageDataGenerator(
    zoom_range=0.1,  # Aleatory zoom
    rotation_range=15,
    width_shift_range=0.1,  # horizontal shift
    height_shift_range=0.1,  # vertical shift
    horizontal_flip=True,
    vertical_flip=True)

n_folds = 10
cv_scores = list()
X,y = AbreDataSet()
for index in range(n_folds):
    print('-----------------------------------------------------------------')
    print('Epoca ' + str(index))
    # split data
    r_state = np.random.randint(1, 1000, 1)[0]
    print(r_state)
    X_train, X_val, y_train, y_val = ReparteDataSet(X,y,r_state)
    # evaluate model
    model, test_acc, history = evaluate_model(X_train, X_val, y_train, y_val, datagen)
    print('>%.3f' % test_acc)
    cv_scores.append(test_acc)

    Variavel = "GoogleNet10EpochsK-Folds"
    Final = ".tf"
    VariavelFinal = Variavel + str(index - 1) + Final

    model.save(VariavelFinal)
    print("Modelo salvo com sucesso!")

    with open('trainHistory' + str(index - 1), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    print("Histórico de treino salvo com sucesso!")

    # OldHistory = pickle.load(open('trainHistory' + str(index-1), 'rb'))

    # plt.plot(OldHistory['accuracy'])
    # plt.plot(OldHistory['val_accuracy'])
    # plt.title('Model Accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epochs')
    # plt.legend(['train', 'test'])
    # plt.show()
    #
    # plt.plot(OldHistory['loss'])
    # plt.plot(OldHistory['val_loss'])
    # plt.title('Model Loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Epochs')
    # plt.legend(['train', 'test'])
    # plt.show()

    print('-----------------------------------------------------------------')

print('Estimated Accuracy %.3f (%.3f)' % (np.mean(cv_scores), np.std(cv_scores)))

