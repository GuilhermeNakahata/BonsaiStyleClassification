import pickle
import warnings

from keras.callbacks import ModelCheckpoint

warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Nadam, Adam
from keras.utils import to_categorical
from keras.layers import Dropout, Flatten, Input, Dense
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.models import model_from_json

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
        X.append(np.array(cv.resize(cv.imread(f), (224,224), interpolation = cv.INTER_AREA)))
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
l_kengai = 2*np.ones(len(kengai_dir))
l_kengai_string = ['kengai' for i in range(len(kengai_dir))]
l_literatti = 3*np.ones(len(literatti_dir))
l_literatti_string = ['literatti' for i in range(len(literatti_dir))]
l_moyogi = 4*np.ones(len(moyogi_dir))
l_moyogi_string = ['moyogi' for i in range(len(moyogi_dir))]
l_shakan = 5*np.ones(len(shakan_dir))
l_shakan_string = ['shakan' for i in range(len(shakan_dir))]

y_string = np.concatenate((l_chokkan_string, l_fukunagashi_string, l_kengai_string, l_literatti_string, l_moyogi_string, l_shakan_string))

y = np.concatenate((l_chokkan, l_fukunagashi, l_kengai, l_literatti, l_moyogi, l_shakan))

# Imprime exemplos de estilos
#
# fig,ax=plt.subplots(2,3)
# fig.set_size_inches(15,15)
# for i in range(2):
#     for j in range (3):
#         r = random.randint(0,len(y_string))
#         ax[i,j].imshow(X[r][:,:,::-1])
#         ax[i,j].set_title('Flower: ' + y_string[r])
#
# plt.tight_layout()
# plt.show()

# Finalização da categorização.
y = to_categorical(y, 6)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.3, random_state=42)

X = []

# Evita problemas de overfitting (Parece que modifica ligeiramente a imagem)
datagen = ImageDataGenerator(
    zoom_range = 0.1, # Aleatory zoom
    rotation_range= 15,
    width_shift_range=0.1,  # horizontal shift
    height_shift_range=0.1,  # vertical shift
    horizontal_flip=True,
    vertical_flip=True)

datagen.fit(X_train)
#
# vgg16_base = tf.keras.applications.VGG16(input_shape=(224,224,3), include_top=False, weights='imagenet')
# googlenet_base = tf.keras.applications.InceptionV3(input_shape=(224,224,3), include_top=False, weights='imagenet')
resnet_base = tf.keras.applications.ResNet101V2(input_shape=(224,224,3), include_top=False, weights='imagenet')

base_learning_rate = 0.0001

class Wrapper(tf.keras.Model):
    def __init__(self, base_model):
        super(Wrapper, self).__init__()

        self.base_model = base_model
        self.average_pooling_layer = tf.keras.layers.GlobalAveragePooling2D()
        self.output_layer = tf.keras.layers.Dense(6,activation='sigmoid')

    def call(self, inputs):
        x = self.base_model(inputs)
        x = self.average_pooling_layer(x)
        output = self.output_layer(x)
        return output

#
x = resnet_base.output
x = Flatten()(x)
x = Dense(512,activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256,activation='relu')(x)
x = Dropout(0.2)(x)
out = Dense(6,activation='softmax')(x)

tf_model=Model(inputs=resnet_base.input,outputs=out)

tf_model.summary()

for layer in tf_model.layers[:20]:
    layer.trainable=False

tf_model.compile(optimizer = Nadam(0.0001) , loss = 'categorical_crossentropy', metrics=["accuracy"])

# # vgg16_base.trainable = False
# # vgg16 = Wrapper(vgg16_base)
# # vgg16.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
# #               loss='binary_crossentropy',
# #               metrics=['accuracy'])
#
# googlenet_base.trainable = False
# googlenet = Wrapper(googlenet_base)
# googlenet.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
#                   loss='binary_crossentropy',
#                   metrics=['accuracy'])
#
# # resnet_base.trainable = False
# # resnet = Wrapper(resnet_base)
# # resnet.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
# #                loss='binary_crossentropy',
# #                metrics=['accuracy'])
#
# # loss1, accuracy1 = vgg16.evaluate(X_val, y_val, steps = 20)
# loss2, accuracy2 = googlenet.evaluate(X_val, y_val, steps = 20)
# # loss3, accuracy3 = resnet.evaluate(X_val, y_val, steps = 20)
# #
# # print("--------VGG16---------")
# # print("Initial loss: {:.2f}".format(loss1))
# # print("Initial accuracy: {:.2f}".format(accuracy1))
# # print("---------------------------")
# #
# print("--------GoogLeNet---------")
# print("Initial loss: {:.2f}".format(loss2))
# print("Initial accuracy: {:.2f}".format(accuracy2))
# print("---------------------------")
# #
# # print("--------ResNet---------")
# # print("Initial loss: {:.2f}".format(loss3))
# # print("Initial accuracy: {:.2f}".format(accuracy3))
# # print("---------------------------")
#
#

new_model = keras.models.load_model('VGG16.tf')
print("Modelo carregado!")


# history = new_model.fit(X_train, y_train,
#                     epochs=5,
#                     validation_data=(X_val, y_val))
#
# history = tf_model.fit(X_train, y_train, batch_size = 1, epochs = 10, initial_epoch = 0,
#                       validation_data = (X_val, y_val))

# history = new_model.fit(X_train, y_train,
#                     epochs=50,
#                     validation_data = (X_val,y_val))

# OldHistory = pickle.load(open('trainHistory' , 'rb'))
#
# tf_model.save('resnet10.tf')
# print("Modelo salvo com sucesso!")
#
# with open('trainHistory', 'wb') as file_pi:
#     pickle.dump(history.history, file_pi)
# print("Histórico de treino salvo com sucesso!")
#
# NewHistory = pickle.load(open('trainHistory' , 'rb'))
#
# OldHistory['accuracy'].extend(NewHistory['accuracy'])
# OldHistory['val_accuracy'].extend(NewHistory['val_accuracy'])
# #
# with open('trainHistory', 'wb') as file_pi:
#     pickle.dump(OldHistory, file_pi)
# #
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


#------------------------------------------------
#- Verificação de precisão e inicio treinamento -
#------------------------------------------------

loss1, accuracy1 = new_model.evaluate(X_val, y_val, steps = 20)

print("--------Precisão Atual---------")
print("Initial loss: {:.2f}".format(loss1))
print("Initial accuracy: {:.2f}".format(accuracy1))
print("---------------------------")
#
# history = new_model.fit(X_train, y_train,
#                     epochs=50,
#                     validation_data = (X_val,y_val))
#
# new_model.save('Renet100Epochs.tf')
# print("Modelo salvo com sucesso")
#
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Model Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epochs')
# plt.legend(['train', 'test'])
# plt.show()


#-----------------------
#- Predição de imagens -
#-----------------------

# Pegando os diretórios da base de dados.
# chokkan_dir = glob.glob(os.path.join('Chokkan/', '*'))
# fukunagashi_dir = glob.glob(os.path.join('Fukunagashi/', '*'))
# kengai_dir = glob.glob(os.path.join('Kengai/', '*'))
# literatti_dir = glob.glob(os.path.join('Literatti/', '*'))
# moyogi_dir = glob.glob(os.path.join('Moyogi/', '*'))
# shakan_dir = glob.glob(os.path.join('Shakan/', '*'))
#
# # Compilando todos os caminhos.
# X_path = chokkan_dir + fukunagashi_dir + kengai_dir + literatti_dir + moyogi_dir + shakan_dir
#
# X = []
#
# # Tamanho da imagem escolhido foi de 224x224, a maioria das redes neurais prontas utilizam o 224x224
# for f in X_path:
#     try:
#         X.append(np.array(cv.resize(cv.imread(f), (224,224), interpolation = cv.INTER_AREA)))
#     except:
#         print(f)
#
# X = np.array(X)
#
# # Normalização dividido pela quantidade de pixel no RGB.
# X = X / 255
#
# # One-Hot-Encondig.
# l_chokkan = np.zeros(len(chokkan_dir))
# l_chokkan_string = ['chokkan' for i in range(len(chokkan_dir))]
# l_fukunagashi = np.ones(len(fukunagashi_dir))
# l_fukunagashi_string = ['fukunagashi' for i in range(len(fukunagashi_dir))]
# l_kengai = 2*np.ones(len(kengai_dir))
# l_kengai_string = ['kengai' for i in range(len(kengai_dir))]
# l_literatti = 3*np.ones(len(literatti_dir))
# l_literatti_string = ['literatti' for i in range(len(literatti_dir))]
# l_moyogi = 4*np.ones(len(moyogi_dir))
# l_moyogi_string = ['moyogi' for i in range(len(moyogi_dir))]
# l_shakan = 5*np.ones(len(shakan_dir))
# l_shakan_string = ['shakan' for i in range(len(shakan_dir))]
#
# y_string = np.concatenate((l_chokkan_string, l_fukunagashi_string, l_kengai_string, l_literatti_string, l_moyogi_string, l_shakan_string))
#
# y = np.concatenate((l_chokkan, l_fukunagashi, l_kengai, l_literatti, l_moyogi, l_shakan))
#
# # Imprime exemplos de estilos
# #
# # fig,ax=plt.subplots(2,3)
# # fig.set_size_inches(15,15)
# # for i in range(2):
# #     for j in range (3):
# #         r = random.randint(0,len(y_string))
# #         ax[i,j].imshow(X[r][:,:,::-1])
# #         ax[i,j].set_title('Flower: ' + y_string[r])
# #
# # plt.tight_layout()
# # plt.show()
#
# # Finalização da categorização.
# y = to_categorical(y, 6)
#
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.3, random_state=42)
#
# X = []
#
# # Evita problemas de overfitting (Parece que modifica ligeiramente a imagem)
# datagen = ImageDataGenerator(
#     zoom_range = 0.1, # Aleatory zoom
#     rotation_range= 15,
#     width_shift_range=0.1,  # horizontal shift
#     height_shift_range=0.1,  # vertical shift
#     horizontal_flip=True,
#     vertical_flip=True)
#
# datagen.fit(X_train)
#
# pred = new_model.predict(X_val)
# pred = np.argmax(pred, axis = 1)
# pred = pd.DataFrame(pred).replace({0:'chokkan',1:'fukunagashi',2:'kengai',3:'literatti',4:'moyogi',5:'shakan'})
#
# y_val_string = np.argmax(y_val, axis = 1)
# y_val_string = pd.DataFrame(y_val_string).replace({0:'chokkan',1:'fukunagashi',2:'kengai',3:'literatti',4:'moyogi',5:'shakan'})
#
# mis_class = []
# for i in range(len(y_val_string)):
#     if(y_val_string[0][i] == pred[0][i]):
#         mis_class.append(i)
#
#     if(len(mis_class)==8):
#         break
#     # if(len(mis_class)==8):
#     #     break
#     # else:
#     #     mis_class.append(i)
#
# mis_class1 = []
# for i in range(len(y_val_string)):
#     if(y_val_string[0][i] == pred[0][i]):
#         mis_class1.append(i)
#
#
# print(len(mis_class1))
#
# count = 0
# fig,ax = plt.subplots(3,2)
# fig.set_size_inches(5,5)
# for i in range (3):
#     for j in range (2):
#         ax[i,j].imshow(X_val[mis_class[count]][:,:,::-1])
#         ax[i,j].set_title("Predicted style : "+str(pred[0][mis_class[count]])+"\n"+"Actual style : " + str(y_val_string[0][mis_class[count]]))
#         plt.tight_layout()
#         count+=1
#
# plt.show()

#---------------------------------
#- Implementação da CNN - Manual -
#---------------------------------

# inp = Input((224,224,3))
# conv1 = Conv2D(64, (5,5), padding='valid', activation= 'relu')(inp)
# conv1 = MaxPooling2D(pool_size=(2,2))(conv1)
# conv1 = BatchNormalization()(conv1)
# conv2 = Conv2D(96, (4,4), padding='valid', activation= 'relu')(conv1)
# conv2 = MaxPooling2D(pool_size=(2,2))(conv2)
# conv2 = BatchNormalization()(conv2)
# conv3 = Conv2D(128, (3,3), padding='valid', activation= 'relu')(conv2)
# conv3 = MaxPooling2D(pool_size=(2,2))(conv3)
# conv3 = BatchNormalization()(conv3)
# conv4 = Conv2D(256, (3,3), padding='valid', activation= 'relu')(conv3)
# conv4 = MaxPooling2D(pool_size=(2,2))(conv4)
# conv4 = BatchNormalization()(conv4)
#
# flat = Flatten()(conv4)
# dense1 = Dense(512, activation= 'relu')(flat)
# dense1 = Dropout(0.5)(dense1)
# dense2 = Dense(64, activation= 'relu')(dense1)
# dense2 = Dropout(0.1)(dense2)
# out = Dense(7, activation = 'softmax')(dense2)
# model = Model(inp, out)
#
# model.compile(optimizer = Nadam(lr = 0.0001) , loss = 'categorical_crossentropy', metrics=['accuracy'])
#
# history = model.fit(X_train, y_train, batch_size = 32, epochs = 3, initial_epoch = 0, validation_data = (X_val, y_val))
#
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('Model Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epochs')
# plt.legend(['train', 'test'])
# plt.show()


#----------------------------------
#- Implementação TransferLearning -
#----------------------------------

# vgg = keras.applications.VGG16(input_shape=(224,224,3), include_top = False, weights= 'imagenet')
#
# x = vgg.output
# x = Flatten()(x)
# x = Dense(3078,activation='relu')(x)
# x = Dropout(0.5)(x)
# x = Dense(256,activation='relu')(x)
# x = Dropout(0.2)(x)
# out = Dense(7,activation='softmax')(x)
#
# tf_model=Model(inputs=vgg.input,outputs=out)
#
# for i,layer in enumerate(tf_model.layers):
#     print(i,layer.name)
#
# for layer in tf_model.layers[:20]:
#     layer.trainable=False
#
# tf_model.summary()
#
# tf_model.load_weights('TF-CNN.29-0.08-0.98-1.03-0.85.hdf5')
# filepath = 'TF-CNN.{epoch:02d}-{loss:.2f}-{acc:.2f}-{val_loss:.2f}-{val_acc:.2f}.hdf5'
# lr_red = keras.callbacks.ReduceLROnPlateau(monitor='acc', patience=3, verbose=1, factor=0.5, min_lr=0.000001)
# chkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
# tf_model.compile(optimizer = Nadam(0.0001) , loss = 'categorical_crossentropy', metrics=["accuracy"])
#
# history = tf_model.fit(X_train, y_train, batch_size = 1, epochs = 30, initial_epoch = 0,
#                        validation_data = (X_val, y_val), callbacks=[lr_red, chkpoint])

