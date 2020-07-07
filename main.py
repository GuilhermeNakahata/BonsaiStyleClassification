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

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

X = []

# Evita problemas de overfitting (Aumenta o data_set)
datagen = ImageDataGenerator(
    zoom_range=0.1,  # Aleatory zoom
    rotation_range=15,
    width_shift_range=0.1,  # horizontal shift
    height_shift_range=0.1,  # vertical shift
    horizontal_flip=True,
    vertical_flip=True)

datagen.fit(X_train)

listStyles = []

for i in range(len(y_string)):
    achado = False
    for j in range(len(listStyles)):
        if y_string[i] == listStyles[j]:
            achado = True
            break
    if achado == False:
        listStyles.append(y_string[i])

listQuantity = []

for i in range(len(listStyles)):
    listQuantity.append(0)
    for j in range(len(y_string)):
        if y_string[j] == listStyles[i]:
            listQuantity[i] += 1

bonsaiDataSet = {'Style': listStyles,
                 'Quantity': listQuantity
                 }

df = pd.DataFrame(bonsaiDataSet, columns=['Style', 'Quantity'])

print(df)
#
# sns.barplot(x="Style", y="Quantity", data=df)
# plt.show()

# plt.figure(figsize = (12,6))
# sns.barplot(df.Style, df.Quantity, alpha = 0.5)
# plt.xticks(rotation = 'vertical')
# plt.xlabel('Styles', fontsize =12)
# plt.ylabel('Quantitys', fontsize = 12)
# plt.show()

print("the number of training examples = %i" % X_train.shape[0])
print("the number of classes = %i" % len(numpy.unique(y_string)))
print("Dimention of images = {:d} x {:d}  ".format(X_train[1].shape[0], X_train[1].shape[1]))

unique, count = numpy.unique(y_string, return_counts=True)
print("The number of occuranc of each class in the dataset = %s " % dict(zip(unique, count)), "\n")
#
# images_and_labels = list(zip(X_train,  y_train))
# for index, (image, label) in enumerate(images_and_labels[:12]):
#     plt.subplot(5, 4, index + 1)
#     plt.axis('off')
#     plt.imshow(image.squeeze(), cmap=plt.cm.gray_r, interpolation='nearest')
#     plt.title('label: %i' % label)
#
# plt.show()

#
#
# googlenet_base = tf.keras.applications.VGG16(input_shape=(224,224,3), include_top=False, weights='imagenet')
googlenet_base = tf.keras.applications.InceptionV3(input_shape=(224,224,3), include_top=False, weights='imagenet')
# resnet_base = tf.keras.applications.ResNet101V2(input_shape=(224,224,3), include_top=False, weights='imagenet')
# resnet_base = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(224,224,3))
# resnet_base = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

base_learning_rate = 0.0001
#
# class Wrapper(tf.keras.Model):
#     def __init__(self, base_model):
#         super(Wrapper, self).__init__()
#
#         self.base_model = base_model
#         self.average_pooling_layer = tf.keras.layers.GlobalAveragePooling2D()
#         self.output_layer = tf.keras.layers.Dense(6,activation='sigmoid')
#
#     def call(self, inputs):
#         x = self.base_model(inputs)
#         x = self.average_pooling_layer(x)
#         output = self.output_layer(x)
#         return output
#
#
# vgg16_base.summary()
#

acc_per_fold = []
loss_per_fold = []

# train_generator = datagen.flow(X_train, y_train, batch_size=32)
# val_generator = datagen.flow(X_val, y_val, batch_size=32)

train_generator = np.concatenate((X_train, X_val))
val_generator = np.concatenate((y_val, y_train))

# K-fold Cross Validation model evaluation
kfold = KFold(n_splits=10, shuffle=True)
fold_no = 1
for train, test in kfold.split(train_generator, val_generator):

    # Define the model architecture
    x = googlenet_base.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    # x = Dense(1024, activation='relu')(x)
    x = Dropout(0.4)(x)
    predictions = Dense(6, activation='softmax')(x)
    model = Model(inputs=googlenet_base.input, outputs=predictions)

    # plot_model(model, to_file='model.png')
    # from IPython.display import SVG
    # from keras.utils.vis_utils import model_to_dot
    #
    # SVG(model_to_dot(model).create(prog='dot', format='svg'))
    #
    # plt.show()

    #Compile the model
    for layer in googlenet_base.layers:
        layer.trainable = False

    model.compile(optimizer=optimizers.SGD(lr=1e-4,momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    # Fit data to model
    history = model.fit(train_generator[train],val_generator[train],
                        validation_data=(train_generator[test],val_generator[test]),
                        epochs=20,
                        verbose=1)

    # Generate generalization metrics
    scores = model.evaluate(train_generator[test], val_generator[test], verbose=0)
    print(
        f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    # Increase fold number
    fold_no = fold_no + 1

    Variavel = "GoogleNet10EpochsK-Folds"
    Final = ".tf"
    VariavelFinal = Variavel + str(fold_no-1) + Final

    model.save(VariavelFinal)
    print("Modelo salvo com sucesso!")

    with open('trainHistory' + str(fold_no-1), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    print("Histórico de treino salvo com sucesso!")

    OldHistory = pickle.load(open('trainHistory' + str(fold_no-1), 'rb'))

    plt.plot(OldHistory['accuracy'])
    plt.plot(OldHistory['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['train', 'test'])
    plt.show()

    plt.plot(OldHistory['loss'])
    plt.plot(OldHistory['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['train', 'test'])
    plt.show()

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
    print('------------------------------------------------------------------------')
    print(f'> Fold {i + 1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')

# model = models.Sequential()
# model.add(googlenet_base)
# model.add(layers.Flatten())
# model.add(layers.Dense(2048, activation='relu'))
# model.add(Dropout(0.3))
# model.add(layers.Dense(6, activation='softmax'))
#
# model.summary()
#
# print('Number of trainable weights before freezing the conv base:', len(model.trainable_weights))
# googlenet_base.trainable = False
# print('Number of trainable weights after freezing the conv base:', len(model.trainable_weights))
# #
# val_datagen = ImageDataGenerator(rescale=1. / 255)
# #
# batch_size = 32
# #
# train_generator = datagen.flow(X_train, y_train, batch_size=batch_size)
# val_generator = datagen.flow(X_val, y_val, batch_size=batch_size)
#
# model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-4,
#                                                                       momentum=0.9), metrics=['accuracy'])

# history = model.fit(X_train, y_train,
#                     epochs=20,
#                     validation_data=(X_val, y_val))

# history = model.fit_generator(train_generator,
#                               steps_per_epoch=len(X_train) // batch_size,
#                               epochs=100,
#                               validation_data=val_generator,
#                               validation_steps=len(X_val) // batch_size)
#
# x = vgg16_base.output
# x = Flatten()(x)
# x = Dense(512,activation='relu')(x)
# x = Dropout(0.5)(x)
# x = Dense(256,activation='relu')(x)
# x = Dropout(0.3)(x)
# out = Dense(6,activation='softmax')(x)
# #
# #
# tf_model=Model(inputs=vgg16_base.input,outputs=out)
#
# # tf_model.summary()
#
# for layer in tf_model.layers[:20]:
#     layer.trainable=False
#
# tf_model.compile(optimizer = Nadam(0.0001) , loss = 'categorical_crossentropy', metrics=["accuracy"])
#
# # history = tf_model.fit(X_train,y_train,validation_data=(X_val, y_val),epochs=10)
#
# history = tf_model.fit(X_train, y_train, batch_size = 1, epochs = 20, initial_epoch = 0,
#                       validation_data = (X_val, y_val))

# output = resnet_base.layers[-1].output
# output = keras.layers.Flatten()(output)
# restnet = Model(inputs=resnet_base.input, outputs=output)

# for layer in restnet.layers:
#     layer.trainable = False
# #
# # restnet.summary()
#
# model = Sequential()
# model.add(restnet)
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(6, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer=Nadam(0.0001), metrics=['accuracy'])
# model.summary()
#
# tf_model.compile(loss='categorical_crossentropy',
#               optimizer=Nadam(0.0001),
#               metrics=['accuracy'])
#
# history = tf_model.fit(X_train,y_train,validation_data=(X_val, y_val),epochs=5)
#
# history = model.fit(X_train, y_train,
#                     epochs=20,
#                     validation_data=(X_val, y_val))

# tf_model.compile(optimizer = Nadam(0.0001) , loss = 'categorical_crossentropy', metrics=["accuracy"])

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
# new_model = keras.models.load_model('VGG16.tf')
# print("Modelo carregado!")
#
# new_model.summary()
#
# for layer in new_model.layers[:20]:
#     layer.trainable=False
#
# new_model.summary()

#
# loss1, accuracy1 = new_model.evaluate(X_val, y_val, steps = 20)
# loss2, accuracy2 = googlenet.evaluate(X_val, y_val, steps = 20)
# # loss3, accuracy3 = resnet.evaluate(X_val, y_val, steps = 20)
#
# print("--------VGG16---------")
# print("Initial loss: {:.2f}".format(loss1))
# print("Initial accuracy: {:.2f}".format(accuracy1))
# print("---------------------------")
#
# ------------------------------
# - Visualização Features Maps -
# ------------------------------
#
# for layer in new_model.layers:
#     # check for convolutional layer
#     if 'conv' not in layer.name:
#         continue
#     # get filter weights
#     filters, biases = layer.get_weights()
#     print(layer.name, filters.shape)
#
# filters, biases = new_model.layers[1].get_weights()
#
# f_min, f_max = filters.min(), filters.max()
# filters = (filters - f_min) / (f_max - f_min)
#
# # plot first few filters
# n_filters, ix = 6, 1
# for i in range(n_filters):
#     # get the filter
#     f = filters[:, :, :, i]
#     # plot each channel separately
#     for j in range(3):
#         # specify subplot and turn of axis
#         ax = pyplot.subplot(n_filters, 3, ix)
#         ax.set_xticks([])
#         ax.set_yticks([])
#         # plot filter channel in grayscale
#         pyplot.imshow(f[:, :, j])
#         ix += 1
# # show the figure
# pyplot.show()
#
# for i in range(len(new_model.layers)):
#     layer = new_model.layers[i]
#     # check for convolutional layer
#     if 'conv' not in layer.name:
#         continue
#     # summarize output shape
#     print(i, layer.name, layer.output.shape)
#
# from keras.applications.vgg16 import VGG16
# from keras.applications.vgg16 import preprocess_input
# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
# from keras.models import Model
# from matplotlib import pyplot
# from numpy import expand_dims
#
# ixs = [2, 5, 9, 13, 17]
# outputs = [new_model.layers[i].output for i in ixs]
# model = Model(inputs=new_model.inputs, outputs=outputs)
# # load the image with the required shape
# img = load_img('exampleImg.jpg', target_size=(224, 224))
# # convert the image to an array
# img = img_to_array(img)
# # expand dimensions so that it represents a single 'sample'
# img = expand_dims(img, axis=0)
# # prepare the image (e.g. scale pixel values for the vgg)
# img = preprocess_input(img)
# # get feature map for first hidden layer
# feature_maps = model.predict(img)
# # plot the output from each block
# square = 5
# for fmap in feature_maps:
#     # plot all 64 maps in an 8x8 squares
#     ix = 1
#     for _ in range(square):
#         for _ in range(square):
#             # specify subplot and turn of axis
#             ax = pyplot.subplot(square, square, ix)
#             ax.set_xticks([])
#             ax.set_yticks([])
#             # plot filter channel in grayscale
#             pyplot.imshow(fmap[0, :, :, ix-1])
#             ix += 1
#     # show the figure
#     pyplot.show()
#
#
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
#
# new_model = keras.models.load_model('resnet50.tf')
# print("Modelo carregado!")


# history = model.fit(X_train, y_train,
#                     epochs=50,
#                     validation_data=(X_val, y_val))
#
# history = tf_model.fit(X_train, y_train, batch_size = 1, epochs = 10, initial_epoch = 0,
#                       validation_data = (X_val, y_val))

#
# history = new_model.fit(X_train, y_train,
#                     epochs=20,
#                     validation_data = (X_val,y_val))
# #
# OldHistory = pickle.load(open('trainHistory' , 'rb'))
#
# model.save('GoogleNet50Epochs.tf')
# print("Modelo salvo com sucesso!")
#
# with open('trainHistory', 'wb') as file_pi:
#     pickle.dump(history.history, file_pi)
# print("Histórico de treino salvo com sucesso!")
#
# NewHistory = pickle.load(open('trainHistory' , 'rb'))
# #
# OldHistory['accuracy'].extend(NewHistory['accuracy'])
# OldHistory['val_accuracy'].extend(NewHistory['val_accuracy'])
#
# with open('trainHistory', 'wb') as file_pi:
#     pickle.dump(OldHistory, file_pi)
#
#
# OldHistory = pickle.load(open('trainHistory', 'rb'))
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

# ------------------------------------------------
# - Verificação de precisão e inicio treinamento -
# ------------------------------------------------
#
# loss1, accuracy1 = new_model.evaluate(X_val, y_val, steps = 20)
#
# print("--------Precisão Atual---------")
# print("Initial loss: {:.2f}".format(loss1))
# print("Initial accuracy: {:.2f}".format(accuracy1))
# print("---------------------------")
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


# -----------------------
# - Predição de imagens -
# -----------------------

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
#     if(not y_val_string[0][i] == pred[0][i]):
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

# ---------------------------------
# - Implementação da CNN - Manual -
# ---------------------------------

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


# ----------------------------------
# - Implementação TransferLearning -
# ----------------------------------

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
