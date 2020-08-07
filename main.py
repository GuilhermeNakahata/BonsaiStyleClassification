import pickle
import warnings

from nets.nn import Sequential
from keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.applications.densenet import DenseNet121
from tensorflow.python.keras.applications.xception import Xception
from tensorflow.keras import layers
from tensorflow.python.keras.callbacks import ModelCheckpoint

warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Nadam
from keras.utils import to_categorical
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras import optimizers

import random
import tensorflow as tf
import cv2 as cv
import os
import glob
import sklearn.metrics as metrics

# ----------------
# - Cria a VGG16 -
# ----------------

def evaluate_modelVGG16(trainGenerator, valGenerator, indexEpochs):

    print("Criando VGG16")

    VGG16 = keras.applications.VGG16(include_top = False, weights= 'imagenet', input_shape=(224,224,3))

    x = VGG16.output
    x = Flatten()(x)
    x = Dense(256,activation='relu')(x)
    x = Dropout(0.1)(x)
    out = Dense(7,activation='softmax')(x)
    tf_model=Model(inputs=VGG16.input,outputs=out)

    for layer in tf_model.layers[:20]:
        layer.trainable=False

    tf_model.compile(optimizer = Nadam(0.0001) , loss = 'categorical_crossentropy', metrics=["accuracy"])

    model, val_acc, history = TreinarModelo(trainGenerator,valGenerator, tf_model, indexEpochs)

    return model, val_acc, history

# --------------------
# - Continua a VGG16 -
# --------------------

def ContinuaVGG16(trainGenerator, valGenerator, indexEpochs):

    print("Continua VGG16")

    VGG16Treinada = keras.models.load_model('GoogleNet10EpochsK-Folds1.tf')
    print("Modelo carregado!")

    for layer in VGG16Treinada.layers:
        layer.trainable = False

    for layer in VGG16Treinada.layers:
        if layer.name in ['dense', 'dropout', 'dense_1', 'dropout_1', 'dense_2']:
            layer.trainable = True

    VGG16Treinada.compile(optimizer = Nadam(0.0001), loss = 'categorical_crossentropy', metrics=["accuracy"])

    model, val_acc, history = TreinarModelo(trainGenerator,valGenerator, VGG16Treinada, indexEpochs)

    return model, val_acc, history

# -----------------
# - Cria a ResNet -
# -----------------

def evaluate_modelResNet(trainGenerator, valGenerator, indexEpochs):

    print("Criando ResNet")

    restnet = ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3))

    output = restnet.layers[-1].output
    output = keras.layers.Flatten()(output)

    restnet = Model(restnet.input, output)

    for layer in restnet.layers:
        layer.trainable = False


    # model = Sequential()
    # model.add(restnet)
    # model.add(Dense(128, activation='relu' , input_dim=(224,244,3)))
    # model.add(Dropout(0.4))
    # # model.add(Dense(256, activation='relu'))
    # # model.add(Dropout(0.3))
    # model.add(Dense(6, activation='softmax'))
    # model.compile(loss='categorical_crossentropy',
    #               optimizer=optimizers.RMSprop(lr=2e-5),
    #               metrics=['accuracy'])

    restnet.trainable = True
    set_trainable = False
    for layer in restnet.layers:
        if layer.name in ['conv5_block3_2_conv', 'conv5_block3_2_bn', 'conv5_block3_2_relu', 'conv5_block3_3_conv', 'conv5_block3_3_bn', 'conv5_block3_add', 'conv5_block3_out', 'flatten']:
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    layers = [(layer, layer.name, layer.trainable) for layer in restnet.layers]
    pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])

    model_finetuned = Sequential()
    model_finetuned.add(restnet)
    model_finetuned.add(Dense(256, activation='relu', input_dim=(224,244,3)))
    model_finetuned.add(Dropout(0.4))
    model_finetuned.add(Dense(256, activation='relu'))
    model_finetuned.add(Dropout(0.1))
    model_finetuned.add(Dense(7, activation='softmax'))
    model_finetuned.compile(loss='categorical_crossentropy',optimizer=optimizers.RMSprop(lr=1e-5),metrics=['accuracy'])

    model_finetuned.summary()

    model, val_acc, history = TreinarModelo(trainGenerator,valGenerator, model_finetuned, indexEpochs)

    return model, val_acc, history

# ---------------------
# - Continua a ResNet -
# ---------------------

def ContinuaResnet(trainGenerator, valGenerator, indexEpochs):

    print("Continua ResNet")

    resnetRetreino = keras.models.load_model('GoogleNet10EpochsK-Folds1.tf')
    print("Modelo carregado!")

    resnetRetreino.summary()

    for layer in resnetRetreino.layers:
        layer.trainable = False

    for layer in resnetRetreino.layers:
        if layer.name in ['dense', 'dropout', 'dense_1', 'dropout_1', 'dense_2']:
            layer.trainable = True

    resnetRetreino.summary()

    resnetRetreino.compile(loss='categorical_crossentropy',optimizer=optimizers.RMSprop(lr=1e-5),metrics=['accuracy'])


    model, val_acc, history = TreinarModelo(trainGenerator,valGenerator, resnetRetreino, indexEpochs)

    return model, val_acc, history

# ---------------------
# - Cria a ResNetNova -
# ---------------------

def evaluate_modelResNetNova(trainGenerator, valGenerator, indexEpochs):

    print("Criando ResNet")

    restnet = ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3))

    output = restnet.layers[-1].output

    restnet = Model(restnet.input, output)

    for layer in restnet.layers:
        layer.trainable = False

    x = restnet.output
    x = Flatten()(x)
    x = Dense(3078,activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256,activation='relu')(x)
    x = Dropout(0.1)(x)
    out = Dense(7,activation='softmax')(x)
    tf_model=Model(inputs=restnet.input,outputs=out)

    tf_model.compile(loss='categorical_crossentropy',
                  optimizer=Nadam(0.0001),
                  metrics=['accuracy'])

    model, val_acc, history = TreinarModelo(trainGenerator,valGenerator, tf_model, indexEpochs)

    return model, val_acc, history

# -------------------
# - Cria a Xception -
# -------------------

def evaluate_modelXception(trainGenerator, valGenerator, indexEpochs):

    print("Criando Xception")

    xCeption = Xception(input_shape=(224,224,3), weights='imagenet', include_top=False)

    for layer in xCeption.layers:
        layer.trainable = False

    x = xCeption.output
    x = Flatten(name='avg_pool_2D')(x)
    x = Dense(256,activation='relu')(x)
    x = Dropout(0.1)(x)
    predictions = Dense(7, activation='softmax')(x)

    model = Model(xCeption.input, predictions)

    model.compile(optimizer=Nadam(0.0001),
                  loss='categorical_crossentropy',  # categorical_crossentropy if multi-class classifier
                  metrics=['accuracy'])

    model, val_acc, history = TreinarModelo(trainGenerator,valGenerator, model, indexEpochs)

    return model, val_acc, history

# -------------------
# - Cria a DenseNet -
# -------------------

def evaluate_modelDenseNet(trainGenerator, valGenerator, indexEpochs):

    print("Criando DenseNet")

    densenet = DenseNet121(weights='imagenet', include_top=False, input_shape=(224,224,3))

    for layer in densenet.layers:
        layer.trainable = False

    model = Sequential()
    model.add(densenet)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(7, activation='softmax'))
    model.summary()

    model.compile(optimizer= Nadam(0.0001),
                  loss='categorical_crossentropy',  # categorical_crossentropy if multi-class classifier
                  metrics=['accuracy'])

    model, val_acc, history = TreinarModelo(trainGenerator,valGenerator, model, indexEpochs)

    return model, val_acc, history


# --------------------
# - Cria a GoogleNet -
# --------------------

def evaluate_modelGoogleNet(trainGenerator, valGenerator, indexEpochs):

    print("Criando GoogleNet")

    googlenet_base = tf.keras.applications.InceptionV3(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    x = googlenet_base.output
    x = Flatten(name='avg_pool_2D')(x)
    x = Dense(256,activation='relu')(x)
    x = Dropout(0.1)(x)
    predictions = Dense(7,activation='softmax')(x)
    model = Model(inputs=googlenet_base.input, outputs=predictions)

    for layer in googlenet_base.layers:
        layer.trainable = False

    # model.load_weights("best_model.hdf5")

    model.compile(optimizer = Nadam(0.0001) , loss = 'categorical_crossentropy', metrics=["accuracy"])

    # model.compile(loss='categorical_crossentropy',
    #               optimizer=optimizers.SGD(lr=1e-4,
    #                                        momentum=0.9),
    #               metrics=['accuracy'])

    # model.compile(loss='categorical_crossentropy',
    #               optimizer=optimizers.RMSprop(lr=2e-5),
    #               metrics=['accuracy'])

    model, val_acc, history = TreinarModelo(trainGenerator,valGenerator, model, indexEpochs)

    return model, val_acc, history

# ------------------------
# - Continua a GoogleNet -
# ------------------------

def ContinuaGoogleNet(trainGenerator, valGenerator, indexEpochs):

    print("Continua ResNet")

    googleNet = keras.models.load_model('GoogleNet10EpochsK-Folds1.tf')
    print("Modelo carregado!")

    googleNet.summary()

    googleNet.compile(optimizer = Nadam(0.0001) , loss = 'categorical_crossentropy', metrics=["accuracy"])

    model, val_acc, history = TreinarModelo(trainGenerator,valGenerator, googleNet, indexEpochs)

    return model, val_acc, history

# -------------------
# - Treina o modelo -
# -------------------

def TreinarModelo(trainGenerator, valGenerator, modelToTrain, indexEpochs):

    print("Inicio Treinamento")

    bestepoch = "best_model" + str(indexEpochs) + ".hdf5"

    checkpoint = ModelCheckpoint(bestepoch, monitor='val_accuracy', verbose=1,
                                 save_best_only=True, mode='auto', period=1)

    history = modelToTrain.fit(trainGenerator,
                        validation_data=valGenerator,
                        epochs=30,
                        verbose=1,
                        callbacks=[checkpoint])

    _, val_acc = modelToTrain.evaluate(X_val, y_val, verbose=1)

    return modelToTrain, val_acc, history

# -----------------------
# - Predição de imagens -
# -----------------------

def PredizerImagem(model, X_val, y_val):

    pred = model.predict(X_val)
    pred = np.argmax(pred, axis=1)
    pred = pd.DataFrame(pred).replace(
        {0: 'chokkan', 1: 'fukunagashi',2: 'han_kengai', 3: 'kengai', 4: 'literatti', 5: 'moyogi', 6: 'shakan'})

    y_val_string = np.argmax(y_val, axis=1)
    y_val_string = pd.DataFrame(y_val_string).replace(
        {0: 'chokkan', 1: 'fukunagashi',2: 'han_kengai', 3: 'kengai', 4: 'literatti', 5: 'moyogi', 6: 'shakan'})

    confusion_matrix = metrics.confusion_matrix(y_true=y_val_string, y_pred=pred)

    print(confusion_matrix)

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

    print('O modelo errou: ' + str(len(mis_class1)))

    count = 0
    fig, ax = plt.subplots(3, 2)
    fig.set_size_inches(5, 5)
    if len(mis_class)  > 6:
        for i in range(3):
            for j in range(2):
                ax[i, j].imshow(X_val[mis_class[count]][:, :, ::-1])
                ax[i, j].set_title("Predicted style : " + str(pred[0][mis_class[count]]) + "\n" + "Actual style : " + str(
                    y_val_string[0][mis_class[count]]))
                plt.tight_layout()
                count += 1

    plt.show()

    return True

# --------------------------
# - Monta confusion matrix -
# --------------------------

def montarConfusionMatrix10Folds(model, X_val, y_val):

    pred = model.predict(X_val)
    pred = np.argmax(pred, axis=1)
    pred = pd.DataFrame(pred).replace(
        {0: 'chokkan', 1: 'fukunagashi',2: 'han_kengai', 3: 'kengai', 4: 'literatti', 5: 'moyogi', 6: 'shakan'})

    y_val_string = np.argmax(y_val, axis=1)
    y_val_string = pd.DataFrame(y_val_string).replace(
        {0: 'chokkan', 1: 'fukunagashi',2: 'han_kengai', 3: 'kengai', 4: 'literatti', 5: 'moyogi', 6: 'shakan'})

    return y_val_string, pred

# ------------------
# - Abre o DataSet -
# ------------------

def AbreDataSet():
    chokkan_dir = glob.glob(os.path.join('Chokkan/', '*'))
    fukunagashi_dir = glob.glob(os.path.join('Fukunagashi/', '*'))
    han_kengai_dir = glob.glob(os.path.join('Han-kengai/', '*'))
    kengai_dir = glob.glob(os.path.join('Kengai/', '*'))
    literatti_dir = glob.glob(os.path.join('Literatti/', '*'))
    moyogi_dir = glob.glob(os.path.join('Moyogi/', '*'))
    shakan_dir = glob.glob(os.path.join('Shakan/', '*'))

    # Compilando todos os caminhos.
    X_path = chokkan_dir + fukunagashi_dir + han_kengai_dir + kengai_dir + literatti_dir + moyogi_dir + shakan_dir

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
    l_han_kengai = 2 * np.ones(len(han_kengai_dir))
    l_kengai = 3 * np.ones(len(kengai_dir))
    l_literatti = 4 * np.ones(len(literatti_dir))
    l_moyogi = 5 * np.ones(len(moyogi_dir))
    l_shakan = 6 * np.ones(len(shakan_dir))

    y = np.concatenate((l_chokkan, l_fukunagashi, l_han_kengai, l_kengai, l_literatti, l_moyogi, l_shakan))

    # Finalização da categorização.
    y = to_categorical(y, 7)

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

# -----------------------------------------------------
# - Imprime exemplos de estilos com data augmentation -
# -----------------------------------------------------

def ImprimirDataAugmentation(X_train, y_train):
    train_generator1 = train_datagen.flow(X_train[15:16], y_train[15:16], batch_size=1)
    examples = [next(train_generator1) for i in range(0,6)]
    fig, ax = plt.subplots(2,2, figsize=(15, 15))
    a = 1;
    for i in range(2):
        for j in range(2):
            ax[i][j].imshow(examples[a][0][0])
            a = a + 1

    plt.show()

#--------------------------------
#- Plota grafico de treinamento -
#--------------------------------

def PlotarGrafico(indexTrain):
    OldHistory = pickle.load(open('trainHistory' + str(indexTrain), 'rb'))

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

#-------------------------------------
#- Plota o grafico de todos os folds -
#-------------------------------------

def PlotarGraficoTodosKfolds():

    for indexKfolds in range(10):
        OldHistory = pickle.load(open('trainHistory' + str(indexKfolds+1), 'rb'))

        plt.plot(OldHistory['accuracy'])

    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['kfolds-1', 'kfolds-2', 'kfolds-3', 'kfolds-4', 'kfolds-5', 'kfolds-6', 'kfolds-7', 'kfolds-8', 'kfolds-9', 'kfolds-10'])
    plt.show()



    for indexKfolds in range(10):
        OldHistory = pickle.load(open('trainHistory' + str(indexKfolds+1), 'rb'))

        plt.plot(OldHistory['val_accuracy'])

    plt.title('Model Val_Accuracy')
    plt.ylabel('Val_Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['kfolds-1', 'kfolds-2', 'kfolds-3', 'kfolds-4', 'kfolds-5', 'kfolds-6', 'kfolds-7', 'kfolds-8', 'kfolds-9', 'kfolds-10'])
    plt.show()


    for indexKfolds in range(10):
        OldHistory = pickle.load(open('trainHistory' + str(indexKfolds+1), 'rb'))

        plt.plot(OldHistory['loss'])

    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['kfolds-1', 'kfolds-2', 'kfolds-3', 'kfolds-4', 'kfolds-5', 'kfolds-6', 'kfolds-7', 'kfolds-8', 'kfolds-9', 'kfolds-10'])
    plt.show()

    for indexKfolds in range(10):
        OldHistory = pickle.load(open('trainHistory' + str(indexKfolds+1), 'rb'))

        plt.plot(OldHistory['val_loss'])

    plt.title('Model Loss')
    plt.ylabel('Val_Loss')
    plt.xlabel('Epochs')
    plt.legend(['kfolds-1', 'kfolds-2', 'kfolds-3', 'kfolds-4', 'kfolds-5', 'kfolds-6', 'kfolds-7', 'kfolds-8', 'kfolds-9', 'kfolds-10'])
    plt.show()

#----------------------------
#- Monta matriz de confusao -
#----------------------------

def montarMatriz():
    model = keras.models.load_model('GoogleNet10EpochsK-Folds1.tf')
    model.load_weights('best_model1.hdf5')

    validacaoAccuracy = []

    X,y = AbreDataSet()

    for train_index, test_index in kf.split(X,y.argmax(1)):
        indexEpoch = indexEpoch + 1
        print('-----------------------------------------------------------------')
        print('Fold ' + str(indexEpoch))

        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = y[train_index], y[test_index]

        y_val_stringGlobal, predGlobal = montarConfusionMatrix10Folds(model, X_val, y_val)

        loss1, accuracy1 = model.evaluate(X_val, y_val, steps = 20)

        validacaoAccuracy.append(accuracy1)

        print("-------- Precisão k-folds 1 ---------")
        print("Initial loss: {:.2f}".format(loss1))
        print("Initial accuracy: {:.2f}".format(accuracy1))
        print("-------------------------------------")


    confusion_matrix = metrics.confusion_matrix(y_true=predGlobal, y_pred=y_val_stringGlobal, labels=['chokkan','fukunagashi','han_kengai','kengai','literatti','moyogi','shakan'])

    import seaborn as sn
    import matplotlib.pyplot as plt

    df_cm = pd.DataFrame(confusion_matrix, index=['chokkan','fukunagashi','han_kengai','kengai','literatti','moyogi','shakan'] , columns=['chokkan','fukunagashi','han_kengai','kengai','literatti','moyogi','shakan'])
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm/np.sum(df_cm), annot=True, annot_kws={"size": 12}, cmap='Greens') # font size

    plt.show()

#---------------------------
#- Começa aqui o algoritmo -
#---------------------------

PlotarGraficoTodosKfolds()
montarMatriz()

# inceptionv3 = keras.models.load_model('GoogleNet10EpochsK-Folds1.tf')
# print("Modelo carregado!")
#
# X1,y1 = AbreDataSet()
# X_train1, X_val1, y_train1, y_val1 = ReparteDataSet(X1,y1,421)
#
# PredizerImagem(inceptionv3,X_val1, y_val1)
# VerificarPrecisao(inceptionv3,X_val1,y_val1)
# PlotarGrafico(1)



train_datagen = ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.1,  # Aleatory zoom
    width_shift_range=0.1,  # horizontal shift
    height_shift_range=0.1,  # vertical shift
    horizontal_flip=True,
    brightness_range=[0.2,1.0],
    shear_range=0.1)

val_datagen = ImageDataGenerator()

n_folds = 10
cv_scores = list()
X,y = AbreDataSet()

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
kf.get_n_splits(X)

print(kf)

indexEpoch = 0
for train_index, test_index in kf.split(X,y.argmax(1)):
    indexEpoch = indexEpoch + 1
    print('-----------------------------------------------------------------')
    print('Fold ' + str(indexEpoch))

    X_train, X_val = X[train_index], X[test_index]
    y_train, y_val = y[train_index], y[test_index]

    train_generator = train_datagen.flow(X_train, y_train, batch_size=20)
    val_genarator = val_datagen.flow(X_val, y_val, batch_size=20)
    # evaluate model
    model, test_acc, history = evaluate_modelGoogleNet(train_generator, val_genarator, indexEpoch)

    print('>%.3f' % test_acc)
    cv_scores.append(test_acc)

    Variavel = "GoogleNet10EpochsK-Folds"
    Final = ".tf"
    VariavelFinal = Variavel + str(indexEpoch) + Final

    model.save(VariavelFinal)
    print("Modelo salvo com sucesso!")

    with open('trainHistory' + str(indexEpoch), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    print("Histórico de treino salvo com sucesso!")

    print('-----------------------------------------------------------------')

print('Estimated Accuracy %.3f (%.3f)' % (np.mean(cv_scores), np.std(cv_scores)))

