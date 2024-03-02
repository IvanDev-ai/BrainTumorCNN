import cv2
import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalMaxPooling2D, Dense, Dropout, BatchNormalization, LeakyReLU

# Parámetros ajustados
img_size = 224
num_filters = 64
kernel_size = (3, 3)
batch_size = 128
epochs = 30

# Directorios de entrenamiento y prueba
test_dir = 'Testing'
train_dir = 'Training'

# Etiquetas de las clases
labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

datagen = ImageDataGenerator(
    rotation_range=20,      # Rango de rotación aleatoria (en grados)
    width_shift_range=0.1,  # Rango de desplazamiento horizontal aleatorio (como fracción del ancho total)
    height_shift_range=0.1, # Rango de desplazamiento vertical aleatorio (como fracción de la altura total)
    zoom_range=0.1,         # Rango de zoom aleatorio
    horizontal_flip=True,   # Volteo horizontal aleatorio
    fill_mode='nearest'     # Estrategia de relleno para los píxeles fuera de los límites de la imagen
)

def load_data(path, labels):
    x, y = [], []
    for i in labels:
        folder_path = os.path.join(path, i)
        for j in os.listdir(folder_path):
            img = cv2.imread(os.path.join(folder_path, j))
            img = cv2.resize(img, (img_size, img_size))
            x.append(img)
            y.append(i)
    return np.array(x), np.array(y)

def OneHotEncoding(labels, y):
    y_new = []
    for i in y:
        y_new.append(labels.index(i))
    return tf.keras.utils.to_categorical(y_new)

def create_model():
    input = Input((img_size, img_size, 3))
    x = BatchNormalization()(input)
    x = Conv2D(num_filters, kernel_size, activation="relu",kernel_regularizer=regularizers.L2(0.01))(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(num_filters * 2, kernel_size, activation="relu",kernel_regularizer=regularizers.L2(0.01))(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(num_filters * 4, kernel_size, activation="relu",kernel_regularizer=regularizers.L2(0.01))(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(num_filters * 8, kernel_size, activation="relu", kernel_regularizer=regularizers.L2(0.01))(x)
    x = MaxPooling2D((2, 2))(x)
    x = GlobalMaxPooling2D()(x)
    x = Dropout(0.6)(x)
    x = Dense(num_filters * 2,kernel_regularizer=regularizers.L2(0.01))(x)
    output = Dense(len(labels), activation='softmax')(x)
    model = Model(input, output)
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])
    return model

def train_model(X_train, Y_train):
    model = create_model()
    tensorboard = TensorBoard(log_dir='./logs')
    checkpoint = ModelCheckpoint("BrainTumorCNN", monitor="val_accuracy", save_best_only=True, mode='auto', verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor="val_accuracy", factor=0.3, patience=2, min_delta=0.001, mode="auto", verbose=True)
    train_generator = datagen.flow(X_train, Y_train, batch_size=batch_size)
    history = model.fit(train_generator, steps_per_epoch=len(X_train) // batch_size, 
                        validation_data=(x_test, y_test), 
                        epochs=epochs, verbose=1, 
                        callbacks=[tensorboard, checkpoint, reduce_lr])

x_train, y_train = load_data(train_dir, labels)
x_test, y_test = load_data(test_dir, labels)
y_train = OneHotEncoding(labels, y_train)
y_test = OneHotEncoding(labels, y_test)

train_model(x_train, y_train)