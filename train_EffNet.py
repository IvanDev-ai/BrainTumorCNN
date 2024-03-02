import cv2
import os
import numpy as np
import tensorflow as tf
from keras.applications import EfficientNetB0
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from keras.models import Model
from keras.layers import  GlobalAveragePooling2D, Dense, Dropout

# Parámetros ajustados
img_size = 150
batch_size = 32
epochs = 12

# Directorios de entrenamiento y prueba. Datos usados: https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri
test_dir = 'Testing'
train_dir = 'Training'

# Etiquetas de las clases
labels = ['glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor']


def  load_data(path, labels):
    x=[]
    y=[]
    for i in labels:
        folder_path = os.path.join(path, i)
        for j in os.listdir(folder_path):
            img = cv2.imread(os.path.join(folder_path,j))
            img = cv2.resize(img,(img_size,img_size))
            x.append(img)
            y.append(i)

    return np.array(x),np.array(y)

def OneHotEncoding(labels,y):
    y_new = []
    for i in y:
        y_new.append(labels.index(i))
    return tf.keras.utils.to_categorical(y_new)

def  create_model():
    effnet = EfficientNetB0(weights='imagenet',include_top=False,input_shape=(img_size,img_size,3))
    model = effnet.output
    model = GlobalAveragePooling2D()(model)
    model = Dropout(rate=0.5)(model)
    model = Dense(4,activation='softmax')(model)
    model = Model(inputs=effnet.input, outputs = model)
    model.compile(loss='categorical_crossentropy', 
                optimizer='adam', 
                metrics=['accuracy'])
    return model

def  train_model(X_train, Y_train):
    model = create_model()
    tensorboard = TensorBoard(log_dir='./logs')
    checkpoint = ModelCheckpoint("BrainTumorCNN",monitor="val_accuracy",save_best_only=True,mode='auto',verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor="val_accuracy",factor= 0.3, patience= 2,min_delta=0.001,mode="auto", verbose= True)
    history = model.fit(X_train,Y_train,batch_size=batch_size,validation_split=0.1,epochs=epochs,verbose=1, callbacks=[tensorboard,checkpoint,reduce_lr])
    loss, accuracy = model.evaluate(x_test, y_test)
    print("Loss:", loss)
    print("Accuracy:", accuracy)
    return model

def prediction(model):
    predicitons = model.predict(x_test)
    predicitons = np.argmax(predicitons,axis=1)
    return predicitons

def save_model(model):
    model.save("BrainTumor.h5")
    return "Model saved succesfully"
x_train,y_train = load_data(train_dir, labels)
x_test,y_test = load_data(test_dir, labels)
y_train = OneHotEncoding(labels,y_train)
y_test = OneHotEncoding(labels,y_test)

trained_model = train_model(x_train,y_train)
print(prediction(trained_model))
save_model(trained_model)