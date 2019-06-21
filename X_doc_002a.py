from os import listdir
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from data.X_doc_tools import *
import pandas as pd

inp_path = './input/lis/'
#inp_path = './input/Cardiomegaly/'


files_list = listdir(inp_path)


#building model structure
if True:
    input_shape = (365, 365, 3)
    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=input_shape, activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy')


for model_name in listdir('./models/.'):
    invertion = int(model_name.split('_')[-1][:-3])
    diagnosis = model_name.split('_')[0]

    print(model_name, invertion)
    model.load_weights('./models/{}'.format(model_name))


    for file in files_list:
        file_path = inp_path + file
        P, im, hm = heat_mapper(model=model, img_path=file_path, inversion=invertion)


        draw_report2(P,img=im, hybrid=hm, file=file, model_name=model_name, diagnosis=diagnosis, save_mode=1)



